function [TAO vL backupStats] = runPBVILean(problem,S,h,epsilon)
% function [TAO vL backupStats] = runPBVILean(problem,S,h,epsilon)
% runs the point-based value iteration algorithm -- Lean version does not
% keep all of the old history; overwrites alpha vectors. (So iter is 1.)
%  S - set of beliefs
%  h - number of backups to perform
%  epsilon - convergence value
% to enable tree structuring later, we currently do NOT prune repeat
% vectors in the alpha set, instead keeping a vector per belief... if this
% is not needed in the future, vectors can easily be pruned using matlab's
% 'unique' or 'union' functions
%
% backup stats stores our solution and intermediate computations
%   backupStats.Q{ iter, belief number }(:, action ) = gamma^a_b 
%   backupStats.Qo{ iter, belief number, action,obs } = ind of maximizing
%       alpha vectors from the previous iteration in decreasing order
%       (currently there are 3 total stored)
%   backupStats.V{ iter }.v( bel number ) = alphaindex; alphalist(:,i) is vec
%   backupStats.Vtable{ iter }.alphaList( :,i ) = unique alpha vector list
%   backupStats.Vtable{ iter }.alphaUserSet{i} = bels that use alphalist(:,i)
%   backupStats.Vtable{ iter }.alphaAction(i) = action for alpha i
%   backupStats.Vtable{ iter }.alphaUse(i) = when alpha i was last used
% all vectors are COLUMN VECTORS

% check for epsilon
useEps = 0;
if nargin == 4
    useEps = 1;
end

% intermediate useful stuffies
vL = [];
TAO = createTAO( problem );

% initialize the q and alpha vectors
% Q(i,:,k) is the q-vector for action k in belief i
beliefCount = size(S,1);

% FAST VERSION : first permute Q into dimensions 2 and 3 (tip on side),
% then replicate slice for every belief
Q = reshape( full( problem.reward ) , [1 problem.nrStates problem.nrActions] );
Q = repmat( Q, [beliefCount 1 1 ] );

% SLOW VERSION
% for i = 1:beliefCount;
%         Q(i,:,:) = problem.reward;
% end
backupStats = findBestQLean(problem,S,Q);

% debug -- progress reporter
[a1 v1] = getAction( S, backupStats.Vtable );

% perform dynamic programming backups
for i = 2:h
    backupStats = backupQ(problem, backupStats, S, TAO);
    
    % convergence/progress reporter
    if useEps
        [a2 v2] = getAction( S, backupStats.Vtable );
        conver = sum(abs(v2-v1)); vL(:,i) = v1; v1 = v2;
        if conver < epsilon
            return;
        end
        % disp(['Completed PBVI iter ' num2str(i) ' conv ' num2str(conver)  ]);
    else
        % disp(['Completed PBVI iter ' num2str(i)]);
    end
end

% initialize to when the alphaList was initiated
backupStats.Vtable{end}.alphaUse = cputime * ones(length(backupStats.Vtable{end}.alphaAction),1);

%------------------------------------------------------------------------
function TAO = createTAO( problem )
% when TAO{action}{obs} is multiplied by backupStats.Vtable{1}.alphaList,
% we get gammaAO
     TAO = cell(problem.nrActions,problem.nrObservations);    
     for action = 1:problem.nrActions
        for obs = 1:problem.nrObservations
     
            % compute transition matrix                        
            if iscell( problem.transition )
                tprob = problem.transition{action};
            else
                tprob = problem.transition(:,:,action);
            end
            if iscell( problem.observation )
                oprob = problem.observation{ action }( : , obs );
            else   
                oprob = problem.observation(:,action,obs);
            end
            
            for i = 1:problem.nrStates
                tprob(:,i) = tprob(:,i) .* (oprob);
            end
            tprob = tprob * problem.gamma;
            TAO{action}{obs} = tprob';
        end
    end

%------------------------------------------------------------------------
function backupStats = backupQ( problem, backupStats, S, TAO)
    % computes the Q vectors for the next backup
    alphaCount = size(backupStats.Vtable{1}.alphaList,2);
    
    % first compute gammoAO, (return if are in state s, see o, do a)
    gammaAO = cell(problem.nrActions,1);
    for action = 1:problem.nrActions
        for obs = 1:problem.nrObservations
            gammaAO{action}{obs} = TAO{action}{obs} * backupStats.Vtable{1}.alphaList;
        end
    end
    
    % next pick q vectors for each belief
    for action = 1:problem.nrActions
        
        % add expected return for each obs + action
        gammaAB = repmat( problem.reward(:,action)  ,1,size(S,1)  );
        for obs = 1:problem.nrObservations
            [ vals inds ] = max( S * gammaAO{action}{obs},[],2 );
            gammaAB = gammaAB + gammaAO{action}{obs}(:, inds);
        end
        Q(:,:,action) = gammaAB';
    end
    
    
    % update the V
    backupStats = findBestQLean(problem, S,Q);
    backupStats.Q = Q;
    
    % get rid of huge gamma structure
    clear gammaAO;
    
%-------------------------------------------------------------------------
function backupStats = findBestQLean(problem, S,Q)
    % updates backup stats for each q vector, belief
    
    % first determine best alpha vector for each belief
    for ac = 1:problem.nrActions
        vals(:,ac) = sum( Q(:,:,ac) .* S,2 );
    end
    [maxVal action] = max(vals');
    for a = 1:problem.nrActions % select the best actions
        mask(:,:,a) = repmat( transpose(action == a), 1, problem.nrStates );
    end
    v = sum( Q .* mask , 3 );
    a = action';
    clear mask;
    
    % next prune the list to a unique set
    [ alphaList ind1 ind2 ] = unique( v, 'rows' );
    backupStats.Vtable{1}.alphaList = alphaList';
    backupStats.Vtable{1}.alphaAction = a( ind1 );
    
    
