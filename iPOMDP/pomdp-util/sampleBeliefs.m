function S = sampleBeliefs(problem, n, alg, maxStagnantCount, useEye, useSet)
% S = sampleBeliefs(problem, n, alg, maxStagnantCount, useEye)
% sampleBeliefs - sample beliefs from a POMDP following PBVI algorithm;
% except that we will sample all the beliefs at once instead of
% interleaving belief addition and value updates
%     n = max number of beliefs
%     S = (n x d) the sampled beliefs
% note: if using the one-step trajectory approach, runtimes may be long if
%    many attempts are needed to create the desired number of unique points!
% note: should initialize problem using initProblem elsewhere (previous)
% note: do NOT useEye (that is, seed the initial belief set with the
%    identity matrix) if you're using pbvi-far -- it won't find far points


% read initial state from problem file (should be specified as 'start')
if nargin == 2
    alg = 'pbvi-rand';
    maxStagnantCount = 100;
    useEye = false;
    useSet = [];
elseif nargin == 3
    maxStagnantCount = 100;
    useEye = false;
    useSet = [];
elseif nargin == 4
    useEye = false;
    useSet = [];
elseif nargin == 5
    useSet = [];
end

if useEye
    S = eye( length( problem.start ) );
    S = [S; problem.start];
else
    S(1,:) = problem.start;
end

if size(useSet,1) > 0
    S = useSet;
end

% in some cases, it may not be possible to add all the beliefs that we
% want.  in this case we (a) limit iters and (b) quit if sample set size
% does not increase for maxStagnate rounds
maxIterCount = 200;
diffeps = .05;

% some useful caching
for action = 1:problem.nrActions
    if iscell( problem.observation )
        problem.action_observation{ action } = ...
            transpose( squeeze( problem.observation{ action } ) );
    else
        problem.action_observation{ action } = ...
            transpose( squeeze( problem.observation(:,action,:)));
    end
end
problem.random_number_index = 1;
problem.random_number_count = 1000;
problem.random_number_set = rand( 1 , problem.random_number_count );

% get a new set of beliefs to add
iterCount = 1;
stagnant = 0;
setSize = size(S,1);
Sorig = S;
newSet = S;
while size(S,1) <= n && iterCount < maxIterCount && stagnant < maxStagnantCount 
    
    % add new samples based on the algorithm
    switch alg
        case 'rand-dirichlet'
            newSet = addRandDirichlet(problem, n);
        case 'rand-unif'
            newSet = addRandUnif(problem, n);
        case 'pbvi-far'
            [ newSet problem ] = addPBVIFar(problem, S);
        case 'pbvi-rand'
            [ newBels problem ] = addPBVIRand(problem, Sorig); newSet = newBels;
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];
            [ newBels problem ] = addPBVIRand(problem, newBels); newSet = [newSet; newBels];            
        case 'pbvi-all'
            newSet = addPBVIAll(problem, newSet);
    end


    % concatinate the new beliefs and remove repeats
    % newSet = setdiff( newSet, S, 'rows'); % finds unique only
    newSet = epSetDiff( newSet, S, diffeps); % finds unique only
    totalCount = size(S,1) + size(newSet,1);
    if totalCount > n
        newCount = n - size(S,1);
        if ~isempty( newSet )
            ind = randperm(size(newSet,1));
            ind = ind(1:newCount);
            newSet = newSet(ind,:);
            S = [S ; newSet ];
        end
        break;
    else    
        S = [S ; newSet ];
    end
    iterCount = iterCount + 1;
    
    % check for stagnation
    if size(S,1) == setSize
        stagnant = stagnant + 1;
    end
    setSize = size(S,1);

end
    
% -----------------------------------------------------------------------
function dbl = addRandUnif(problem, n)
    % creates n uniformly (indep per state prob) random belief vectors
    dbl = rand( problem.nrStates, n );
    dblsum = sum(dbl);
    dbl = dbl ./ repmat( dblsum, problem.nrStates, 1 );
    dbl = transpose(dbl);

% -----------------------------------------------------------------------
function dbl = addRandDirichlet(problem, n)
    % creates n uniform dirichlet random belief vectors
    dbl = sample_dirichlet( ones(1,problem.nrStates), n );
       
% -----------------------------------------------------------------------
function dbl = addPBVIAll( problem, S)
    % samples |a||o| additional beliefs for each belief in the set
    newS = S;
    dbl = [];
    
    for action = 1:problem.nrActions 
        if iscell( problem.transition )
            actionSet = problem.transition{action} * newS';
        else
            actionSet = problem.transition(:,:,action) * newS';
        end
        for obs = 1:problem.nrObservations
            
            % update probability for each observation
            if iscell( problem.observation )
                obsSet = actionSet .* ...
                    repmat( problem.observation{action}(:,obs), 1, size(newS,1) );
            else
                obsSet = actionSet .* ...
                    repmat( problem.observation(:,action,obs), 1, size(newS,1) );
            end
            obsSum = sum(obsSet);
            warning off;
            obsSet = obsSet ./ repmat( obsSum, problem.nrStates, 1 );
            warning on;
            obsSet = obsSet';
            
            % remove any NaN rows
            obsSet(any(isnan(obsSet),2),:) = [];
            
            % concatinate to the dbl list
            dbl = [dbl; obsSet];
        end
    end
                    
% -----------------------------------------------------------------------
function [ dbl problem ] = addPBVIRand(problem, S)
    % samples one additional belief for each belief in the set
    % pick a random belief to add after action is taken
    
    % get all the transitions
    for i = 1:problem.nrActions 
        if iscell( problem.transition )
            propSet(:,:,i) = problem.transition{i} * S';
        else
            propSet(:,:,i) = problem.transition(:,:,i) * S';
        end
        
        % sample an observation and update belief
        for j = 1:size(S,1)
            [ propSet(:,j,i) problem ] = sampleObsAndUpdate(problem,  propSet(:,j,i), i );
        end
    end

    % pick one randomly
    avec = ceil( problem.nrActions * rand( size(S,1), 1 ) );
    for i = 1:size(S,1)
        dbl(i,:) = transpose( propSet(:,i, avec(i)  )  );
    end
    
% -----------------------------------------------------------------------
function [ dbl problem ] = addPBVIFar(problem, S)
    % samples one additional belief for each belief in the set
    % as per PBVI, try each action, take farthest belief
    
    % get all the transitions
    for i = 1:problem.nrActions 
        if iscell( problem.transition )
            propSet(:,:,i) = problem.transition{i} * S';
        else
            propSet(:,:,i) = problem.transition(:,:,i) * S';
        end
        
        % sample an observation and update belief
        for j = 1:size(S,1)
            [ propSet(:,j,i) problem ] = sampleObsAndUpdate(problem,  propSet(:,j,i), i );
        end
        
        % compute distance to new belief
        propDist(:,i) = sum( (propSet(:,:,i) - S' ).^2 );
    end
        
    % pick one with farthest distance
    [maxVal maxInd] = max( propDist' );
    for i = 1:size(S,1)
        dbl(i,:) = transpose( propSet(:,i, maxInd(i) )  );
    end
    
% -----------------------------------------------------------------------
% samples observation and updates belief state
% bel, prevBel are COLUMN vectors
function [ bel problem ] = sampleObsAndUpdate(problem,  prevBel, action )
    [ obs problem ] = sampleObs( problem, prevBel, action );
    if iscell( problem.observation )
        bel = prevBel .* problem.observation{ action }( : , obs);
    else
        bel = prevBel .* problem.observation(:,action,obs);
    end
    bel = bel/sum(bel);
    
% samples observation given a belief state and action
% bel is a COLUMN vector
function [ obs problem ] = sampleObs( problem ,  bel , action )
    pdf = problem.action_observation{ action } * bel;
    obs = sample_multinomial( pdf , 1 , problem.random_number_set( problem.random_number_index ) );
    problem.random_number_index = problem.random_number_index + 1;
    if problem.random_number_index > problem.random_number_count
        problem.random_number_index = 1;
    end
    
% -----------------------------------------------------------------------
% epsilon l1 distance then same
function c = epSetDiff(A,B,ep);

     A = unique(A,'rows');
    [c,ndx] = sortrows([A;B]);
    [rowsC,colsC] = size(c);
    if rowsC > 1 && colsC ~= 0
      % d indicates the location of non-matching entries
      % d = c(1:rowsC-1,:) ~= c(2:rowsC,:);
      d = sum(  abs(  c(1:rowsC-1,:) - c(2:rowsC,:)  ), 2  ) > ep;
    else
      d = zeros(rowsC-1,0);
    end
    % d = any(d,2);
    d(rowsC,1) = 1;   % Final entry always included.
    
    % d = 1 now for any unmatched entry of A or of B.
    n = size(A,1);
    d = d & ndx <= n; % Now find only the ones in A.
    
    c = c(d,:);
    