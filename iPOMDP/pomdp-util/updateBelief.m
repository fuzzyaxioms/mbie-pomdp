function bel = updateBelief( pomdp, prev_bel, action, obs );
    % function bel = updateBelief( pomdp, prev_bel, action, obs );
    % updates the belief given a NON-permuted pomdp
    % beliefs should be COLUMN vectors
    % if no observation is given, just does an update based on the
    % transition matrix
    % obs can be a matrix or a single value -- should be a ROW
    
    % check if we're using observations
    useObs = ~isempty( obs );
    
    % update transition
    if iscell( pomdp.transition )
        action_bel = pomdp.transition{action} * prev_bel;
    else
        action_bel = pomdp.transition(:,:,action) * prev_bel;
    end
    bel = action_bel;
    
    % update observation -- single observation case
    if useObs == true
        
        % single update
        if length(obs) == 1
            if iscell( pomdp.observation )
                bel = bel .* pomdp.observation{ action }( : ,obs );
            else
                bel = bel .* pomdp.observation(:,action,obs);
            end
            
        % multiple (expectation) update
        else
            if iscell( pomdp.observation )
                belmat = repmat( bel, 1, pomdp.nrObservations ) .* ...
                    pomdp.observation{ action };
            else
                belmat = repmat( bel, 1, pomdp.nrObservations ) .* ...
                    squeeze( pomdp.observation(:,action,:) );
            end
            belmat = repmat( obs, pomdp.nrStates, 1 ) .* ...
                belmat;
            bel = sum(belmat');
            bel = bel'; 
        end
    end
    
    % normalize -- watch for invalid observations!!
    if sum(bel) == 0
        bel = action_bel;
        disp('updateBelief.m: warning -- invalid observation');
    else
        bel = bel/sum(bel);
    end
        
    
