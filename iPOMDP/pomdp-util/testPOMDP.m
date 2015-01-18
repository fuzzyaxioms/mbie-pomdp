function [ rvec frvec ivec history_set ] = testPOMDP( sim_pomdp , test_pomdp , ...
    test_sol , repCount , maxIterCount)
% function [ rvec frvec ivec history_set ] = testPOMDP( sim_pomdp , test_pomdp , ...
%   test_sol , repCount )

% loop through tests
rvec = []; frvec = []; ivec = [];
for rep = 1:repCount;

    % set up
    bel = test_pomdp.start';
    reward = 0;
    iterCount = 0;
    state = sample_multinomial( sim_pomdp.start_dist , 1 );
    new_reward = -1;

    % loop until maxed or done
    clear history;  
    while( iterCount < maxIterCount && new_reward < sim_pomdp.maxReward )
    % while( iterCount < maxIterCount );

        % get an action
        if ~isempty( test_sol )
            action = getAction( bel', test_sol.Vtable );
        else
            action = ceil( rand() * sim_pomdp.nrActions );
        end
        history(iterCount+1, 2) = action;
        history(iterCount+1, 4) = state;
        
        % sample a transition and update reward
        new_reward = sim_pomdp.reward( state, action );
        reward = reward + new_reward;
        if iscell( sim_pomdp.transition )
            state = sample_multinomial( sim_pomdp.transition{ action }( :, state ), 1 );
        else   
            state = sample_multinomial( sim_pomdp.transition(:, state, action ), 1 );
        end

        % sample an observation from the current state
        if iscell( sim_pomdp.observation )
            obs = sample_multinomial( sim_pomdp.observation{ action }( state , :), 1 );
        else
            obs = sample_multinomial( sim_pomdp.observation( state, action, :), 1 );
        end
        
        % store history so far
        history(iterCount+1, 1) = obs;
        history(iterCount+1, 3) = new_reward;
 
        % update belief
        bel = updateBelief( test_pomdp , bel, action, obs );
    
        % update the iters
        iterCount = iterCount + 1;
    
        % store stuff
        rvec( iterCount , rep ) = new_reward;
        
        % if the reward is the maximum, reset to the start
        if new_reward == sim_pomdp.maxReward
            bel = test_pomdp.start';
            state = sample_multinomial( sim_pomdp.start_dist , 1 );
            % keep_running = false;
        end
        
    end
 
    % store reward information
    ivec = [ ivec iterCount ];    
    frvec = [ frvec sum( history(:,3) ) ];
    history_set{ rep } = history;
        
end % repCount
