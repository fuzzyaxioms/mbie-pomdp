function problem = make_harvest_problem( crop_count )
% this is the harvest problem used in constructing states for reinforcement
% learning, mahmud uai 2010: need to set the number of crops; use their
% model in which each crop has 2 phases each
phase_count = 2;
crop_reward = -1;
harvest_reward = 5;
p_right_crop = .7;

% set the number of stuff
problem.nrActions = crop_count + 1;
problem.nrObservations = crop_count;
problem.nrStates = crop_count * ( phase_count )^( crop_count );
problem.gamma = .95;
problem.start = zeros( 1 , problem.nrStates ); problem.start(1) = 1;
problem.start_dist = problem.start';

% do the transitions etc.
bit_count = phase_count ^ crop_count;
bit_vector = phase_count .^ ( ( crop_count:-1:1 ) - 1 );
for state = 1:problem.nrStates
   
    % get the vector-valued state from the state index -- the state index
    % represents which crop was actually changed in the last step as well
    % as the state of all the crops at that time...
    bit_state = 1 + mod( state - 1 , bit_count );    
    obs_state = 1 + floor( ( state - 1 ) / bit_count );
    crop_vector = change_base( bit_state - 1 , 10 , phase_count );
    state_vector = zeros( 1 , crop_count );
    state_vector( ( crop_count - length( crop_vector ) + 1 ):crop_count ) = crop_vector;
    state_vector = state_vector + 1;
    
    % loop over actions
    for action = 1:( crop_count + 1 )
    
        % rewards for harvest action, other
        if action == ( crop_count + 1 )
            problem.reward( state , action ) = sum( harvest_reward * ( state_vector == phase_count ) );
        else
            problem.reward( state , action ) = crop_reward;
        end
        
        % observations
        problem.observation( state , action , obs_state ) = 1;
                
        % transitions
        if action == ( crop_count + 1 )
            next_state_vector = state_vector;
            next_state_vector( state_vector == phase_count ) = 1;
            next_state = bit_count * ( crop_index - 1 ) + ...
                sum( next_state_vector .* bit_vector );
            problem.transition( next_state , state , action ) = 1;
        else
            for crop_index = 1:crop_count
                next_state_vector = state_vector;
                next_state_vector( crop_index ) = next_state_vector( crop_index ) + 1;
                if next_state_vector( crop_index ) > phase_count
                    next_state_vector( crop_index ) = 1;
                end
                next_state_vector( state_vector == phase_count ) = 1;
                next_state = bit_count * ( crop_index - 1 ) + ...
                    sum( next_state_vector .* bit_vector );
                if crop_index == action
                    problem.transition( next_state , state , action ) = p_right_crop;
                else
                    problem.transition( next_state , state , action ) = ...
                        ( 1 - p_right_crop ) / ( crop_count - 1 );
                end
            end
        end
    end
end
problem.maxReward = max( problem.reward(:) );

