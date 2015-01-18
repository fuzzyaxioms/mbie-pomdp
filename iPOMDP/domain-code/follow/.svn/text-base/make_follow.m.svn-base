function problem = make_follow

% actions are none n s e w
people_transition{ 1 } = [ .3 .4 .2 .05 .05  ];
% people_transition{ 2 } = [ .1 .05 .8 .03 .02 ];
people_count = numel( people_transition );

% basic parameters
grid_count = 26;
problem.nrStates = people_count * grid_count;
problem.nrObservations = 6;
problem.nrActions = 5;
problem.gamma = .95;
problem.start = zeros( 1 , problem.nrStates );
for people_index = 1:people_count
    problem.start( 1 + grid_count * ( people_index - 1 ) ) = 1 / people_count;
end
problem.start_dist = problem.start';

% reward parameters
same_r = 1;
near_r = 0;
far_r = -1;
out_r = -20;

% observation parameters (same1, n2, s3, e4, w5, out6)
p_obs = 0.8;

% matrices
problem.reward = zeros( problem.nrStates , problem.nrActions );
problem.transition = zeros( problem.nrStates , problem.nrStates , problem.nrActions );
problem.observation = zeros( problem.nrStates , problem.nrActions , problem.nrObservations );
problem.observation( : , : , end ) = 1 - p_obs;
for people_index = 1:people_count
    my_t = people_transition{ people_index };
    for x = 1:5
        for y = 1:5
            curr_state = xy_to_state( x , y ) + ( people_index - 1 ) * grid_count;
            
            % set obs
            if ( ( x == 3 ) && ( y == 3 ) )
                problem.observation( curr_state , : , 1 ) = p_obs;
            elseif ( abs( x - 3 ) >= abs( y - 3 ) )
                if x < 3
                    problem.observation( curr_state , : , 5 ) = p_obs; 
                else
                    problem.observation( curr_state , : , 4 ) = p_obs; 
                end
            else
                if y < 3
                    problem.observation( curr_state , : , 2 ) = p_obs; 
                else
                    problem.observation( curr_state , : , 3 ) = p_obs; 
                end
            end
            
            % set reward
            if ( ( x == 3 ) && ( y == 3 ) )
                problem.reward( curr_state , : ) = same_r;
            elseif ( ( x > 1 ) && ( x < 5 ) && ( y > 1 ) && ( y < 5 ) )
                problem.reward( curr_state , : ) = near_r;
            else
                problem.reward( curr_state , : ) = far_r;
            end
            
            % set transitions
            for action = 1:problem.nrActions
               
                % apply the effect of your movement
                [ x_mid y_mid ] = apply_action( x , y , action );
                
                % consider all states that the people could move to
                for people_action = 1:problem.nrActions
                    [ x_next y_next ] = apply_action( x_mid , y_mid , people_action );
                    next_state = xy_to_state( x_next , y_next );
                    problem.transition( next_state , curr_state , action ) = ...
                        problem.transition( next_state , curr_state , action ) + ...
                        my_t( people_action );
                end
            end
        end
    end
    
    % set obs/reward/trans for being out of bounds
    curr_state = people_index * grid_count;
    problem.reward( curr_state , : ) = out_r;
    problem.observation( curr_state , : , end ) = 1;
    for action = 1:problem.nrActions
        problem.transition( : , curr_state , action ) = problem.start_dist;
    end
end
problem.maxReward = max( problem.reward(:) );

save follow problem


% ----------------------------------------------------------------------- %
function state = xy_to_state( x , y )
% x is how many squares across from the top (1 2 3 4 5) -- col number; y is
% how many squares down from the top (1 2 3 4 5) -- row number
if ( ( x <= 5 ) && ( y <= 5 ) && ( x >= 1 ) && ( y >= 1 ) )
    state = x + 5 * ( y - 1 );
else
    state = 26;
end

% ----------------------------------------------------------------------- %
function [x y] = apply_action( x , y , action )
% none n s e w
if action == 2
    y = y - 1;
elseif action == 3
    y = y + 1;
elseif action == 4
    x = x + 1;
elseif action == 5
    x = x - 1;
end


