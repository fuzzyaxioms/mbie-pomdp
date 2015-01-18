function problem = create_image_problem

% overall settings
filter_type_count = 2;
scale_count = 3;
pixel_count = ( 2^(scale_count-1) )^2;
filter_count = sum( 4 .^ ( 0:scale_count-1 ) ); 

% reward settings
r_submit_right = 500;
r_submit_wrong = -250;
r_base_cost = [ -1 -3 ];

% obs settings
p_obs = .9;

% auxiliary cache
index = 1;
for scale = 1:scale_count
    for scale_x = 1:( 2 ^ ( scale - 1 ) )
        for scale_y = 1:( 2 ^ ( scale - 1 ) )
            scale_cache( index , : ) = [ scale scale_x scale_y ];
            index = index + 1;
        end
    end
end

% state space corresponds to: where the target is, which filter, which
% scale and position the filter is on, final absorbing state
problem.nrStates = pixel_count * filter_count * filter_type_count + 1;

% the observation space tells which quadrant has the highest signal or
% returns null if the action did not involve running the filter
problem.nrObservations = 5;

% the actions are running a filter, submitting a state, switching a filter,
% moving down one of the quadrants ([1 2; 3 4]), and moving up a scale
problem.nrActions = 1 + 1 + filter_type_count + 4 + 1;
filter_action_set = 3:( 3 + filter_type_count - 1 );
scale_action_set = (filter_action_set(end)+1):(filter_action_set(end)+5);

% other stuffs
problem.start = zeros( 1 , problem.nrStates );
problem.start( 1:pixel_count ) = 1 / pixel_count;
problem.start_dist = problem.start';
problem.gamma = .95;

% matrices
for action = 1:problem.nrActions
    problem.transition{ action } = sparse( problem.nrStates , problem.nrStates );
end
problem.observation = zeros( problem.nrStates , problem.nrActions , problem.nrObservations );
problem.reward = sparse( problem.nrStates , problem.nrActions );
for state = 1:problem.nrStates
    if state == problem.nrStates
        for action = 1:problem.nrActions
            problem.transition{ action }( state , state ) = 1;
            problem.observation( state , action , : ) = 1 / problem.nrObservations;
        end
    else
        [ target_x target_y filter scale scale_x scale_y ] = ...
            get_vec( state , pixel_count , filter_type_count , scale_cache );
        for action = 1:problem.nrActions
            
            % action was run filter
            if action == 1
                problem.transition{ action }( state , state ) = 1;
                quad = filter_quadrant( target_x , target_y , scale , scale_x , scale_y , scale_count );
                if scale <= filter
                    problem.observation( state , action , : ) = ( 1 - p_obs ) / ( problem.nrObservations - 1 );
                    problem.observation( state , action , quad ) = p_obs;
                else
                    problem.observation( state , action , : ) = 1 / problem.nrObservations;
                end
                problem.reward( state , action ) = r_base_cost( filter ) * ( 4 ^ ( scale_count - scale ) );
                
                if sum( problem.observation( state , action , : ) ) ~= 1
                    disp('wonky')
                end
                
            % action was submit state
            elseif action == 2
                problem.transition{ action }( problem.nrStates , state ) = 1;
                quad = filter_quadrant( target_x , target_y , scale , scale_x , scale_y , scale_count );
                problem.observation( state , action , : ) = 1 / problem.nrObservations;
                if ( quad < 5 && scale == scale_count )
                    problem.reward( state , action ) = r_submit_right;
                else
                    problem.reward( state , action ) = r_submit_wrong;
                end
                
            % action was swap filter
            elseif ( ( action >= min( filter_action_set ) ) && ( action <= max( filter_action_set ) ) )
                problem.observation( state , action , : ) = 1 / problem.nrObservations;
                new_filter = find( action == filter_action_set );
                new_state = get_state( target_x , target_y , new_filter , scale , ...
                    scale_x , scale_y , pixel_count , filter_type_count , scale_cache );
                problem.transition{ action }( new_state , state ) = 1;
                
                if sum( problem.transition{ action }( : , state ) ) ~= 1
                    disp('wonky')
                end
                
            % action was adjust scale
            else
                problem.observation( state , action , : ) = 1 / problem.nrObservations;
                scale_move = find( action == scale_action_set );
                if scale_move == 5
                    new_scale = max( 1 , scale - 1 );
                    if new_scale ~= scale
                        new_scale_x = ceil( scale_x / 2 );
                        new_scale_y = ceil( scale_y / 2 );
                    else
                        new_scale_x = scale_x;
                        new_scale_y = scale_y;
                    end
                else
                    new_scale = min( scale_count , scale + 1 );
                    if new_scale ~= scale
                        new_scale_x = 2 * ( scale_x - 1 ) + 1;
                        new_scale_y = 2 * ( scale_y - 1 ) + 1;
                        if ( scale_move == 2 || scale_move == 4 )
                            new_scale_x = new_scale_x + 1;
                        end
                        if ( scale_move == 3 || scale_move == 4 )
                            new_scale_y = new_scale_y + 1;
                        end
                    else
                        new_scale_x = scale_x;
                        new_scale_y = scale_y;
                    end
                end
                new_state = get_state( target_x , target_y , filter , new_scale , ...
                    new_scale_x , new_scale_y , pixel_count , filter_type_count , scale_cache );
                problem.transition{ action }( new_state , state ) = 1;
                
                if sum( problem.transition{ action }( : , state ) ) ~= 1
                    disp('wonky')
                end
                
                
            end
        end
    end
end
problem.maxReward = max( problem.reward(:) );
    

%----FULL STATE STUFF----------%
function [ target_x target_y filter scale scale_x scale_y ] = ...
    get_vec( state , pixel_count , filter_type_count , scale_cache )
side_count = sqrt( pixel_count );
scale_state = ceil( state / pixel_count / filter_type_count );
[ scale scale_x scale_y ] = get_scale_vec( scale_state , scale_cache );
rest = state - ( scale_state - 1 ) * pixel_count * filter_type_count;
filter = ceil( rest / pixel_count );
rest = rest - ( filter - 1 ) * pixel_count;
[ target_x target_y ] = get_loc_vec( rest , side_count );

function state = get_state( target_x , target_y , filter , scale , ...
    scale_x , scale_y , pixel_count , filter_type_count , scale_cache )
side_count = sqrt( pixel_count );
scale_state = get_scale_state( scale , scale_x , scale_y , scale_cache );
loc_state = get_loc_state( target_x , target_y , side_count );
state = loc_state + ( filter - 1 ) * pixel_count + ...
    ( scale_state - 1 ) * ( pixel_count * filter_type_count );

%----SCALE STUFF----------%
function [ scale scale_x scale_y ] = get_scale_vec( state , scale_cache )
scale = scale_cache( state , 1 );
scale_x = scale_cache( state , 2 );
scale_y = scale_cache( state , 3 );

function state = get_scale_state( scale , scale_x , scale_y , scale_cache )
state = find( ( scale_cache(:,1) == scale ) & ...
    ( scale_cache(:,2) == scale_x ) & ...
    ( scale_cache(:,3) == scale_y ) );

%----LOCATION STUFF----------%
function [x y] = get_loc_vec( state , side_count )
y = ceil( state / side_count );
x = state - ( y - 1 ) * side_count;

function state = get_loc_state( x , y , side_count )
state = x + ( y - 1 ) * side_count;

%----SET CHECK STUFF--------%
function quad = filter_quadrant( x , y , scale , scale_x , scale_y , scale_count )
quad = NaN;
side_count = 2^( scale_count - scale );
start_x = 1 + ( scale_x - 1 ) * side_count;
start_y = 1 + ( scale_y - 1 ) * side_count;
end_x = start_x + side_count - 1;
end_y = start_y + side_count - 1;
if ( ( x >= start_x ) && ( x <= end_x ) && ( y >= start_y ) && ( y <= end_y ) )
    if start_x == end_x
        quad = 1;
    else
        if ( ( ( ( end_x - x ) / side_count ) >= .5 ) && ( ( ( end_y - y ) / side_count ) >= .5 ) )
            quad = 1;
        elseif ( ( ( ( end_x - x ) / side_count ) >= .5 ) && ( ( ( end_y - y ) / side_count ) < .5 ) )
            quad = 3;
        elseif ( ( ( ( end_x - x ) / side_count ) < .5 ) && ( ( ( end_y - y ) / side_count ) >= .5 ) )
            quad = 2;
        elseif ( ( ( ( end_x - x ) / side_count ) < .5 ) && ( ( ( end_y - y ) / side_count ) < .5 ) )
            quad = 4;
        end
    end
else
    quad = 5;
end

