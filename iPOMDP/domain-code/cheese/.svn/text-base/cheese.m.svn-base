function pomdp = cheese()
%function [ stateSet obsSet actionSet rewardSet rowSet colSet pomdp ] = ...
%    robotGridWorld( rowCount, colCount, sampleCount )
% make a simple grid-world
% * action 1 is right, action 2 is left, action 3 is up, action 4 is down
% * 16 observations, corresponding to whether there is something left,
% right, up, or down of the robot
% * you cannot go through things
rowCount = 5;
colCount = 7;

% quality parameters
pGoodObs = 1; % else completely different
pGoodTrans = 1; % else completely different

% reward parameters
rMove = -.1;
rGoal = 1;
rCollide = -1;

% blocked spots, top-left is the start state
gridmap = ...
  [ 1 1 1 1 1 1 1 1 1 ; ...
    1 0 0 0 0 0 0 0 1 ; ...
    1 0 1 1 0 1 1 0 1 ; ...
    1 0 1 1 0 1 1 0 1 ; ...
    1 0 1 1 0 1 1 0 1 ; ...
    1 0 0 0 0 0 0 0 1 ; ...
    1 1 1 1 1 1 1 1 1 ];
goal = [ 4 5 ];

% convert the gridmap to state values and basic observation values
row_col_state_obs_map = []; state_count = 1;
ovec_set = [];
for row = 1:size( gridmap , 1 )
    for col = 1:size( gridmap , 2 )
        if gridmap( row , col ) == 0

            % process the observations
            ovec = [ gridmap( row + 1 , col )  gridmap( row - 1 , col )  ...
                gridmap( row , col + 1 )  gridmap( row , col - 1 ) ];
            obs = find( ismember( ovec_set , ovec , 'rows' ) );
            if isempty( obs )
                ovec_set( end + 1 , : ) = ovec; 
                obs = size( ovec_set , 1 );
            end
            
            % save mapping
            row_col_state_obs_map( end + 1 , : ) = [ row col state_count obs ];
            state_count = state_count + 1;
        end
    end
end

% basics %
stateCount = size( row_col_state_obs_map , 1 );
actionCount = 4;
obsCount = max( row_col_state_obs_map( : , 4 ) );
pomdp.nrActions = actionCount;
pomdp.nrStates = stateCount;
pomdp.nrObservations = obsCount;
pomdp.reward = zeros( stateCount , actionCount );
pomdp.transition = zeros( stateCount , stateCount , actionCount );
pomdp.observation = zeros( stateCount , actionCount , obsCount );

% fancy stuff
pBadObs = ( 1 - pGoodObs ) / ( obsCount - 1 );
pBadTrans = ( 1 - pGoodObs ) / 3;
for row = 1:size( gridmap , 1 )
    for col = 1:size( gridmap , 2 )
        for a = 1:actionCount
            if gridmap( row , col ) == 0
                [ state obs ] = find_so( row_col_state_obs_map , [ row col ] );
                
                % assign the observations
                pomdp.observation( state , a , : ) = pBadObs;
                pomdp.observation( state , a , obs ) = pGoodObs;
                
                % the next state, for deciding whether you'll collide
                if a == 1
                    next_row = row + 1;
                    next_col = col;
                elseif a == 2
                    next_row = row - 1;
                    next_col = col;
                elseif a == 3
                    next_row = row;
                    next_col = col + 1;
                elseif a == 4
                    next_row = row;
                    next_col = col - 1;
                end
                
                % assign rewards
                if ( ( row == goal( 1 ) ) && ( col == goal( 2 ) ) )
                    pomdp.reward( state , a ) = rGoal;
                elseif gridmap( next_row , next_col ) == 1;
                    pomdp.reward( state , a ) = rCollide;
                else
                    pomdp.reward( state , a ) = rMove;
                end
                
                % the neighboring states -- up
                up_row = row + 1; up_col = col;
                if gridmap( up_row , up_col ) == 1
                    up_row = row; up_col = col;
                end
                next_state = find_so( row_col_state_obs_map , [ up_row up_col ] );
                if a == 1
                    pomdp.transition( next_state , state , a ) = pGoodTrans + ...
                        pomdp.transition( next_state , state , a );
                else
                    pomdp.transition( next_state , state , a ) = pBadTrans + ...
                        pomdp.transition( next_state , state , a );
                end
                
                % the neighboring states -- down
                down_row = row - 1; down_col = col;
                if gridmap( down_row , down_col ) == 1
                    down_row = row; down_col = col;
                end
                next_state = find_so( row_col_state_obs_map , [ down_row down_col ] );
                if a == 2
                    pomdp.transition( next_state , state , a ) = pGoodTrans + ...
                        pomdp.transition( next_state , state , a );
                else
                    pomdp.transition( next_state , state , a ) = pBadTrans + ...
                        pomdp.transition( next_state , state , a );
                end
                
                % the neighboring states -- left
                left_row = row; left_col = col + 1;
                if gridmap( left_row , left_col ) == 1
                    left_row = row; left_col = col;
                end
                next_state = find_so( row_col_state_obs_map , [ left_row left_col ] );
                if a == 3
                    pomdp.transition( next_state , state , a ) = pGoodTrans + ...
                        pomdp.transition( next_state , state , a );
                else
                    pomdp.transition( next_state , state , a ) = pBadTrans + ...
                        pomdp.transition( next_state , state , a );
                end
                
                % the neighboring states -- right
                right_row = row; right_col = col - 1;
                if gridmap( right_row , right_col ) == 1
                    right_row = row; right_col = col;
                end
                next_state = find_so( row_col_state_obs_map , [ right_row right_col ] );
                if a == 4
                    pomdp.transition( next_state , state , a ) = pGoodTrans + ...
                        pomdp.transition( next_state , state , a );
                else
                    pomdp.transition( next_state , state , a ) = pBadTrans + ...
                        pomdp.transition( next_state , state , a );
                end
            end
        end
    end
end

% final bits
pomdp.gamma = .95;
pomdp.start = zeros( 1 , stateCount ); pomdp.start( 1 ) = 1;
pomdp.start_dist = pomdp.start';
pomdp.maxReward = max( pomdp.reward(:) );

% ----------------------------------------------------------------------- %
function [ s o ] = find_so( rcso , rc )
index = find( ismember( rcso( : , 1:2 )  , rc , 'rows' ) );
s = rcso( index , 3 );
o = rcso( index , 4 );

