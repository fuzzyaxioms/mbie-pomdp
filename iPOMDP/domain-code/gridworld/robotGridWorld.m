function pomdp = robotGridWorld( rowCount, colCount )

%function [ stateSet obsSet actionSet rewardSet rowSet colSet pomdp ] = ...
%    robotGridWorld( rowCount, colCount, sampleCount )


% make a simple grid-world
% * action 1 is right, action 2 is left, action 3 is up, action 4 is down
% * 16 observations, corresponding to whether there is something left,
% right, up, or down of the robot
% * you cannot go through things

% quality parameters
pGoodObs = 1; % multiplied four times, so actually .88 of seeing correct
pGoodTrans = .8;

% reward parameters
rMove = -1;
rGoal = 10;
rBad = -20;

% state parameters; do not block [1 1] as this is the start state
% blockedSet = [ 1 2 ];
blockedSet = [2 2];
% blockedSet = [];

% bad states
badSet = [3 3 ; 2 4 ; 1 5];

% === GENERATING A RANDOM WALK === %
% generate a motion sequence using row and col, random policy
blockedSet = [ blockedSet ; ... 
    repmat( 0 , rowCount + 2 , 1 ) [0:(rowCount+1)]' ; ...
    repmat( colCount + 1 , rowCount + 2 , 1 ) [0:(rowCount+1)]' ; ...
    [1:colCount]' repmat( 0 , colCount , 1 ) ;
    [1:colCount]' repmat( rowCount + 1 , colCount , 1 ) ];
blockedSet = [ blockedSet(:,2) blockedSet(:,1) ];

% rowSet( 1 ) = 1;
% colSet( 1 ) = 1;
% for i = 2:(sampleCount+1)
%     actionSet( i ) = ceil( rand * 4 );
%     [ rowSet( i ) colSet( i ) ] = moveRC( rowSet( i - 1 ), ...
%         colSet( i - 1 ), actionSet( i ), pGoodTrans , blockedSet );
% end
% actionSet = actionSet(2:end);
% rowSet = rowSet(2:end);
% colSet = colSet(2:end);
% 
% % convert motion sequence to single state number
% stateSet = rowSet + rowCount*( colSet - 1 );
% 
% % convert motion sequence to observation sequence
% rbs = ismember( [ rowSet' (colSet + 1)' ] , blockedSet , 'rows' );
% lbs = ismember( [ rowSet' (colSet - 1)' ] , blockedSet , 'rows' );
% dbs = ismember( [ (rowSet + 1)' colSet' ] , blockedSet , 'rows' );
% ubs = ismember( [ (rowSet - 1)' colSet' ] , blockedSet , 'rows' );
% obsSet = [ lbs rbs ubs dbs ];
% 
% % add noise to observation sequence
% obsSet = abs( obsSet - ( rand( size( obsSet ) ) > pGoodObs ) );
% obsSet = ( obsSet * [ 1 2 4 8 ]' )' + 1;
% 
% % create rewards
% rewardSet = ones( size( stateSet ) ) * rMove;
% rewardSet( find( stateSet == rowCount * colCount ) ) = rGoal;

% === GENERATING A MODEL === %
stateCount = rowCount * colCount;
actionCount = 4;
obsCount = 16;
pomdp.nrActions = actionCount;
pomdp.nrStates = stateCount;
pomdp.nrObservations = obsCount;
pomdp.reward = zeros( stateCount , actionCount );
pomdp.transition = zeros( stateCount , stateCount , actionCount );
pomdp.observation = zeros( stateCount , actionCount , obsCount );
for row = 1:rowCount
    for col = 1:colCount
        for a = 1:actionCount
            s =  row + rowCount*( col - 1 );

            % set reward model
            if row == rowCount && col == colCount
                pomdp.reward( s , a ) = rGoal;
            elseif ismember( [ row col ] , badSet , 'rows' );
                pomdp.reward( s , a ) = rBad;
            else
                pomdp.reward( s , a ) = rMove;
            end

            % set observation model
            rbs = ismember( [ row (col + 1) ] , blockedSet , 'rows' );
            lbs = ismember( [ row (col - 1) ] , blockedSet , 'rows' );
            dbs = ismember( [ (row + 1) col ] , blockedSet , 'rows' );
            ubs = ismember( [ (row - 1) col ] , blockedSet , 'rows' );
            tobs = [ rbs lbs dbs ubs ];
            for r = 0:1; for l = 0:1; for d = 0:1; for u = 0:1;
                obsVec = [ r l d u ];
                obs = obsVec * [ 1 2 4 8 ]' + 1;
                pobs = prod( (obsVec == tobs) * pGoodObs + (obsVec ~= tobs) * ( 1 - pGoodObs ) );
                pomdp.observation( s , a , obs ) = pobs;
            end; end; end; end;
            
            % set transition model
            if ismember( [ row col ] , blockedSet , 'rows' )
                pomdp.transition( s , s , a ) = 1;
            else
                [ pSet locSet ] = tprob( row , col , a , pGoodTrans , blockedSet );
                for i = 1:size( locSet , 1 )
                    sprime = locSet( i , 1 ) + rowCount * ( locSet( i , 2 ) - 1 );
                    pomdp.transition( sprime , s , a ) = pSet( i );
                end
            end            
        end
    end
end
pomdp.gamma = .95;
pomdp.start = zeros( 1 , stateCount ); pomdp.start(1) = 1;
pomdp.start_dist = pomdp.start';
pomdp.maxReward = rGoal;


% ---------------------------------------------------------------------- %
function [ pSet locSet ] = tprob( x , y , a , p , bS )

% locations and probs
locSet = [ x y ; x + 1 y ; x - 1 y ; x y + 1 ; x y - 1 ];
if a == 1
    pSet = [ 0 ; p ; 0 ; ( 1 - p )/2 ; (1 - p)/2 ];
elseif a == 2
    pSet = [ 0 ; 0 ; p ; ( 1 - p )/2 ; (1 - p)/2 ];
elseif a == 3
    pSet = [ 0 ; ( 1 - p )/2 ; (1 - p)/2 ; p ; 0 ];
elseif a == 4
    pSet = [ 0 ; ( 1 - p )/2 ; (1 - p)/2 ; 0 ; p ];
end

% get rid of blocked
[ a bInd  b ] = intersect( locSet , bS , 'rows' );
locSet( bInd , : ) = [];
pSet( 1 ) = pSet( 1 ) + sum( pSet( bInd ) );
pSet( bInd ) = [];

% ----------------------------------------------------------------------- %
function [ x y ] = moveRC( x , y , a , p , bS )

% get pset/locSet
[ pSet locSet ] = tprob( x , y , a , p , bS );

% sample a location
loc = sampleMultinomial( pSet , 1 );
x = locSet( loc , 1 );
y = locSet( loc , 2 );
