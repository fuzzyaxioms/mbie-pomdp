function [ s y a ] = robotLineWorld( sampleCount )

% make a simple line-world
% action 1 is left, action 2 is right
% obs 1 is left end, obs 2 is middle, obs 3 is right end
pGoodObs = 1;
pGoodTrans = 1;
stateCount = 4;

% generate a sequence using a random policy
s( 1 ) = ceil( rand * stateCount );
y( 1 ) = getObs( s(1) , stateCount , pGoodObs );
for i = 2:(sampleCount+1)
    a( i ) = ceil( rand * 2 );
    s( i ) = getTrans( s( i - 1 ) , a( i ) , stateCount , pGoodTrans );
    y( i ) = getObs( s( i ) , stateCount , pGoodObs );
end
a = a(2:end);
s = s(2:end);
y = y(2:end);

% ---------------------------------------------------------------------- %
function y = getObs( s , stateCount , pGoodObs )
wrongObs = 1:3;
if s == 1; rightObs = 1; 
elseif s == stateCount; rightObs = 3;
else rightObs = 2; end;
wrongObs( rightObs ) = [];   
if rand < pGoodObs
    y = rightObs;
else
    if rand < .5
        y = wrongObs(1);
    else
        y = wrongObs(2);
    end
end

% ----------------------------------------------------------------------- %
function s = getTrans( s , a , stateCount , pGoodTrans )
leftS = max( s - 1 , 1 );
rightS = min( s + 1 , stateCount );
if a == 1
    goodS = leftS; badS = rightS;
elseif a == 2
    goodS = rightS; badS = leftS;
end
if rand < pGoodTrans
    s = goodS;
else
    s = badS;
end
    
    


