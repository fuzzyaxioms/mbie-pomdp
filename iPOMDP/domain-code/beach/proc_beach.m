problem = readPOMDP( 'beach.pomdp' , 0 );
problem.reward = zeros( problem.nrStates , problem.nrActions );

start = 'S33U';
reward_state = '33';
for state_index = 1:problem.nrStates
    if strmatch( problem.states( state_index , : ) , start )
        start_index = state_index;
    end
    if ~isempty( strfind( problem.states( state_index , : ) , reward_state ) )
        problem.reward( state_index , : ) = 1;
    end
end
problem.start = zeros( 1 , problem.nrStates );
problem.start( start_index ) = 1;
problem.start_dist = problem.start';
problem.maxReward = max( problem.reward(: );