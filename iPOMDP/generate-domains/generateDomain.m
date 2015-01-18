filename = 'tiger';
problem = readPOMDP( strcat(filename, '.pomdp') , 0 );
problem.reward = zeros( problem.nrStates , problem.nrActions );

for state_index = 1:problem.nrStates
    problem.reward(state_index, :) = problem.reward3(1, state_index, :);
end
problem.start = ones( 1 , problem.nrStates ) * 1/problem.nrStates;
% problem.start = [1 0];
problem.start_dist = problem.start';
problem.maxReward = max( problem.reward(: ));
problem.minReward = min( problem.reward(: ));
save( strcat('../domain-code/', filename, '/', filename, '.mat') , 'problem')