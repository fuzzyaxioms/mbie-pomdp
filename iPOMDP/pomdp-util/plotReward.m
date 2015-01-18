horizon = 1900;
num_runs = 6;
% num_rows = fix((horizon - 250) / 100);
num_rows = 40;
% num_points = fix(num_rows / 5) + 1;
num_points = num_rows;
rewards = zeros(num_points, 1);
lengths = zeros(num_points, 1);

for i=1:num_runs
    load(['simulation_rep_test_tiger_ffbs_epsilon_greedy_' int2str(i) '_episodic.mat'])

%     load(['testingsimulation_rep_test_tiger_ipomdp_weighted_stochastic_' int2str(i) '.mat'])
%     load(['testingsimulation_rep_test_hallway_ipomdp_weighted_stochastic_' int2str(i) '.mat'])

%     load(['simulation_rep_test_tworoom_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])

    for j=1:size(reward_set_test, 2)
        for k=1:num_rows
            idx = k;
%             idx = fix((k - 1) / 5) + 1;
            rewards(idx) = rewards(idx) + sum(reward_set_test{k, j});
            lengths(idx) = lengths(idx) + size(reward_set_test{k, j}, 1);
%             lengths(idx) = lengths(idx) + 1;

        end
    end
end
rewards = rewards ./ lengths
t = 500;
plot(t*(1:num_points), rewards);
xlabel('Step');
ylabel('Rewards');