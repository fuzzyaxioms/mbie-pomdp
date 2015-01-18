horizon = 4050;
rewards = zeros(horizon, 1);
num_runs = 1;
for i=1:num_runs
%     load(['simulation_train_tworoom_ffbs_weighted_stochastic_' int2str(i) '.mat'])
%     load(['simulation_train_tworoom_ffbs_boss_' int2str(i) '.mat'])
%     load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic__episodic.mat'])
%     load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['simulation_train_tiger_ffbs_epsilon_greedy_' int2str(i) '_episodic.mat'])
    load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['testingsimulation_train_tiger_ipomdp_epsilon_greedy_' int2str(i) '.mat'])

%     load(['simulation_train_tiger_em_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['simulation_train_tiger_ipomdp_epsilon_greedy_' int2str(i) '.mat'])
%     load(['simulation_train_shuttle_ipomdp_weighted_stochastic_' int2str(i) '.mat'])

    startidx = 1;
    for j=1:size(reward_set, 2)
%         totreward = totreward + sum(reward_set{j});
        endidx = startidx + size(reward_set{j}, 1) - 1;
        rewards(startidx:endidx) = rewards(startidx:endidx) + reward_set{j};
        startidx = endidx + 1;
    end
end
% totreward/(4*200*50);
t = 500;
rewards = rewards/(num_runs);
num_points = (horizon - 250)/t;
cumrewards = zeros(num_points, 1);
for i=1:num_points;
    cumrewards(i) = sum(rewards((i-1)*t + 251 : i*t + 250))/t;
end
plot(t*(1:num_points), cumrewards);
xlabel('Step');
ylabel('Rewards (averaged over 500 steps)');