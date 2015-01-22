horizon = int64(50*50);
rewards = zeros(horizon, 1);
num_runs = 10;
for i=1:num_runs
%     load(['simulation_train_tworoom_ffbs_weighted_stochastic_' int2str(i) '.mat'])
%     load(['simulation_train_tworoom_ffbs_boss_' int2str(i) '.mat'])
%     load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic__episodic.mat'])
%     load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['simulation_train_tiger_ffbs_epsilon_greedy_' int2str(i) '_episodic.mat'])
%    load(['simulation_train_tiger_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%    load(['simulation_train_tiger_em_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['testingsimulation_train_tiger_ipomdp_epsilon_greedy_' int2str(i) '.mat'])

%     load(['simulation_train_tiger_em_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['simulation_train_tiger_ipomdp_epsilon_greedy_' int2str(i) '.mat'])
%     load(['simulation_train_shuttle_ipomdp_weighted_stochastic_' int2str(i) '.mat'])

%     load(['../domain-code/outputs/simulation_train_tworoom_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
    load(['../domain-code/outputs/simulation_train_tworoom_ipomdp_weighted_stochastic_' int2str(i) '_episodic.mat'])

    startidx = 1;
    for j=1:size(reward_set, 2)
%         totreward = totreward + sum(reward_set{j});
        endidx = startidx + size(reward_set{j}, 1) - 1;
        rewards(startidx:endidx) = rewards(startidx:endidx) + reward_set{j};
        startidx = endidx + 1;
    end
end

% totreward/(4*200*50);
t = 100;
rewards = rewards/(num_runs);
% num_points = (horizon - t/2)/t;
cumrewards = zeros(horizon, 1);
for i=1:horizon;
    tt = min(horizon-i+1,t);
    cumrewards(i) = mean(rewards(i : i+tt-1));
end
plot(1:horizon, cumrewards);
xlabel('Step');
ylabel(['Rewards (averaged over ' int2str(t) ' steps)']);
