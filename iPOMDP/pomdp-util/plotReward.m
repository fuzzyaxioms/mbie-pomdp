horizon = int64(50*50);
num_runs = 10;
rewards = zeros(1,horizon);
horizons = zeros(1,num_runs);

for i=1:num_runs
%     load(['../domain-code/outputs/simulation_rep_test_2sensortiger_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['../domain-code/outputs/simulation_rep_test_2sensortiger_ffbs_epsilon_greedy_' int2str(i) '_episodic.mat'])
%     load(['../domain-code/outputs/simulation_rep_test_2sensortiger_ffbs_softmax_' int2str(i) '_episodic.mat'])

%     load(['../domain-code/outputs/tworoom_ffbs_weighted_stochastic/simulation_rep_test_tworoom_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['../domain-code/outputs/simulation_rep_test_tworoom_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
%     load(['../domain-code/outputs/simulation_rep_test_tworoom_ffbs_softmax_' int2str(i) '_episodic.mat'])
    load(['../domain-code/outputs/simulation_rep_test_tworoom_ffbs_epsilon_greedy_' int2str(i) '_episodic.mat'])
    
%     load(['testingsimulation_rep_test_tiger_ipomdp_weighted_stochastic_' int2str(i) '.mat'])
%     load(['testingsimulation_rep_test_hallway_ipomdp_weighted_stochastic_' int2str(i) '.mat'])

%     load(['simulation_rep_test_tworoom_ffbs_weighted_stochastic_' int2str(i) '_episodic.mat'])
    num_trials = size(experience_set_test,2);
    startidx = 1;
    endidx = 0;
    for k=1:num_trials
        endidx = experience_set_test(k);
        if endidx > horizon
            endidx = horizon;
        end
        
        avg_reward_per_step = 0.0;
        num_reps = size(reward_set_test, 2);
        for j=1:num_reps
            ep_reward = sum(reward_set_test{k, j});
            ep_length = size(reward_set_test{k,j}, 1);
            avg_reward_per_step = avg_reward_per_step + ep_reward/ep_length;
        end
        avg_reward_per_step = avg_reward_per_step / num_reps;
        rewards(startidx:endidx) = avg_reward_per_step;
        
        startidx = endidx+1;
    end
    horizons(i) = endidx;
end
realhorizon = min(horizons);
lengths = 1:realhorizon;
rewards = rewards(1:realhorizon);
plot(lengths,rewards);
xlabel('Step');
ylabel('Rewards');

plotdata = {lengths, rewards};
save('catch_trials.mat','plotdata');