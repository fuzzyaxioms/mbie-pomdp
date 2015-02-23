function graphTrain()
    horizon = int64(50*50);
    num_runs = 10;
    
%     root = 'simulation_train_tworoom_ffbs_epsilon_greedy_';
%     root = 'simulation_train_tworoom_ffbs_weighted_stochastic_';
%     root = 'simulation_train_tworoom_ffbs_boss_';
%     root = 'simulation_train_tworoom_ffbs_beb_';
    
%     root = 'simulation_train_2sensortiger_ffbs_epsilon_greedy_';
    root = 'simulation_train_2sensortiger_ffbs_weighted_stochastic_';
    
    [xs1, ys1] = processTrainRewards( [ '../domain-code/outputs/keep_start100/' root ], '_episodic.mat', horizon, num_runs);
    
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/tworoom_ffbs_weighted_stochastic/simulation_train_tworoom_ffbs_weighted_stochastic_', '_episodic.mat', horizon, num_runs);
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_tworoom_ffbs_weighted_stochastic_', '_episodic.mat', horizon, num_runs);
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_tworoom_ffbs_softmax_', '_episodic.mat', horizon, num_runs);
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_tworoom_ffbs_epsilon_greedy_', '_episodic.mat', horizon, num_runs);
    

%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_2sensortiger_ffbs_weighted_stochastic_', '_episodic.mat', horizon, num_runs);
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_2sensortiger_ffbs_epsilon_greedy_', '_episodic.mat', horizon, num_runs);
%     [xs1, ys1] = processTrainRewards('../domain-code/outputs/simulation_train_2sensortiger_ffbs_softmax_', '_episodic.mat', horizon, num_runs);
    plot(xs1, ys1, 'b');
    hold on;
%     plot(xs2, ys2, 'g');
    xlabel('Step');
    ylabel(['Reward']);
%     ylim([0.35 0.75]);
    
    plotdata = {xs1, ys1};
%     save('train_rewards.mat','plotdata');
    save([ root '.mat' ], 'plotdata' );
    
end

function [xs, ys] = processTrainRewards(prefix, suffix, horizon, num_runs)
    rewards = zeros(1,horizon);
    for i=1:num_runs
        load([prefix int2str(i) suffix]);

        startidx = 1;
        for j=1:size(reward_set, 2)
    %         totreward = totreward + sum(reward_set{j});
            endidx = startidx + size(reward_set{j}, 1) - 1;
            if endidx > horizon
                endidx = horizon;
            end
            count = endidx - startidx + 1;
            rewards(startidx:endidx) = rewards(startidx:endidx) + reward_set{j}(1:count)';
            startidx = endidx + 1;
        end
    end
    rewards = rewards ./ (num_runs);
    xs = 1:horizon;
    ys = rewards;
end
