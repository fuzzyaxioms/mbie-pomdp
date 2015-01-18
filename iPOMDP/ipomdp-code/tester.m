function tester( dir_name , action_type , solver_type , rep , input_set, savedir )
% function tester( dir_name , action_type , solver_type , rep , input_set )

% This file sets a bunch of parameters for running tests with the
% iPOMDP that you probably don't need to edit often/from domain to
% domain.  They are all set to "reasonable" values.  The inputs to
% this function are:
%   - dir_name: contains the directory where the problem file is
%
%   - action_type: choose between the strings:
%     stochastic_forward_search, epsilon_greedy, beb, boss,
%     weighted_stochastic, softmax.  NOTE: beb, boss apply the
%     concepts from beb/boss to the POMDP domain; originally these
%     were for MDPs
%
%   - solver_type: choose between the strings: ipomdp, em, ffbs,
%     em_big, ffbs_big, utree
%
%   - rep: is which repetition this is (for saving lots of trials)
%
%   - input_set: contains fields for alpha0, gamma (concentration parameters)
%

% ---- Load the Problem ---- %
% load up the problem (should be saved as "problem") 
fname = dir_name;
addpath( dir_name );
load( fname );
sim_pomdp = problem;

% ---- Set the HMM Hypers ---- %
% hyper-parameters for the iHMM: gamma is the base concentration,
% alpha0 is the leaf concentration (small = sparse).  If the alpha0
% and gamma are set, then they will not be sampled; else, they will
% be sampled.
hyper_set.alpha0 = input_set.alpha0;
hyper_set.gamma = input_set.gamma;

% hyper priors for the concentration parameters 
hyper_set.alpha0_a = 1;  % alpha: second level - sparsity
hyper_set.alpha0_b = 1;
hyper_set.gamma_a = 5; % gamma: top level - state count
hyper_set.gamma_b = 1;

% pull other counts of things that is used in the ihmm beam sampling
% code (basically converting between names)
hyper_set.M = problem.nrActions;
hyper_set.maxReward = sim_pomdp.maxReward;
hyper_set.minReward = sim_pomdp.minReward;
hyper_set.action_count = problem.nrActions;
hyper_set.obs_count = problem.nrObservations;
hyper_set.reward_count = length( unique( sim_pomdp.reward(:) ) );

% set the hypers for the reward and observation models
hyper_set.H = 1 * ones( 1 , sim_pomdp.nrObservations );
% unique_reward_set = unique( sim_pomdp.reward(:) );
unique_reward_set = sim_pomdp.rmap';
hyper_set.HR = .1 * ones( 1 , length( unique_reward_set ) );
hmm_param_set.unique_reward_set = unique_reward_set;

% ---- Set the POMDP Solver Parameters ---- %
% belief count is the number of beliefs used for PBVI, as the number of
% episodes increases, we scale the amount of computation done to solve the
% model from min_iter_count to iter_count
pomdp_param_set.belief_count = 500;
pomdp_param_set.pbvi_iter_count = 35;
pomdp_param_set.pbvi_min_iter_count = 10;

% ---- Set the Test Parameters ---- %
% the total_experience_count is the number of interactions the agent
% makes with the world in run; no single episode is allowed to be
% longer than max_iter_count.  Models are updated every
% inter_update_count (up to the nearest end of trial) after
% initial_update_count OR when neffective drops below .4N, depending
% on what flags are set.

% ORIGINAL SETTINGS from Finale's Code
% test_param_set.solver_type = solver_type;
% test_param_set.max_iter_count = 50;
% test_param_set.total_experience_count = 50 * test_param_set.max_iter_count;
% test_param_set.inter_update_count = 100; 
% test_param_set.initial_update_count = 250;
% test_param_set.resample_based_on_neffective = false;
% MODIFIED SETTINGS for our experiments
test_param_set.solver_type = solver_type;
test_param_set.max_iter_count = 75; %horizon length/inter_update_count (or episode length for episodic domains)
test_param_set.total_experience_count = 100 * test_param_set.max_iter_count; % number of updates wanted (or number of episodes) * max_iter_count
test_param_set.inter_update_count = 100; 
test_param_set.initial_update_count = 250;
test_param_set.resample_based_on_neffective = false;

% action-selection related parameters: algorithms are epsilon greedy, beb,
% boss, weighted stochastic, softmax, and stochastic forward search
test_param_set.action_selection_type = action_type;
test_param_set.r_epsilon = .1;    % epsilon for the epsilon greedy 
test_param_set.search_depth = 4;   % for the value iteration algs
test_param_set.branch_count = 3;   % for stochastic forward search
test_param_set.beb_constant = 10;  % for the beb algorithm
test_param_set.leaf_action_selection_type = 'ub'; % for tree algs
test_param_set.sparse_sample_count = 10;

% ---- Set the HMM Inference Parameters ---- %
% use_tempering starts chain hot, cools (useful).  hot_start
% uses ihmm of the previous iteration to do the next (sometimes useful,
% sometimes just gets stuck in back optima).  if sample_start = true, we
% sample Pi0 as well as the transition matrix Pi.  if S0_init = true, then
% we initialize the ihmm inference state sequence with the obs sequence
hmm_param_set.use_tempering = true;
hmm_param_set.hot_start = true;
hmm_param_set.sample_start = true;
hmm_param_set.S0_init = false;

% depending on the solver type, set remaining parameters.  sample_m is
% whether to sample the model (vs. maximize) and sample_s is whether
% to sample the state sequence (vs. use expectations).
switch solver_type
    
    % em case
    case 'em'
        hmm_param_set.state_count = problem.nrStates;
        hmm_param_set.sample_s = false;
        hmm_param_set.sample_m = false;
        hmm_param_set.burnin_count = 1;
        hmm_param_set.model_count = 1;
        hmm_param_set.inter_iter_count = 1;
        
    % large em
    case 'em_big'
        hmm_param_set.state_count = problem.nrStates * 10;
        hmm_param_set.sample_s = false;
        hmm_param_set.sample_m = false;
        hmm_param_set.burnin_count = 1;
        hmm_param_set.model_count = 1;
        hmm_param_set.inter_iter_count = 1;
        
    % ipomdp
    case 'ipomdp'
        hmm_param_set.state_count = Inf;
        hmm_param_set.sample_s = true;
        hmm_param_set.sample_m = true;
        hmm_param_set.burnin_count = 50;
        hmm_param_set.model_count = 10;
        hmm_param_set.inter_iter_count = 10;
        
    % ffbs, correct state count
    case 'ffbs'
        hmm_param_set.state_count = problem.nrStates;
        hmm_param_set.sample_s = true;
        hmm_param_set.sample_m = true;
        hmm_param_set.burnin_count = 50;
        hmm_param_set.model_count = 10;
        hmm_param_set.inter_iter_count = 10;
        
    % ffbs, large state count
    case 'ffbs_big'
        hmm_param_set.state_count = 10 * problem.nrStates;
        hmm_param_set.sample_s = true;
        hmm_param_set.sample_m = true;
        hmm_param_set.burnin_count = 50;
        hmm_param_set.model_count = 10;
        hmm_param_set.inter_iter_count = 10;
                
    % utree
    case 'utree'
        hmm_param_set.max_depth = 8;
end

% ---- Info for Saving the Results ---- %
% create the save name and specify directory where the outputs should
% be saved (name + directory)
test_param_set.sname = [ fname '_' solver_type ...
                    '_' test_param_set.action_selection_type ...
                    '_' num2str( rep ) ];
if isfield( input_set , 'sname' )
    test_param_set.sname = [ test_param_set.sname input_set.sname ];
end
test_param_set.savedir = savedir;

% ---- Run ---- %
% run the tester
simulate_domain( sim_pomdp , hmm_param_set , hyper_set , ...
    pomdp_param_set , test_param_set )

% clean up
rmpath( dir_name )
