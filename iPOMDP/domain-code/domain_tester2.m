function domain_tester2(dir_name, action_type, start_rep, end_rep, num_eps, stop_early, search_depth)
    % This is a little script that runs the ipomdp code for a particular
    % domain.  Just fill in the following:
    %
    % 1. dir_name: the name of the directory with the problem in it
    %
    % 2. solver_type: can be ipomdp, em, ffbs, em_big, ffbs_big, utree
    %
    % 3. action_type: stochastic_forward_search, epsilon_greedy, beb, boss,
    %    weighted_stochastic, softmax.  NOTE: beb, boss apply the
    %    concepts from beb/boss to the POMDP domain; originally these
    %    were for MDPs
    %
    % 4. hyper_set.gamma: the top-level concentration parameter in the HDP-HMM
    %    used by the ipomdp (you can think of this as a guess on the expected
    %    number of visited states)
    %
    % 5. hyper_set.alpha: the bottom-level concentration parameter in the HDP-HMM
    %    used by the ipomdp (you can think of this as how similar transitions
    %    are expected to be)
    %
    addpath ../ipomdp-code/
    solver_type = 'ffbs';
    savedir = 'outputs/';
    for rep = start_rep:end_rep;
         hyper_set.gamma = 10;
         hyper_set.alpha0 = 1;
         hyper_set.search_depth = search_depth;
         hyper_set.num_eps = num_eps;
         hyper_set.stop_early = stop_early;
        tester( dir_name , action_type , solver_type , rep , hyper_set, savedir )
    end
end
