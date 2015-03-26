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
dir_name = 'tworoom';
solver_type = 'ffbs';
action_type = 'beb'; 
savedir = 'outputs/';
for rep = 1:1;
     hyper_set.gamma = 10;
     hyper_set.alpha0 = 1;
     hyper_set.search_depth = 0;
     hyper_set.num_eps = 10;
     hyper_set.stop_early = false;
    tester( dir_name , action_type , solver_type , rep , hyper_set, savedir )
end
