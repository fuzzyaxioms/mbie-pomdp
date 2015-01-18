function simulate_domain( sim_pomdp , hmm_param_set , hyper_set , ...
    pomdp_param_set , test_param_set )
% function simulate_domain( sim_pomdp , hmm_param_set , hyper_set , ...
%     pomdp_param_set , test_param_set )
% this function actually runs the rl simulation using the desired
% learning algorithm (e.g. the ipomdp)

% add paths
addpath('../util/');
addpath('../pomdp-util/');
addpath('../ihmm-code/');

% ----- INITIALIZE THINGS ----- %
% start time
tic;

% hardcoded for now -Shayan
episodic = true;
state = sample_multinomial( sim_pomdp.start_dist , 1 );

if episodic
    epi = '_episodic';
else
    epi = '';
end
% initialize things for the loop
pomdp_set = []; utree_set = []; weight_set = [];
switch test_param_set.solver_type
    case 'utree'
        utree_set.tree.is_leaf = 1;
        utree_set.tree.use_action_index = 0;
        utree_set.tree.suffix_depth = 0;
        utree_set.tree.parent( 1 ) = NaN;
        utree_set.state_action_count_set = zeros( 1 , sim_pomdp.nrActions );
        utree_set.trajectory_assignment_set = [];
        utree_set.trajectory_set = [];
        utree_set.curr_leaf_node = 1;
        utree_set.qtable = rand( 1 , sim_pomdp.nrActions );
        utree_set.transition_count_set = zeros( [ 1 1 sim_pomdp.nrActions ] );
        utree_set.transition_set = ones( [ 1 1 sim_pomdp.nrActions ] );
        utree_set.reward_count_set = zeros( [ 1 sim_pomdp.nrActions ] );
        utree_set.reward_value_set = zeros( [ 1 sim_pomdp.nrActions ] );
        utree_set.reward_set = sim_pomdp.maxReward * ones( [ 1 sim_pomdp.nrActions ] );
end
experience_count = 0; rep = 1;
most_recent_update_iter = 1; update_count = 1;
most_recent_save_iter = 1;

% For non-episodic domains we have to set this before the loop. -Shayan
% pomdp_set = resetBeliefSet( pomdp_set );

% ----- TESTING LOOP ----- %
% loop until we hit the number of experience (however many episodes it takes)
while experience_count < test_param_set.total_experience_count
    disp(['starting experience ' num2str( experience_count ) ]);
    
    % ---- Run a Training Trial ---- %
    % otherwise, get the trial from the agent
    pomdp_set = resetBeliefSet( pomdp_set );
    % added this belief update, so we can maintain beliefs when not
    % episodic -Shayan
    if ~episodic
        for i=1:rep-1
            for j=1:test_param_set.max_iter_count
                pomdp_set = updateBeliefSet( pomdp_set , action_set{ i }( j ) , obs_set{ i }( j ) );
            end
        end
    end
    switch test_param_set.solver_type
        case { 'utree' } , 	
	     [ history utree_set node_count_set ] = testUTREESet( sim_pomdp , ...
                    utree_set , test_param_set , hmm_param_set );
	otherwise ,
        if episodic
	     [ history test_param_set weight_set ] = testPOMDPSet( sim_pomdp , ...
                    pomdp_set , test_param_set , weight_set );
        else
         [ history test_param_set weight_set] = testPOMDPSet_infhorizon( sim_pomdp , ...
                    pomdp_set , test_param_set , weight_set, state);
         state = history( test_param_set.max_iter_count , 4);
        end
    end
    experience_count = experience_count + size( history , 1 );
    
    % store trial information
    obs_set{ rep } = history( : , 1 );
    action_set{ rep } = history( : , 2 );
    action_match_set{ rep } = history( : , 6 );
    reward_set{ rep } = history( : , 3 );
    state_set{ rep } = history( : , 4 );
    
    % change reward values to reward indices for modeling purposes
    for r = 1:length( hmm_param_set.unique_reward_set )
        reward_index_set{ rep }( find( hmm_param_set.unique_reward_set( r ) == reward_set{ rep } ) ) = r;
    end
    
    % ---- Update the Models ---- %
    % update if the experience has exceeded the inter_update_count, if the
    % weights are bad, and if the experience count is overall high enough
    do_resample = ( experience_count >= ( most_recent_update_iter + ...
        test_param_set.inter_update_count ) );
    if test_param_set.resample_based_on_neffective
        if ~isempty( weight_set )
            n_effective = 1 / sum( weight_set .^ 2 );
            if n_effective < .4 * length( weight_set )
                do_resample = true;
            end
        end
    end
    if experience_count < test_param_set.initial_update_count
        do_resample = false;
    end
    if do_resample
        most_recent_update_iter = experience_count;
        disp('doing ipomdp update')
        
        % sample new hmms (world models)
        clear hmm_set
        [ pomdp_set hmm_param_set node_count_set ] = sample_new_model_set( hmm_param_set , hyper_set , ...
            obs_set , action_set , reward_index_set );
        % solve the new modele
        pomdp_set = solve_pomdp_set( pomdp_set , pomdp_param_set , ...
            experience_count / test_param_set.total_experience_count );
        weight_set = [];
                
    % ---- Do Catch Trials ---- %
	% In the catch trials, we don't do any exploration, just
        % exploitation, to test how good the model is
        tmp_test_param_set = test_param_set;
        tmp_test_param_set.action_selection_type = 'epsilon_greedy';
        tmp_test_param_set.r_epsilon = 0;
        for test_rep = 1:50
            switch test_param_set.solver_type
 	    	case { 'utree' }  
                    tmp_utree_set = utree_set;
                    test_history = testUTREESet( sim_pomdp , ...
                        tmp_utree_set , tmp_test_param_set , hmm_param_set );	       
		otherwise,
            % I introduced tmp_pomdp_set here so it doesn't interefere
            % with our current models, which we might not want reset
            % -Shayan
		    tmp_pomdp_set = resetBeliefSet( pomdp_set );                
                    test_history = testPOMDPSet( sim_pomdp , tmp_pomdp_set , ...
                        tmp_test_param_set , weight_set );               
            end
            
            % store trial information
            obs_set_test{ update_count , test_rep } = test_history( : , 1 );
            action_set_test{ update_count , test_rep } = test_history( : , 2 );
            action_match_set_test{ update_count , test_rep } = test_history( : , 6 );
            reward_set_test{ update_count , test_rep } = test_history( : , 3 );
            state_set_test{ update_count , test_rep } = test_history( : , 4 );
        end
            
        % ---- Save Data ---- %
        experience_set_test( update_count ) = experience_count;
        update_count = update_count + 1;
        save( [test_param_set.savedir 'simulation_rep_test_' test_param_set.sname epi ] , ...
            'action_set_test' , 'action_match_set_test' , 'state_set_test' , 'reward_set_test' , ...
            'obs_set_test' , 'experience_set_test' , 'pomdp_set' );
    end
    
    % store time
    time_set( rep ) = toc;
    rep = rep + 1;
end

% save overall stuff
save( [ test_param_set.savedir 'simulation_train_' test_param_set.sname epi ] , ...
    'action_set' , 'action_match_set' , 'state_set' , 'reward_set' , 'obs_set' ,  ...
    'pomdp_set' , 'time_set' , ...
    'pomdp_param_set' , 'test_param_set' , 'hyper_set' , 'hmm_param_set' );


% ----------------------------------------------------------------------- %
function [ pomdp_set hmm_param_set node_count_set ] = sample_new_model_set( ...
        hmm_param_set , hyper_set , obs_set , action_set , reward_index_set )
% does the padding and samples the new models
if hmm_param_set.S0_init
    start_with_one = false; pad_in_front = true;
    hmm_param_set.S0 = pad_sequence( obs_set , pad_in_front , start_with_one );
end

% sample the hmms
[ hmm_set ] = sample_new_hmm_set( obs_set , action_set , reward_index_set , ...
    hyper_set , hmm_param_set );
if hmm_param_set.hot_start
    hmm_param_set.model = hmm_set{ end };
    hmm_param_set.model = rmfield( hmm_param_set.model , 'S' );
end

% convert to pomdp format
for m = 1:hmm_param_set.model_count
    pomdp = hmm2pomdp( hmm_set{ m } , hyper_set , hmm_param_set );
    pomdp_set{ m }.nsa = permute( sum( hmm_set{ m }.nssa , 2 ) , [ 1 3 2 ] );
    pomdp_set{ m }.pomdp = pomdp;
    node_count_set( 1 , m ) = pomdp.nrStates;
end

% ----------------------------------------------------------------------- %
function out_set = pad_sequence( in_set , pad_in_front , start_with_one )
% pads either the front or end of the sequence with a one, makes all the
% sequences have continuous numbers, and can make the first element of each
% sequence start with one

% do the start with one thing while collecting the unique elements
number_set = [];
for sequence_index = 1:numel( in_set )
    in_sequence = in_set{ sequence_index };
    if start_with_one
        first_value = in_sequence( 1 );
        in_sequence( in_sequence == first_value ) = NaN;
        in_sequence( in_sequence == 1 ) = first_value;
        in_sequence( isnan( in_sequence ) ) = 1;
    end
    out_set{ sequence_index } = in_sequence;
    number_set = [ number_set ; in_sequence ];
end

% find the unique ones and pad
unique_set = unique( number_set );
unique_count = length( unique_set );
number_map( unique_set ) = 1:length( unique_set );
for sequence_index = 1:numel( in_set )
    out_set{ sequence_index } = number_map( out_set{ sequence_index } );
    if pad_in_front
        out_set{ sequence_index } = [ ( unique_count + 1 ) out_set{ sequence_index } ];
    else
        out_set{ sequence_index } = [ out_set{ sequence_index } ( unique_count + 1 ) ];
    end
end

% ----------------------------------------------------------------------- %
function pomdp_set = solve_pomdp_set( pomdp_set , pomdp_param_set , completion_fraction )

% solve a set of models
pbvi_iter_count = pomdp_param_set.pbvi_min_iter_count + ceil( completion_fraction * ...
    ( pomdp_param_set.pbvi_iter_count - pomdp_param_set.pbvi_min_iter_count ) );
for m = 1:numel( pomdp_set )
    pomdp = pomdp_set{ m }.pomdp;
    belief_set = sampleBeliefs( pomdp , pomdp_param_set.belief_count );
    [TAO vL sol] = runPBVILean( pomdp , belief_set , pbvi_iter_count );
    pomdp_set{ m }.S = belief_set;
    pomdp_set{ m }.sol = sol;
end
                
% ----------------------------------------------------------------------- %
function mdp_set = solve_mdp_set( mdp_set )
                    
% solve a set of models
for m = 1:numel( mdp_set )
    mdp = mdp_set{ m }.mdp;
    [ V Q policy ] = value_iteration( mdp );
    mdp_set{ m }.Q = Q;
    mdp_set{ m }.policy = policy;
end
