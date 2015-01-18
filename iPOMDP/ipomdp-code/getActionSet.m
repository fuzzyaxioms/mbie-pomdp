function [ a q_var param_set was_greedy ] = getActionSet( sample_set , weight_set , param_set , action_count )
% function a = getActionSample( sample_set , weight_set , param_set )
%
% Returns an action given a set of POMDPS.  WARNING: there are many
% hard-coded things here, especially with the BeB and BOSS options!
% 
% The choices for the action-selection approach (among models, we assume
% that we have solved each individual POMDP) are:
%
% 1. epsilon greedy, if being greedy, chooses average of pomdp
% 
% 2. bayesian exploration bonus (adapted to POMDPs: basic idea is to
% modify the stochastic forward search by adding a bonus to rewards of
% less known rewards)
%        
% 3. best of sampled set (adapted to POMDPs: basic idea is to modify the
% stochastic forward search by taking the most optimistic q-value at
% each iteration)
% 
% 4. weighted stochastic - pick a model based on its weight and execute
% its action (kind of like thompson sampling, except sample a pomdp per
% action instead of per episode)
% 
% 5. softmax - choose an action based on its value function (implemented
% as each model puts forward its best action/value pair, and we choose
% based on the values... should probably do a full-out q-function
% version of this as well
%
% 6. forward search - expand the tree, sampling observations using the
% average model (equivalently, or as implemented, choose a model based
% on its weight and sample an observation from it)
%
% NOTE: if r_epsilon > 0, an epsilon-greedy will ALWAYS be applied
% REGARDLESS of the action selection approach!

% ----- Logging Variance, Greedy Actions ----- %
q_var = 0; was_greedy = NaN;
if iscell( sample_set )
        for i = 1:numel( sample_set )
            Q( i , : ) = getQLean( sample_set{ i }.bel' , sample_set , i , ...
                1:sample_set{ 1 }.pomdp.nrActions );
        end
        
        % variance
        if numel( sample_set ) > 1
            q_var = sum( var( Q ) );
        end
        
        % greedy action
        weight_set = weight_set / sum( weight_set );
        flat_Q = weight_set * Q;
    [ val greedy_action ] = max( flat_Q );
end

% ----- Simple Case ----- %
% Do random if not given a sample set or we are to do random policy
if ~iscell( sample_set )
    a = ceil( rand * action_count );
    return;
end

% ----------------------------------------- %
%    Go through action selection types      %
% ----------------------------------------- %
switch param_set.action_selection_type
    
    % epsilon greedy, if being greedy, chooses average of pomdp
    case 'epsilon_greedy'
        a = greedy_action;
        
    % bayesian exploration bonus (adapted to POMDPs: basic idea is to
    % modify the stochastic forward search by adding a bonus to rewards of
    % less known rewards)
    case 'beb'
        action_count = sample_set{ 1 }.pomdp.nrActions;
        for a = 1:action_count
            param_set.mean_model_value_cache = [];
            [ val( a ) param_set ] = score_action( sample_set , weight_set , a , param_set.search_depth , param_set );
        end
        [ max_val a ] = max( val );
        
    % best of sampled set (adapted to POMDPs: basic idea is to modify the
    % stochastic forward search by taking the most optimistic q-value at
    % each iteration)
    case 'boss'
        action_count = sample_set{ 1 }.pomdp.nrActions;
        for a = 1:action_count
            param_set.mean_model_value_cache = [];
            [ val( a ) param_set ] = score_action( sample_set , weight_set , a , param_set.search_depth , param_set );
        end
        [ max_val a ] = max( val );
    
    % weighted stochastic - pick a model based on its weight and execute
    % its action (kind of like thompson sampling, except sample a pomdp per
    % action instead of per episode)
    case 'weighted_stochastic'
        pomdp_index = sample_multinomial( weight_set );
        a = getAction( sample_set{ pomdp_index }.bel' , sample_set{ pomdp_index }.sol.Vtable );

    % softmax - choose an action based on its value function (implemented
    % as each model puts forward its best action/value pair, and we choose
    % based on the values... should probably do a full-out q-function
    % version of this as well
    case 'softmax'
        for i = 1:numel( sample_set )
            [ bestA( i ) bestV( i ) ] = getAction( sample_set{ i }.bel' , sample_set{ i }.sol.Vtable );
        end
        a = bestA( sample_multinomial( bestV  ) );
        
    % forward search - expand the tree, sampling observations using the
    % average model (equivalently, or as implemented, choose a model based
    % on its weight and sample an observation from it)
    case 'stochastic_forward_search' 
        action_count = sample_set{ 1 }.pomdp.nrActions;
        for a = 1:action_count
            for i = 1:param_set.branch_count
                param_set.mean_model_value_cache = [];
                [ val( i , a ) param_set ] = score_action( sample_set , weight_set , a , param_set.search_depth , param_set );
            end
        end
        [ max_val a ] = max( sum( val , 1 ) );
end

% record whether we wanted to take the greedy action before applying the
% epsilon greedy
was_greedy = ( a == greedy_action );

% take a random action if the epsilon greedy is set nonzero.  NOTE: this
% epsilon-greedy will ALWAYS be applied UNLESS r_epsilon = 0, REGARDLESS of
% the action selection approach
if rand < param_set.r_epsilon
    a = ceil( rand * sample_set{ 1 }.pomdp.nrActions );
end

% ----------------------------------------------------------------------- %
function [ val param_set ] = score_action( sample_set , weight_set , a , depth , param_set )

% disp( ['Running score action at depth ' num2str( depth ) ' with weights ' num2str( weight_set ) ] );

% ----- if depth = 0, estimate using individual q fuctions ----- %
if depth == 0
    clear q;
    for i = 1:numel( sample_set )
        q( i ) = getQ( sample_set{ i }.bel , sample_set{ i }.sol , a );
    end
    switch param_set.leaf_action_selection_type
        case 'ub'
            switch param_set.action_selection_type
                case 'beb'
                    na = get_beb_count( sample_set , weight_set , a );
                    gamma = sample_set{ 1 }.pomdp.gamma;
                    val = sum( weight_set .* q ) + param_set.beb_constant / ( 1 + na ) ...
                        * gamma^( param_set.search_depth + 1 ) / ( 1 - gamma );
                case 'boss'
                    val = max( q );
                case 'stochastic_forward_search'
                    val = sum( weight_set .* q );
            end
        case 'lb'
            % finale! not specific to the different algorithms! only looks
            % at the start state, not overall!  bel count and pbvi iters
            % are hard coded !!  similarity thresh is hard-coded !!
            
            
            % check if weight set is similar enough to weights that we've
            % already seen, or compute the solution
            model_diff_thresh = .1;
            compute_value = true;
            if ~isempty( param_set.mean_model_value_cache )
                diff_set = sum( abs( param_set.mean_model_value_cache( : , 1:end-1 ) - repmat( weight_set , ...
                    [ size( param_set.mean_model_value_cache , 1 ) 1 ] ) ) , 2 );
                [ min_val min_ind ] = min( diff_set );
                if min_val < model_diff_thresh
                    val = param_set.mean_model_value_cache( min_ind , end );
                    compute_value = false;
                end
            end
            % else compute the value
            if compute_value
                disp( ['Computed value for weight ' weight_set ] );
                pomdp = param_set.joint_model;
                pomdp.start = compute_joint_start( pomdp , weight_set );
                belief_set = sampleBeliefs( pomdp , 500 );
                [TAO vL sol] = runPBVILean( pomdp , belief_set , 25 );
                [ a val ] = getAction( pomdp.start , sol.Vtable );
                param_set.mean_model_value_cache( end + 1 , : ) = ...
                    [ weight_set val ];
            end
    end
    
% ----- else compute intermediate values ----- %    
else
    
    % compute the current reward
    for i = 1:numel( sample_set )
        sample_reward( i ) = sample_set{ i }.bel' * ...
            sample_set{ i }.pomdp.reward( : , a );
    end
    switch param_set.action_selection_type
        case 'beb'
            na = get_beb_count( sample_set , weight_set , a );
            current_reward = sum( sample_reward .* weight_set ) + param_set.beb_constant / ( na + 1 );
        case 'boss'
            current_reward = max( sample_reward );
        case 'stochastic_forward_search'
            current_reward = sum( sample_reward .* weight_set );
    end
    
    % forward sample future rewards
    p_obs = [];
    tmp_sample_set = updateBeliefSet( sample_set , a , [] );
    for i = 1:numel( sample_set )
        p_obs( : , i ) = transpose( tmp_sample_set{ i }.bel' * squeeze( ...
            tmp_sample_set{ i }.pomdp.observation( : , a , : ) ) );
    end
    switch param_set.action_selection_type
        case {'beb', 'stochastic_forward_search'}
            used_obs_p_set = []; new_p_obs = p_obs;
            for branch_index = 1:param_set.branch_count
                if sum( new_p_obs * weight_set' ) > 0
                    my_dist = new_p_obs * weight_set' / sum( new_p_obs * weight_set' );
                    obs = sample_multinomial( my_dist );
                    used_obs_p_set = [ used_obs_p_set new_p_obs( obs ) ];
                    new_p_obs( obs , : ) = 0;
                    
                    % update weights
                    forward_weight_set = weight_set;
                    forward_weight_set = forward_weight_set .* p_obs( obs , : );
                    forward_weight_set = forward_weight_set / sum( forward_weight_set );
                    
                    % update beliefs within the pomdps
                    new_sample_set = updateBeliefSet( sample_set , a , obs );
                    
%                     % decide on which actions to branch on based on the most
%                     % promising, bias towards the best actions by each model!
%                     for i = 1:numel( sample_set )
%                         [ bestA( i ) q( i ) ] = getAction( new_sample_set{ i }.bel' , new_sample_set{ i }.sol.Vtable );
%                     end
%                     [ sort_q forward_a_set ] = sort( q , 'descend' );
%                     forward_a_set = bestA( forward_a_set );
%                     forward_a_set = forward_a_set( 1:min( length( forward_a_set ) , param_set.branch_count ) );
%                     forward_a_set = unique( forward_a_set );
%                     action_count_deficit = param_set.branch_count - length( forward_a_set );
%                     if action_count_deficit > 0
%                         remaining_action_set = setdiff( 1:sample_set{ 1 }.pomdp.nrActions , forward_a_set );
%                         remaining_action_index_set = randperm( length( remaining_action_set ) );
%                         remaining_action_set = remaining_action_set( ...
%                             remaining_action_index_set( 1:min( length( remaining_action_set ) , action_count_deficit ) ) );
%                         forward_a_set = [ forward_a_set remaining_action_set ];
%                     end
                    forward_a_set = 1:sample_set{ 1 }.pomdp.nrActions;
                    for forward_a_ind = 1:numel( forward_a_set )
                        [ forward_value_set( forward_a_ind ) param_set ] = score_action( new_sample_set , ...
                            forward_weight_set , forward_a_set( forward_a_ind ) , ...
                            depth - 1 , param_set );
                    end
                    forward_value( branch_index ) = max( forward_value_set );
                end
            end
            used_obs_p_set = used_obs_p_set / sum( used_obs_p_set );
            val = current_reward + sample_set{ 1 }.pomdp.gamma * sum( used_obs_p_set .* forward_value );
            
        % for the boss case, we max instead of average weights -- finale! hack for sampling which observations!    
        case 'boss'
            used_obs_p_set = []; new_p_obs = mean( p_obs , 2 );
            for branch_index = 1:param_set.branch_count
                if sum( new_p_obs ) > 0
                    obs = sample_multinomial( new_p_obs );
                    used_obs_p_set = [ used_obs_p_set ; p_obs( obs , : ) ];
                    new_p_obs( obs , : ) = 0;
                    
                    % update beliefs within the pomdps and try out
                    % actions on the updated beliefs
                    new_sample_set = updateBeliefSet( sample_set , a , obs );
                    forward_a_set = 1:sample_set{ 1 }.pomdp.nrActions;
                    for forward_a_ind = 1:numel( forward_a_set )
                        [ forward_value_set( forward_a_ind ) param_set ] = score_action( new_sample_set , ...
                            weight_set , forward_a_set( forward_a_ind ) , ...
                            depth - 1 , param_set );
                    end
                    forward_value( branch_index ) = max( forward_value_set );
                end
            end
            for model_index = 1:numel( sample_set )
                normalized_used_obs_p_set = used_obs_p_set( : , model_index );
                normalized_used_obs_p_set = normalized_used_obs_p_set / sum( normalized_used_obs_p_set );
                val( model_index ) = current_reward + sample_set{ 1 }.pomdp.gamma * ( forward_value * normalized_used_obs_p_set );
            end
            val = max( val );
    end
end

%-------------------------------------------------------------------------%
function count = get_beb_count( sample_set , weight_set , a )
count = 0;

% use the belief to compute the number of times action has been tried
for sample_index = 1:numel( sample_set )
    size( sample_set{ sample_index }.bel );
    size( sample_set{ sample_index }.nsa( : , a ) );
    weight_set( sample_index )
    count = count + weight_set( sample_index ) * ...
        sample_set{ sample_index }.bel' * sample_set{ sample_index }.nsa( : , a );
end

%-------------------------------------------------------------------------%
function Q = getQLean( bel, sample_set, i, actionSet );

% there are only going to be a few relevant actions for our beliefs
actionCount = length(actionSet);
for a = 1:actionCount
    action = actionSet(a);
    [ val ind ] = max( bel * sample_set{ i }.sol.Q(:,:,a)' );
    Q( 1 , a ) = val;
end

% %-------------------------------------------------------------------------%
% function Q = getQ( bel, sample_set, i, actionSet );
% alphaCount = size( sample_set{ i }.sol.Vtable{end}.alphaList , 2 );
% perm = randperm( alphaCount );
% perm = perm( 1 : ceil(sqrt( alphaCount )) );
% 
% % there are only going to be a few relevant actions for our beliefs
% actionCount = length(actionSet);
% for a = 1:actionCount
%     action = actionSet(a);
%     actionInd(a) = find( 1:sample_set{ i }.pomdp.nrActions == action );
% end
% 
% % first compute gammoAO, (return if are in state s, see o, do a) --
% % sort to get the permutation with the value that is most likely to be
% % helpful in the future.
% TAO = sample_set{i}.TAO;
% gammaAO = cell(actionCount,1);
% for a = 1:actionCount
%     aind = actionInd(a);
%     for obs = 1:sample_set{ i }.pomdp.nrObservations
%         gammaAO{a}{obs} = TAO{aind}{obs} * sample_set{ i }.sol.Vtable{end}.alphaList(:,perm);
%     end
% end
% 
% % next pick q vectors for each belief
% for a = 1:actionCount
%     action = actionSet(a);
%     
%     % add expected return for each obs + action
%     gammaAB = sample_set{i}.pomdp.reward(:,action);
%     for obs = 1:sample_set{ i }.pomdp.nrObservations
%         [ vals ind ] = max( bel * gammaAO{a}{obs},[],2 );
%         gammaAB = gammaAB + gammaAO{a}{obs}(:, ind );
%     end
%     Q(a) = bel * gammaAB;
% end
