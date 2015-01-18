function [ history utree_set node_count_set ] = testUTREESet( sim_pomdp , utree_set , test_param_set , model_param_set )
% function [ history utree_set ] = testUTREESet( sim_pomdp , utree_set , test_param_set , model_param_set )
% Runs the utree algorithm, based on various papers/the original
% description in Andrew McCallum's thesis.  There are many parameters
% and decisions regarding the hypothesis tests.  I have optimized the
% best I could using tiger and 4x3 gridworld as my test domains, but
% PLEASE BE AWARE THERE ARE MANY HARD-CODED PARAMETERS HERE!

% general initialization stuff
reward = 0; 
iter_count = 0; 
state = sample_multinomial( sim_pomdp.start_dist , 1 );

% loop until maxed or done
curr_leaf_node = utree_set.curr_leaf_node;
keep_running = true;
while( keep_running )

    % --- basic action/observation --- %
    % get an action 
    [ min_count min_action ] = min( utree_set.state_action_count_set( curr_leaf_node , : ) );
    if min_count < 5
        action = min_action;
    else
        if rand < test_param_set.r_epsilon
            action = ceil( rand * sim_pomdp.nrActions );
        else
            [ value action ] = max( utree_set.qtable( curr_leaf_node , : ) );
        end
    end
   
    % store your action, q variance
    history(iter_count+1, 2) = action;
    history(iter_count+1, 5) = NaN; 
    history(iter_count+1, 6) = NaN;
    
    % sample a transition and update reward
    new_reward = sim_pomdp.reward( state, action );
    reward = reward + new_reward;
    if iscell( sim_pomdp.transition )
        state = sample_multinomial( sim_pomdp.transition{ action }( :, state ), 1 );
    else
        state = sample_multinomial( sim_pomdp.transition(:, state, action ), 1 );
    end
    
    % sample an observation from the current state
    if iscell( sim_pomdp.observation )
        obs = sample_multinomial( sim_pomdp.observation{ action }( state , :), 1 );
    else
        obs = sample_multinomial( sim_pomdp.observation( state, action, :), 1 );
    end
    
    % store the reward, state, and obs you received out
    history(iter_count+1, 4) = state;
    history(iter_count+1, 1) = obs;   
    history(iter_count+1, 3) = new_reward;
    
    % --- update the utree stuff --- % 
    % update the state_action_count_set
    sa_node = curr_leaf_node;
    while ~isnan( sa_node )
        utree_set.state_action_count_set( sa_node , action ) = 1 + ...
            utree_set.state_action_count_set( sa_node , action );
        sa_node = utree_set.tree.parent( sa_node );
    end
        
    % add the trajectory to the tree
    prev_leaf_node = curr_leaf_node;
    utree_set.trajectory_set = [ utree_set.trajectory_set ; action obs reward ];
    curr_leaf_node = find_leaf_node( utree_set.trajectory_set , utree_set.tree );
    utree_set.trajectory_assignment_set = [ utree_set.trajectory_assignment_set curr_leaf_node ];
    
    % update the transition and reward matrices -- note that the
    % transitions are definied (s,s',a) for ease of value iteration in
    % the next step (*not* how the pomdp transitions are defined!)
    [ utree_set.transition_count_set utree_set.transition_set ] = update_transition_set( ...
        utree_set.transition_count_set , utree_set.transition_set , ...
        prev_leaf_node , curr_leaf_node , action , utree_set.tree );
    [ utree_set.reward_count_set utree_set.reward_value_set utree_set.reward_set ] = update_reward_set( ...
        utree_set.reward_count_set , utree_set.reward_value_set , utree_set.reward_set , ...
        prev_leaf_node , reward , action , utree_set.tree );
        
    % do a few steps of value iteration
    value_iter_count = 3;
    for value_iter = 1:value_iter_count
        vtable = max( utree_set.qtable , [] , 2 );
        utree_set.qtable = utree_set.reward_set;
        for action_index = 1:sim_pomdp.nrActions
            utree_set.qtable( : , action_index ) = utree_set.qtable( : , action_index ) + ...
                sim_pomdp.gamma * utree_set.transition_set( : , : , action_index ) * vtable;
        end
    end
        
    % determine whether to adjust the tree
    test_leaf_node = curr_leaf_node;
    [ expand_fringe expand_depth ] = test_fringe( utree_set.trajectory_set , utree_set.trajectory_assignment_set , ...
        test_leaf_node , utree_set.tree , utree_set.qtable , sim_pomdp.gamma );
    if expand_fringe
        
        % adjust the tree
        if utree_set.tree.use_action_index( test_leaf_node )
            child_count_set = [ sim_pomdp.nrActions sim_pomdp.nrObservations ];
        else
            child_count_set = [ sim_pomdp.nrObservations sim_pomdp.nrActions ];
        end
        utree_set.tree = expand_utree( test_leaf_node , utree_set.tree , child_count_set , ...
            model_param_set.max_depth , expand_depth );
        
        % reassign trajectories for that leaf
        changed_leaf_set = find( utree_set.trajectory_assignment_set == test_leaf_node );
        for changed_index = 1:numel( changed_leaf_set )
            changed_leaf_index = changed_leaf_set( changed_index );
            leaf_node = find_leaf_node( utree_set.trajectory_set( 1:changed_leaf_index , : ) , utree_set.tree );
            utree_set.trajectory_assignment_set( changed_leaf_index ) = leaf_node;
        end
        
        % clear the transition and reward matrices
        node_count = length( utree_set.tree.is_leaf );
        action_count = sim_pomdp.nrActions;
        utree_set.transition_set = zeros( node_count , node_count , action_count );
        utree_set.transition_count_set = zeros( node_count , node_count , action_count );
        utree_set.reward_set = zeros( node_count , action_count );
        utree_set.reward_count_set = zeros( node_count , action_count );
        utree_set.reward_value_set = zeros( node_count , action_count );
        
        % update the transition and reward matrices with all the data
        for time_index = 2:length( utree_set.trajectory_assignment_set )
            prev_leaf_node = utree_set.trajectory_assignment_set( time_index - 1 );
            curr_leaf_node = utree_set.trajectory_assignment_set( time_index );
            node_action = utree_set.trajectory_set( time_index , 1 );
            node_reward = utree_set.trajectory_set( time_index , 3 );
            [ utree_set.transition_count_set utree_set.transition_set ] = update_transition_set( ...
                utree_set.transition_count_set , utree_set.transition_set , ...
                prev_leaf_node , curr_leaf_node , node_action , ...
                utree_set.tree );
            [ utree_set.reward_count_set utree_set.reward_value_set utree_set.reward_set ] = update_reward_set( ...
                utree_set.reward_count_set , utree_set.reward_value_set , utree_set.reward_set , ...
                prev_leaf_node , node_reward , node_action , ...
                utree_set.tree );
        end
        utree_set.reward_set = clean_reward_set( utree_set.reward_set , utree_set.reward_count_set , utree_set.tree );
        utree_set.transition_set = clean_transition_set( utree_set.transition_set , utree_set.transition_count_set , utree_set.tree );
        
        % adjust the qtable: copy over the values from the old leaf
        % to the new ones
        new_child_count = length( utree_set.tree.is_leaf ) - size( utree_set.qtable , 1 );
        start_index = size( utree_set.qtable , 1 ) + 1;
        end_index = size( utree_set.qtable , 1 ) + new_child_count;
        utree_set.qtable( start_index:end_index , : ) = repmat( utree_set.qtable( test_leaf_node , : ) , ...
            [ new_child_count 1 ] );
        value_iter_count = 3;
        for value_iter = 1:value_iter_count
            vtable = max( utree_set.qtable , [] , 2 );
            utree_set.qtable = utree_set.reward_set;
            for action_index = 1:sim_pomdp.nrActions
                utree_set.qtable( : , action_index ) = utree_set.qtable( : , action_index ) + ...
                    sim_pomdp.gamma * utree_set.transition_set( : , : , action_index ) * vtable;
            end
        end
        
        % expand the state_action_count_set
        utree_set.state_action_count_set = [ utree_set.state_action_count_set ; ...
            zeros( new_child_count , sim_pomdp.nrActions ) ];
    end
    
    % --- decide whether to continue running --- %
    % update the iters, reset if we get the max reward
    iter_count = iter_count + 1;
    keep_running = ( iter_count < test_param_set.max_iter_count );
    if new_reward == sim_pomdp.maxReward
        keep_running = false;
    end
end

% reset the utree current state for the next to the starting trajectory
% assignment (can vary based on if nodes are reassigned)
utree_set.current_leaf_node = utree_set.trajectory_assignment_set( 1 );
node_count_set = length( utree_set.tree.is_leaf );

% --------------- %
%   Basic Utils   %
% --------------- %
% ----------------------------------------------------------------------- %
function reward_set = remove_nan_reward( reward_set , rmax )
reward_set( isnan( reward_set(:) ) ) = rmax;

% ----------------------------------------------------------------------- %
function transition_set = build_transition_set( transition_count_set , leaf_vector )
transition_set = transition_count_set ./ repmat( sum( transition_count_set , 2 ) , ...
    [ 1 size( transition_count_set , 1 ) 1 ] );
node_count = size( transition_count_set , 1 );
action_count = size( transition_count_set , 3 );

% add vector to NaN columns
for start_node = 1:node_count
    for action = 1:action_count
        if isnan( transition_set( start_node , 1 , action ) )
            transition_set( start_node , : , action ) = leaf_vector;
        end
    end
end
    
% ----------------------------------------------------------------------- %
function tree_node = find_leaf_node( trajectory_set , tree )
% traverses the suffix tree, starting with the most recent observation,
% then the most recent action, etc.
tree_node = 1;
is_leaf = tree.is_leaf; 
if ~is_leaf( tree_node )
    child_set = tree.child_set; 
    use_action_index_set = tree.use_action_index;
    suffix_start = size( trajectory_set , 1 );
    use_action_index = use_action_index_set( tree_node );
end
while is_leaf( tree_node ) ~= 1
    if use_action_index
        branch_index = trajectory_set( suffix_start , 1 );
        suffix_start = suffix_start - 1;
    else
        branch_index = trajectory_set( suffix_start , 2 );
    end
    tree_node = child_set( tree_node , branch_index );
    use_action_index = use_action_index_set( tree_node );
    
    % break if too short
    if suffix_start == 0
        tree_node = NaN;
        break
    end
end

% ----------------------------------------------------------------------- %
function [ use_action_index suffix_depth ] = get_next_suffix_depth( tree, leaf_node )
suffix_depth = tree.suffix_depth( leaf_node );
use_action_index = tree.use_action_index( leaf_node );
if use_action_index
    suffix_depth = suffix_depth + 1;
end
use_action_index = ~use_action_index;

% ----------------------------------------------------------------------- %
function next_suffix_set = get_next_suffix_set( trajectory_set , ...
    leaf_set , suffix_depth , use_action_index , expand_depth )
% NOTE: for now the depth can only be 1 or 2
if use_action_index
    next_suffix_set = trajectory_set( leaf_set - suffix_depth , 1 );
    if expand_depth == 2
        next_suffix_set = [ next_suffix_set trajectory_set( leaf_set - suffix_depth - 1 , 2 ) ];
    end
else
    next_suffix_set = trajectory_set( leaf_set - suffix_depth , 2 );
    if expand_depth == 2
        next_suffix_set = [ next_suffix_set trajectory_set( leaf_set - suffix_depth , 1 ) ];
    end
end

% -------------- %
%    Specifics   %
% -------------- %
% ----------------------------------------------------------------------- %
function [ expand_fringe expand_depth ] = test_fringe( trajectory_set , ...
    trajectory_assignment_set , test_leaf_node , tree , qtable , gamma )
% NOTE: we just try two search depths: one/two items in the past
expand_fringe = false; expand_depth = 0;
if sum( trajectory_assignment_set == test_leaf_node ) < 10
    disp( 'not enough data in the leaves' )
    return
end

% determine max depth for looking at suffixes
max_suffix_depth = max( tree.suffix_depth ) + 1;

% find all nodes with the same state as the specified test leaf, ignore the
% most recent trajectory because we don't have future information for it
all_changed_leaf_set = find( trajectory_assignment_set( 1: end-1 ) == test_leaf_node );
all_changed_leaf_set( all_changed_leaf_set <= max_suffix_depth ) = [];

% loop over future actions to compute the q-values
next_action_set = trajectory_set( all_changed_leaf_set + 1 , 1 );
unique_action_set = unique( next_action_set );
for action_index = 1:length( unique_action_set )
    action = unique_action_set( action_index );
    changed_leaf_set = all_changed_leaf_set( next_action_set == action );
    
    % NOTE: arbitrary cut-off for fringe!
    if numel( changed_leaf_set ) > 0
        
        % compute their q-values = reward_i + gamma * v( i+1 ); so shift the
        % indices in the assignment set by one to see where we've ended up
        vtable = max( qtable , [] , 2 );
        v_set = gamma * vtable( trajectory_assignment_set( changed_leaf_set + 1 ) );
        q_value_set = trajectory_set( changed_leaf_set + 1 , 3 ) + v_set(:);
        
        % compute the next values - try for depth one and depth two in the past
        for test_expand_depth = 1:2
            next_suffix_set = get_next_suffix_set( trajectory_set , changed_leaf_set , ...
                tree.suffix_depth( test_leaf_node ) , tree.use_action_index( test_leaf_node ) , ...
                test_expand_depth );
            
            % compute k-s statistic for each of the next values vs the full set
            if expand_fringe == false
                [ unique_suffix_set tmp unique_suffix_index_set ] = unique( next_suffix_set , 'rows' );
                if size( unique_suffix_set , 1 ) > 1
                    for unique_index1 = 1:size( unique_suffix_set , 1 )
                        for unique_index2 = ( unique_index1 + 1 ):size( unique_suffix_set , 1 )
                            q_next1 = q_value_set( unique_suffix_index_set == unique_index1 );
                            q_next2 = q_value_set( unique_suffix_index_set == unique_index2 );
                            
                            
                            h = kstest2( q_next1 , q_next2 , .001 );
                            if h == 1
                                expand_fringe = true;
                                expand_depth = test_expand_depth;
                            end
                        end
                    end
                end
            end
        end
    end
end

% ----------------------------------------------------------------------- %
function tree = expand_utree( test_leaf_node , tree , child_count_set , ...
    max_depth , expand_depth )

% if the tree is too deep, don't expand
if tree.suffix_depth( test_leaf_node ) < max_depth
    new_child_set = [];

    % make the leaf node no longer a leaf node
    tree.is_leaf( test_leaf_node ) = false;
    
    % attach children to the leaf nodes, make nodes for them
    node_index = length( tree.is_leaf ) + 1;
    [ use_action_index suffix_depth ] = get_next_suffix_depth( tree, test_leaf_node );
    for child_index = 1:child_count_set( 1 )
        tree.child_set( test_leaf_node , child_index ) = node_index;
        tree.use_action_index( node_index ) = use_action_index;
        tree.suffix_depth( node_index ) = suffix_depth;
        tree.parent( node_index ) = test_leaf_node;
        tree.is_leaf( node_index ) = true;
        if expand_depth > 1
            tree = expand_utree( node_index , tree , child_count_set(2:end) , ...
                max_depth , expand_depth - 1 );
        end
        node_index = length( tree.is_leaf ) + 1;
    end
end

% ----------------------------------------------------------------------- %
function [ reward_count_set reward_value_set reward_set ] = ...
    recompute_reward_set( trajectory_set , trajectory_assignment_set , ...
    test_leaf_reward , node_count , action_count )
reward_count_set = zeros( node_count , action_count );
reward_value_set = zeros( node_count , action_count );

% loop through the trajectory assignment set: note that we want to know the
% reward that resulted in taking the action from a state -- which is
% recorded as the data for the next state -- so actions and rewards need to
% be shifted by one ( unshifted are the ones that happened in the past )
for node_index = 1:( length( trajectory_assignment_set ) - 1 )
    action = trajectory_set( node_index + 1 , 1 );
    reward = trajectory_set( node_index + 1 , 3 );
    node = trajectory_assignment_set( node_index );
    if ~isnan( node )
        reward_count_set( node , action ) = reward_count_set( node , action ) + 1;
        reward_value_set( node , action ) = reward_value_set( node , action ) + reward;
    end
end
reward_set = reward_value_set ./ reward_count_set;

% check for any nans and add in the test values
for node = 1:size( reward_set , 1 )
    for action = 1:size( reward_set , 2 )
        if isnan( reward_set( node , action ) )
            reward_set( node , action ) = test_leaf_reward( action );
        end
    end
end

% ----------------------------------------------------------------------- %
function [ transition_count_set transition_set ] = ...
    recompute_transition_set( trajectory_set , test_leaf_transition , ...
    trajectory_assignment_set , node_count , action_count )
transition_count_set = zeros( [ node_count node_count action_count ] );

% loop through the trajectory assignment set -- note that we use the action
% for the next node, because that is the action that happened after the
% current node and resulted in the next node happening
for node_index = 1:( length( trajectory_assignment_set ) - 1 )
    node = trajectory_assignment_set( node_index );
    next_node = trajectory_assignment_set( node_index + 1 );
    action = trajectory_set( node_index + 1 , 1 );
    if ( ~isnan( node ) && ~isnan( next_node ) )
        transition_count_set( node , next_node , action ) = 1 + ...
            transition_count_set( node , next_node , action );
    end
end

% adjust only the new nodes
leaf_vector = leaf_set / sum( leaf_set );
transition_set = build_transition_set( transition_count_set , leaf_vector );

% ----------------------------------------------------------------------- %
function [ transition_count_set transition_set ] = update_transition_set( ...
    transition_count_set , transition_set , ...
    prev_leaf_node , curr_leaf_node , action , ...
    tree )

% add the count for the prev_leaf_node and all its parents
parent = tree.parent;
start_node = prev_leaf_node;
if ~isnan( curr_leaf_node )
    while ~isnan( start_node )
        
        transition_count_set( start_node , curr_leaf_node , action ) = ...
            transition_count_set( start_node , curr_leaf_node , action ) + 1;
        transition_set( start_node , : , action ) = transition_count_set( start_node , : , action ) / ...
            sum( transition_count_set( start_node , : , action ) );
        start_node = parent( start_node );
    end
end

% ----------------------------------------------------------------------- %
function [ reward_count_set reward_value_set reward_set ] = update_reward_set( ...
    reward_count_set , reward_value_set , reward_set , ...
    prev_leaf_node , reward , action , ...
    tree )
        
% add the count for the prev_leaf_node and all its parents
parent = tree.parent;
start_node = prev_leaf_node;
while ~isnan( start_node )
    reward_count_set( start_node , action ) = ...
        reward_count_set( start_node , action ) + 1;
    reward_value_set( start_node , action ) = ...
        reward_value_set( start_node , action ) + reward;
    reward_set( start_node , action ) = ...
        reward_value_set( start_node , action ) / reward_count_set( start_node , action );
    start_node = parent( start_node );
end

% ----------------------------------------------------------------------- %
function reward_set = clean_reward_set( reward_set , reward_count_set , tree )

% go through all the leaves, and make sure that they have values filled in
parent_set = tree.parent;
is_leaf = tree.is_leaf;
for node = 1:numel( is_leaf )
    if is_leaf( node )
        for action = 1:size( reward_set , 2 )
            count_node = node;
            while ( ~isnan( count_node ) && ( reward_count_set( count_node , action ) == 0 ) )
                count_node = parent_set( count_node );
            end
            if isnan( count_node )
                reward_set( node , action ) = max( reward_set( : ) );
            elseif count_node ~= node
                reward_set( node , action ) = reward_set( count_node , action );
            end
        end
    end
end

% ----------------------------------------------------------------------- %
function transition_set = clean_transition_set( transition_set , transition_count_set , tree )

% go through all the leaves, and make sure that they have values filled in
parent_set = tree.parent;
is_leaf = tree.is_leaf;
for node = 1:numel( is_leaf )
    if is_leaf( node )
        for action = 1:size( transition_set , 3 )
            count_node = node;
            while ( ~isnan( count_node ) && ( sum( transition_count_set( count_node , : , action ) ) == 0 ) )
                count_node = parent_set( count_node );
            end
            if isnan( count_node )
                mean_transition_count = sum( sum( transition_count_set( is_leaf == 1 , : , : ) , 1 ) , 3 );                
                transition_set( node , : , action ) = mean_transition_count / sum( mean_transition_count );
            elseif count_node ~= node
                transition_set( node , : , action ) = transition_set( count_node , : , action );
            end
        end
    end
end



