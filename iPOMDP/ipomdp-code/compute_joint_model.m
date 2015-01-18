function joint_model = compute_joint_model( pomdp_set )

% stuff we know
joint_model.gamma = pomdp_set{ 1 }.pomdp.gamma;
joint_model.nrObservations = pomdp_set{ 1 }.pomdp.nrObservations;
joint_model.nrActions = pomdp_set{ 1 }.pomdp.nrActions;

% stuff to add incrementally
joint_model.nrStates = 0;
joint_model.reward = [];
joint_model.observation = [];
joint_model.transition = [];
joint_model.start = {};

% go through all models
for model_index = 1:numel( pomdp_set )
    model_state_count = pomdp_set{ model_index }.pomdp.nrStates;
    joint_model.start{ model_index } = pomdp_set{ model_index }.pomdp.start;
    joint_model.reward = [ joint_model.reward ; pomdp_set{ model_index }.pomdp.reward ];
    joint_model.observation = [ joint_model.observation ; pomdp_set{ model_index }.pomdp.observation ];
    joint_model.transition = [ joint_model.transition ...
        zeros( [ joint_model.nrStates model_state_count joint_model.nrActions ] ) ; ...
        zeros( [ model_state_count joint_model.nrStates joint_model.nrActions ] ) ...
        pomdp_set{ model_index }.pomdp.transition ];
    joint_model.nrStates = joint_model.nrStates + model_state_count;
end
joint_model.maxReward = max( joint_model.reward(:) );



