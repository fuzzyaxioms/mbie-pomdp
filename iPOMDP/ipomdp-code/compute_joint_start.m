function start = compute_joint_start( joint_model , weight_set )

% build the start vector
start = [];
for model_index = 1:numel( weight_set )
    start = [ start ( weight_set( model_index ) * joint_model.start{ model_index } ) ];
end
