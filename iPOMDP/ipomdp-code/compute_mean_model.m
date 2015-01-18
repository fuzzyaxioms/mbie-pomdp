function mean_model = compute_mean_model( aligned_pomdp_set , pomdp_weight_set )
mean_model = aligned_pomdp_set{ 1 };
transition_sum = aligned_pomdp_set{ 1 }.transition * pomdp_weight_set( 1 );
observation_sum = aligned_pomdp_set{ 1 }.observation * pomdp_weight_set( 1 );
reward_sum = aligned_pomdp_set{ 1 }.reward * pomdp_weight_set( 1 );
start_sum = aligned_pomdp_set{ 1 }.start * pomdp_weight_set( 1 );
for pomdp_index = 2:numel( aligned_pomdp_set )
    transition_sum = transition_sum + aligned_pomdp_set{ pomdp_index }.transition * pomdp_weight_set( pomdp_index );
    observation_sum = observation_sum + aligned_pomdp_set{ pomdp_index }.observation * pomdp_weight_set( pomdp_index );
    start_sum = start_sum + aligned_pomdp_set{ pomdp_index }.start * pomdp_weight_set( pomdp_index );
    reward_sum = reward_sum + aligned_pomdp_set{ pomdp_index }.reward * pomdp_weight_set( pomdp_index );
end
mean_model.transition = transition_sum;
mean_model.observation = observation_sum;
mean_model.reward = reward_sum;
mean_model.start = start_sum;