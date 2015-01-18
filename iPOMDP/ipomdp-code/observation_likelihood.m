function ll = observation_likelihood( fsc , obs_set , action_set )
% function ll = observation_likelihood( fsc , obs_set , action_set )
ll = 0;
for rep = 1:numel( action_set )
    ll = ll + single_episode_likelihood( fsc , obs_set{ rep } , action_set{ rep } );
end

% ----------------------------------------------------------------------- %
function ll = single_episode_likelihood( fsc , obs_set , action_set )
ll = 0;

% initialize the belief over nodes and loop
belief = fsc.start';
for time_index = 1:length( action_set )
    
    % get the policy output probability
    ll = ll + log( sum( belief .* fsc.policy( : , action_set( time_index ) ) ) ); 
    
    % update the belief given the action, obs
    belief = belief .* fsc.policy( : , action_set( time_index ) );
    belief = fsc.transition( : , : , obs_set( time_index ) ) * belief;
    belief = belief / sum( belief );
end
