function pomdp = hmm2pomdp( hmm , hypers , param_set );
% function pomdp = hmm2pomdp( hmm );
%  - hmm is a sample from the beam sampler
%  - hypers is the hypers from the beam sampler
%  - pomdp is in simplified spaan format
% hardcoded parameters
%  - pomdp starts in state 1
%  - discount factor is .95
%  - unobserved rewards are set to 0
is_infinite = size( hmm.Pi , 1 ) ~= size( hmm.Pi , 2 );

% basic paramaters
pomdp.nrActions = size( hmm.Phi , 3 );
pomdp.nrStates = size( hmm.Pi , 2 );
pomdp.nrObservations = size( hmm.Phi , 2 );
pomdp.gamma = .95;
pomdp.start = hmm.Pi0';
pomdp.start_dist = pomdp.start';

% info from hmm for transitions and observations
pomdp.transition = permute( hmm.Pi , [2 1 3] );
pomdp.observation = permute( hmm.Phi , [1 3 2] );

% add row if we have the ihmm case
if is_infinite
    for a = 1:pomdp.nrActions
        pomdp.transition( : , pomdp.nrStates , a ) =  hmm.Beta;
        pomdp.observation( pomdp.nrStates , a , : ) = hypers.H / sum( hypers.H );
    end
end

% reward parameters
reward_prob = permute( hmm.PhiR , [1 3 2] );
if is_infinite
    for a = 1:pomdp.nrActions
        reward_prob( pomdp.nrStates , a , : ) = hypers.HR / sum( hypers.HR );
    end
end
for s = 1:pomdp.nrStates
    for a = 1:pomdp.nrActions
        pomdp.reward( s , a ) = sum( squeeze( reward_prob( s , a , : ) ) ...
            .* param_set.unique_reward_set );
    end
end
pomdp.maxReward = max( pomdp.reward(:) );
pomdp.minReward = min( pomdp.reward(:) );