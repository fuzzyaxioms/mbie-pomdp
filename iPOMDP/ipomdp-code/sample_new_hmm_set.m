function [ hmm_set stats ] = sample_new_hmm_set( obs_set , action_set , reward_set , ...
    hyper_set , param_set )
% function hmm_set = sample_new_hmm_set( obs_set , action_set , reward_set , ...
%    hypers , param_set , hmm_set , init_hmm );
if param_set.state_count < Inf
    [ hmm_set stats ] = IOHmmSampleBeam( obs_set, action_set, reward_set, ...
        hyper_set, param_set );
else
    [ hmm_set stats ] = iIOHmmSampleBeam( obs_set, action_set, reward_set, ...
        hyper_set , param_set );
    
    % extend the count matrix to make the dimensionality the same
    if isfield( hmm_set{ 1 } , 'nssa' )
        for hmm_index = 1:numel( hmm_set )
            hmm_set{ hmm_index }.nssa( end + 1 , end + 1 , : ) = 0;
        end
    end
end

