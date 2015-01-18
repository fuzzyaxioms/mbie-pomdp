function [S, stats ] = iIOHmmSampleBeam(Y, I, R, hyper_set, param_set)
% IHMMSAMPLEBEAM Samples states from the iIO-HMM with multinomial output
% using the Beam sampler.
%
% [S, stats] = iHmmSampleBeam( Y , I , R , hyper_set , param_set ) uses the
% beam sampling training algorithm for the infinite HMM.
%
%   Input Parameters:
%   - Y: cell array with output sequence of arbitrary length,
%   - I: cell array with input sequence, same length as the training sequence,
%   - R: cell array with reward sequence, same length as the training sequence
%   - hyper_set: a structure that describes the hyperparameters for the beam
%             sampler.
%             * If this structure contains alpha0 and gamma, it will
%               not resample these during sampling.
%             * If alpha0 and gamma are not specified, one needs to specify
%               hyperparameters for alpha0 and gamma (alpha0_a, alpha0_b, gamma_a, gamma_b).
%             * H: prior for the emission alphabet
%             * HR: prior for the reward alphabet
%             * M: param_set.burnin_counter of input symbols
%   - param_set: a structure that contains
%             * param_set.burnin_count: the param_set.burnin_counter of burnin iterations,
%             * param_set.model_count: the param_set.burnin_counter of samples to output,
%             * param_set.inter_iter_count: the param_set.burnin_counter of iterations between samples
%             * S0: is the initial assignment to the sequence; it should be
%               one longer than Y/I/R (s0, s1, s2..) vs. (y1,y2,...)
%             * sample.K: param_set.burnin_counter of states
%             * model: struct with a Pi, Pi0, Phi, PhiR for initialisation
%             * sample_s: if true, state sequence done with ffbs, else fb
%             * sample_m: if true, samples model, else maximises
%
%   Output Parameters:
%   - S: is a cell array of sample structures where each sample contains the
%        hidden state sequence S, the param_set.burnin_counter of states K, the Beta, Pi,
%        Phi's used for that sample.

% ------------------- %
%    Set Key Flags    %
% ------------------- %
% whether we want to train with rewards (depend on current state, current 
% action) and observations (depend on next state, current action)
use_reward_output = ~isempty( R );
use_obs_output = ~isempty( Y );
if isfield( param_set , 'train_with_obs' )
    use_obs_output = param_set.train_with_obs;
end

% inference flags: decide whether to use tempering
hyper_set.tempering_discount = 0;
use_tempering = false;
if isfield( param_set , 'use_tempering' )
    use_tempering = param_set.use_tempering;
end

% inference flags: decide whether to use an initial set of fake counts (can
% consider this as part of the prior.  note that these fake counts stay -- 
% don't get tempered away since they're part of the prior
if ( isfield( hyper_set , 'use_faux_count_set' ) && ...
        ( hyper_set.use_faux_count_set == true ) )
    faux_count_set = hyper_set.faux_count_set;
    if isfield( faux_count_set , 'Phi' )
       use_obs_output = true;
    end
    if isfield( faux_count_set , 'PhiR' )
        use_reward_output = true;
    end
else
    hyper_set.use_faux_count_set = false;
    faux_count_set.Phi = [];
    faux_count_set.Phi_flat = [];
    faux_count_set.PhiR = [];
    faux_count_set.PhiR_flat = [];
    faux_count_set.Pi = [];
end

% ------------------------------ %
%     Initialize the sampler     %
% ------------------------------ %
% Cache random variables
random_cache_length = 5000;
random_cache_set = rand( 1 , random_cache_length );
random_cache_index = 1;

% Setup structures to store the output %
S = {};
stats.K = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));
stats.alpha0 = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));
stats.gamma = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));

% --- Set initial state sequence --- %
episode_count = numel( I );
sample.S = {};
if ~isfield( param_set , 'S0' )
    if isfield( param_set , 'model' ) && ~isempty( param_set.model )
        sample.K = size( param_set.model.Pi, 1);
    else
        sample.K = 5;
    end
    for episode_index = 1:episode_count
        sample.S{ episode_index } = ceil( sample.K * rand( 1 , length( I{ episode_index } ) + 1 ) );
    end
    
% else set the state sequence as given and adjust the model to make it 
% have the same number of states    
else
    sample.S = param_set.S0;
    
    % set the number of instantiated states to the max
    sample.K = 0;
    for episode_index = 1:episode_count
        sample.K = max( sample.K , max( sample.S{ episode_index } ) );
    end
    
     % adjust the model if it is not consistent with number of states in
     % the state sequence provided
     if isfield( param_set , 'model' ) && ~isempty( param_set.model )
                  
         % make sample.K bigger if the model had some other states 
         if sample.K < size( param_set.model.Pi, 1)
             sample.K = size( param_set.model.Pi, 1);
         
         % make the model bigger if sample.K had some more states
         elseif sample.K > size( param_set.model.Pi, 1)

         end
     end
end

% ---- Initialize the Hyper Set ---- %
% Resample a few times as our inital guess might be off.
if isfield( param_set , 'model' ) && ~isempty( param_set.model )
    sample.alpha0 = param_set.model.alpha0;
    sample.gamma = param_set.model.gamma;
    sample.Beta = param_set.model.Beta;
else
    if isfield(hyper_set, 'alpha0')
        sample.alpha0 = hyper_set.alpha0;
    else
        sample.alpha0 = randg(hyper_set.alpha0_a) / hyper_set.alpha0_b;
    end
    if isfield(hyper_set, 'gamma')
        sample.gamma = hyper_set.gamma;
    else
        sample.gamma = randg(hyper_set.gamma_a) / hyper_set.gamma_b;
    end
    sample.Beta = ones( sample.K+1, 1 ) / (sample.K+1);
    for i=1:5
        [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hyper_set, 20);
    end
end

% --- Sample the emission and transition probabilities --- %
% copy over if given an initial model, else sample based on the state
% sequence/hyper parameters we just initialized
if isfield( param_set , 'model' ) && ~isempty( param_set.model )
    Pi = param_set.model.Pi;
    Pi0 = param_set.model.Pi0;
    if use_obs_output
        Phi = param_set.model.Phi;
        Phi_flat = param_set.model.Phi_flat;
    else
        Phi = [];
        Phi_flat = [];
    end
    if use_reward_output
        PhiR = param_set.model.PhiR;
        PhiR_flat = param_set.model.PhiR_flat;
    else
        PhiR = [];
        PhiR_flat = [];
    end
else
    Pi = UpdateTransitionMatrices( true , sample.alpha0 * sample.Beta , ...
        sample.S , I , hyper_set.action_count , [] , hyper_set , faux_count_set.Pi );
    Pi(sample.K+1,:,:) = [];
    if param_set.sample_start
        Pi0 = UpdateStartDistribution( true , sample.S , sample.alpha0 * ...
            sample.Beta , hyper_set );
    else
        Pi0 = zeros( length( sample.Beta ) , 1 );
        Pi0( 1 ) = 1;
    end
    if use_obs_output
        Phi = UpdateEmissionMatrices( true , sample.S , I , Y , ...
            sample.K , hyper_set.H , hyper_set.action_count , true , [] , hyper_set , faux_count_set.Phi );
        Phi_flat = UpdateFlatEmissionMatrices( true , sample.S , I , Y , ...
            sample.K , hyper_set.H , hyper_set.action_count , true , [] , hyper_set , faux_count_set.Phi_flat );
    else
        Phi = [];
        Phi_flat = [];
    end
    if use_reward_output
        PhiR = UpdateEmissionMatrices( true , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.action_count , false , [] , hyper_set , faux_count_set.PhiR );
        PhiR_flat = UpdateFlatEmissionMatrices( true , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.action_count , false , [] , hyper_set , faux_count_set.PhiR_flat );
    else
        PhiR = [];
        PhiR_flat = [];
    end
end

% ---- Build the First Sample ---- %
sample.Pi = Pi;
sample.Pi0 = Pi0;
if use_obs_output
    sample.Phi = Phi;
    sample.Phi_flat = Phi_flat;
end
if use_reward_output
    sample.PhiR = PhiR;
    sample.PhiR_flat = PhiR_flat;
end

% ------------------------------ %
%          Run Inference         %
% ------------------------------ %
iter = 1;
burnin_count = param_set.burnin_count; burnin_done = false;
while iter <= ( burnin_count + (param_set.model_count-1)*param_set.inter_iter_count )
    
    % check if tempering
    hyper_set.tempering_discount = 0;
    if use_tempering
        hyper_set.tempering_discount = max( 0 , .5 * ( 1 - iter / burnin_count ) );
    end
    
    % ---- Sample the auxilary variables ---- %
    clear u;
    for episode_index = 1:episode_count
        my_S = sample.S{ episode_index };
        my_I = I{ episode_index };
        rand_value = random_cache_set( random_cache_index );
        random_cache_index = random_cache_index + 1;
        if random_cache_index > random_cache_length
            random_cache_index = 1;
        end
        my_u( 1 ) = rand_value * Pi0( my_S( 1 ) );
        for t=2:numel( my_S )
            rand_value = random_cache_set( random_cache_index );
            random_cache_index = random_cache_index + 1;
            if random_cache_index > random_cache_length
                random_cache_index = 1;
            end
            my_u( t ) = rand_value * Pi( my_S( t - 1 ) , ...
                my_S( t ) , my_I( t - 1 ) );
        end
        u{ episode_index } = my_u;
    end
    
    % ---- Extend Pi , Phi as needed ---- %
    while max(max( Pi(:, end , : ) )) > min( cell2mat( u ) )
        pl = size(Pi, 2);
        bl = length(sample.Beta);
        
        % Safety check.
        assert(bl == pl);
        
        % Add row to transition , observation , reward matrix.
        for m = 1:hyper_set.action_count
            Pi(bl,:,m) = sample_dirichlet( sample.alpha0 * sample.Beta );
            if use_obs_output
                Phi(bl,:,m) = sample_dirichlet( hyper_set.H );
            end
            if use_reward_output
                PhiR(bl,:,m) = sample_dirichlet( hyper_set.HR );
            end
        end
        
        % Break beta stick 
        be = sample.Beta(end);
        bg = randbeta_fast( 1 , sample.gamma );
        sample.Beta(bl) = bg * be;
        sample.Beta(bl+1) = (1-bg) * be;
        
        % adjust transition matrix, using an approx for beta when a or b are
        % really small; binornd is faster than randbinom for multiple samples
        a = sample.alpha0 * sample.Beta(end-1);
        b = sample.alpha0 * (1 - sum(sample.Beta(1:end-1)));
        if ( a < 0.01 ) || ( b < 0.01 )
            pg = binornd( 1 , a./(a+b) , [ bl hyper_set.action_count ] );
        else
            pg = betarnd( a , b , [ bl hyper_set.action_count ] );
        end
        for m = 1:hyper_set.action_count
            pe = Pi( : , end , m );
            Pi(:, pl , m ) = pg( : , m ) .* pe;
            Pi(:, pl+1 , m ) = ( 1 - pg( : , m ) ) .* pe;
        end
        
        % adjust the prior distribution 
        if param_set.sample_start
            pe = Pi0( end );
            if ( a < 1e-2 ) || ( b < 1e-2 )
                pg = binornd( 1 , a./(a+b));
            else
                pg = randbeta_fast( a , b + pl );
            end
            Pi0(  pl ) = pg .* pe;
            Pi0( pl+1 ) = (1-pg) .* pe;
        else
            Pi0( end + 1 ) = 0;
        end
        
        % add stuffs to the fake counts as well
        if hyper_set.use_faux_count_set
            faux_count_set.Pi( : , end+1 , : ) = 0;
            faux_count_set.Pi( end+1 , : , : ) = 0;
            if use_obs_output
                faux_count_set.Phi( end+1 , : , : ) = 0;
                faux_count_set.Phi_flat( end+1 , : ) = 0;
            end
            if use_reward_output
                faux_count_set.PhiR( end+1 , : , : ) = 0;
                faux_count_set.PhiR_flat( end+1 , : ) = 0;
            end
        end
    end
    sample.K = size(Pi, 1);
    
    % Safety check.
    assert(sample.K == length(sample.Beta) - 1);
    
    % ---- Resample the hidden state sequence ---- %
    count_set.T = zeros( sample.K , sample.K , hyper_set.action_count );
    if use_obs_output
        count_set.Y = zeros( sample.K, length( hyper_set.H ) , hyper_set.action_count );
    end
    if use_reward_output
        count_set.R = zeros( sample.K, length( hyper_set.HR ) , hyper_set.action_count );
    end
    PiT = permute( Pi , [ 2 1 3 ] );
    for episode_index = 1:episode_count
        my_u = u{ episode_index };
        my_I = I{ episode_index };
        my_S = sample.S{ episode_index };
        if use_obs_output
            my_Y = Y{ episode_index };
        end
        if use_reward_output
            my_R = R{ episode_index };
        end
        T = length( my_S );
        dyn_prog = zeros( sample.K , T );
        
        % first step involves the prior distribution
        dyn_prog( : , 1 ) = Pi0( 1:sample.K ) > my_u( 1 );
        if use_reward_output
            dyn_prog( 1:sample.K , 1 ) = dyn_prog( 1:sample.K , 1 ) .* ...
                PhiR( 1:sample.K , my_R( 1 ) , my_I( 1 ) );
        end
        dyn_prog(:,1) = dyn_prog(:,1) / sum(dyn_prog(:,1));
        
        % later steps use the transitions
        for t=2:T
            A = PiT( 1:sample.K , 1:sample.K , my_I( t - 1 ) ) > my_u(t);
            dyn_prog( : , t ) = A * dyn_prog( : , t - 1 );
            if use_obs_output
                dyn_prog( 1:sample.K , t ) = dyn_prog( 1:sample.K , t ) .* ...
                    Phi( 1:sample.K , my_Y( t - 1 ) , my_I( t - 1 ) );
            end
            if use_reward_output && t~=T
                dyn_prog( 1:sample.K , t ) = dyn_prog( 1:sample.K , t ) .* ...
                    PhiR( 1:sample.K , my_R( t ) , my_I( t ) );
            end
            
            if sum( sum(dyn_prog(:,t)) ) ~= 0
                dyn_prog(:,t) = dyn_prog(:,t) / sum(dyn_prog(:,t));
            end
        end
        
        % Backtrack to sample a path through the HMM.
        if sum(dyn_prog(:,T)) ~= 0.0 && isfinite(sum(dyn_prog(:,T)))
            my_S(T) = 1 + sum(rand() > cumsum(dyn_prog(:,T)));
            for t=T-1:-1:1
                r = dyn_prog( : , t ) .* ( Pi( : , my_S( t+1 ) , my_I( t ) ) > my_u( t + 1 ) );
                r = r ./ sum(r);
                rand_value = random_cache_set( random_cache_index );
                random_cache_index = random_cache_index + 1;
                if random_cache_index > random_cache_length
                    random_cache_index = 1;
                end
                my_S(t) = 1 + sum( rand_value > cumsum(r) );
                
                % cache the counts
                count_set.T( my_S( t ) , my_S( t + 1 ) , my_I( t ) ) = ...
                    count_set.T( my_S( t ) , my_S( t + 1 ) , my_I( t ) ) + 1;
                if use_obs_output
                    count_set.Y( my_S( t + 1 ) , my_Y( t ) , my_I( t ) ) = ...
                        count_set.Y( my_S( t + 1 ) , my_Y( t ) , my_I( t ) ) + 1;
                end
                if use_reward_output
                    count_set.R( my_S( t ) , my_R( t ) , my_I( t ) ) = ...
                        count_set.R( my_S( t ) , my_R( t ) , my_I( t ) ) + 1;
                end
            end
            sample.S{ episode_index } = my_S;
        end
    end
    
    % ---- Cleanup our state space by removing redundant states ---- %
    active_state_set = unique( cell2mat( sample.S ) );
    if length( active_state_set ) < sample.K
        
        % adjust the zeros index set based on faux_count_set stuff
        state_map = zeros( 1 , sample.K ); state_map( active_state_set ) = 1:length( active_state_set );
        zero_ind = setdiff( 1:sample.K, active_state_set );
        if hyper_set.use_faux_count_set
            input_active_set = find( sum( sum( faux_count_set.Pi , 1 ) , 3 ) > 0 );
            if use_obs_output
                phi_active_set = find( sum( sum( faux_count_set.Phi , 2 ) , 3 ) > 0 );
                input_active_set = union( input_active_set , phi_active_set );
            end
            if use_obs_output
                phir_active_set = find( sum( sum( faux_count_set.PhiR , 2 ) , 3 ) > 0 );
                input_active_set = union( input_active_set , phir_active_set );
            end
            zero_ind = setdiff( zero_ind , input_active_set );
        end
        
        % actually do the deletion
        if ~isempty( zero_ind )
            sample.Beta(end) = sample.Beta(end) + sum( sample.Beta( zero_ind ) );
            sample.Beta(zero_ind) = [];
            Pi0( zero_ind ) = [];
            Pi( : , zero_ind , : ) = [];
            Pi( zero_ind , : , : ) = [];
            count_set.T( zero_ind , : , : ) = [];
            count_set.T( : , zero_ind , : ) = [];
            if use_obs_output
                Phi( zero_ind , : , : ) = [];
                count_set.Y( zero_ind , : , : ) = [];
            end
            if use_reward_output
                PhiR( zero_ind , : , : ) = [];
                count_set.R( zero_ind , : , : ) = [];
            end
            for episode_index = 1:episode_count
                sample.S{ episode_index } = state_map( sample.S{ episode_index } );
            end
            
            % clear stuff from the input sticks as well
            if hyper_set.use_faux_count_set
                zero_ind = zero_ind( zero_ind <= size(faux_count_set.Pi,2)-1 );
                faux_count_set.Pi( : , zero_ind , : ) = [];
                faux_count_set.Pi( zero_ind , : , : ) = [];
                if use_obs_output
                    faux_count_set.Phi( zero_ind , : , : ) = [];
                    faux_count_set.Phi_flat( zero_ind , : ) = [];
                end
                if use_reward_output
                    faux_count_set.PhiR( zero_ind , : , : ) = [];
                    faux_count_set.PhiR_flat( zero_ind , : ) = [];
                end
            end
        end
    end
    sample.K = size(Pi,1);
    
    % ---- Resample Beta given the transition probabilities ---- %
    [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample( sample.S, ...
        sample.Beta, sample.alpha0, sample.gamma, hyper_set, 20, sum( count_set.T , 3 ) );
    
    % ---- Update model parameters ---- %
    Pi = UpdateTransitionMatrices( param_set.sample_m , sample.alpha0 * sample.Beta , ...
        sample.S , I , hyper_set.action_count , count_set.T , hyper_set , faux_count_set.Pi );
    if param_set.sample_start
        Pi0 = UpdateStartDistribution( param_set.sample_m , sample.S , ...
            sample.alpha0 * sample.Beta , hyper_set );
    else
        Pi0 = zeros( length( sample.Beta ) , 1 );
        Pi0( 1 ) = 1;
    end
    if use_obs_output
        Phi = UpdateEmissionMatrices( param_set.sample_m , sample.S , I , Y , ...
            sample.K , hyper_set.H , hyper_set.action_count , true , count_set.Y , hyper_set , faux_count_set.Phi );
        Phi_flat = UpdateFlatEmissionMatrices( param_set.sample_m , sample.S , I , Y , ...
            sample.K , hyper_set.H , hyper_set.action_count , true , count_set.Y , hyper_set , faux_count_set.Phi_flat );
    end
    if use_reward_output
        PhiR = UpdateEmissionMatrices( param_set.sample_m , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.action_count , false , count_set.R , hyper_set , faux_count_set.PhiR );
        PhiR_flat = UpdateFlatEmissionMatrices( param_set.sample_m , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.action_count , false , count_set.R , hyper_set , faux_count_set.PhiR_flat );
    end
    Pi(sample.K+1,:,:) = [];
    
    % Safety checks
    assert(size(Pi,1) == sample.K);
    assert(size(Pi,2) == sample.K+1);
    assert(sample.K == length(sample.Beta) - 1);
    assert(min( Pi(:) ) >= 0);
    
    % ---- Save Stuff ---- %
    stats.alpha0(iter) = sample.alpha0;
    stats.gamma(iter) = sample.gamma;
    stats.K(iter) = sample.K;
    if iter >= burnin_count && mod( iter - burnin_count, param_set.inter_iter_count) == 0
        sample.Pi = Pi;
        sample.Pi0 = Pi0;
        if use_obs_output
            sample.Phi = Phi;
            sample.Phi_flat = Phi_flat;
        end
        sample.nssa = count_set.T;
        if use_reward_output
            sample.PhiR = PhiR;
            sample.PhiR_flat = PhiR_flat;
        end
        S{end+1} = sample;
    end
    iter = iter + 1;
end

% ----------------------------------------------------------------------- %
function [ Phi ] = UpdateEmissionMatrices( do_sample , S , I , Y, K, H , ...
    M , shift_forward , N , hyper_set , faux_count_set )
%UpdateEmissionMatrices Samples the emission matrices from a given state
% sequence S, a corresponding observation vector Y, an input sequence I
% a Dirichlet prior H, and the param_set.burnin_counter of possible input symbols A for a K
% state HMM.
L = size( H , 2 );
Phi = zeros(K,L,M);

% compute empirical counts
if isempty( N )
    N = zeros(K,L,M);
    shift_t = 0;
    if shift_forward
        shift_t = 1;
    end
    for i = 1:numel( S )
        my_S = S{ i };
        my_I = I{ i };
        my_Y = Y{ i };
        for t = 1:length( my_Y )
            N( my_S( t + shift_t ) , my_Y( t ) , my_I( t ) ) = 1 + ...
                N( my_S( t + shift_t ) , my_Y( t ) , my_I( t ) );
        end
    end
end

% sample Phi
for m=1:M
    for k=1:K
        
        % adjust H if hyper_set tells us to use some input pre-sticks and
        % if the input stick is big enough
        my_H = H;
        if hyper_set.use_faux_count_set
            try
                my_H = my_H + faux_count_set( k , : , m );
            end
        end
                
        % do the sampling
        if do_sample
            Phi(k,:,m) = sample_dirichlet( N(k,:,m) + my_H );
        else
            sumN = sum( N( k , : , m ) );
            if sumN > 0
                Phi(k,:,m) = N(k,:,m) / sumN;
            else
                Phi(k,:,m) = my_H / sum( my_H );
            end
        end
    end
end

% ----------------------------------------------------------------------- %
function [ Phi ] = UpdateFlatEmissionMatrices( do_sample , S , I , Y, K, H , ...
    M , shift_forward , N , hyper_set , faux_count_set )
%UpdateEmissionMatrices Samples the emission matrices from a given state
% sequence S, a corresponding observation vector Y, an input sequence I
% a Dirichlet prior H, and the param_set.burnin_counter of possible input symbols A for a K
% state HMM.
L = size( H , 2 );
Phi = zeros(K,L);

% compute empirical counts
if isempty( N )
    N = zeros(K,L,M);
    shift_t = 0;
    if shift_forward
        shift_t = 1;
    end
    for i = 1:numel( S )
        my_S = S{ i };
        my_I = I{ i };
        my_Y = Y{ i };
        for t = 1:length( my_Y )
            N( my_S( t + shift_t ) , my_Y( t ) , my_I( t ) ) = 1 + ...
                N( my_S( t + shift_t ) , my_Y( t ) , my_I( t ) );
        end
    end
end

% sample Phi
N = sum( N , 3 );
for k=1:K
    
    % adjust H if hyper_set tells us to use some input pre-sticks and if
    % the input sticks are big enough
    my_H = H;
    if hyper_set.use_faux_count_set
        try
            my_H = my_H + faux_count_set( k , : );
        end
    end
    
    % do the sampling
    if do_sample
        Phi(k,:) = sample_dirichlet( N(k,:) + my_H );
    else
        sumN = sum( N( k , : ) );
        if sumN > 0
            Phi(k,:) = N(k,:) / sumN;
        else
            Phi(k,:) = my_H / sum( my_H );
        end
    end
end

% ----------------------------------------------------------------------- %
function [ Pi ] = UpdateTransitionMatrices( do_sample , H , S , I , M , N ...
    , hyper_set , faux_count_set )
%UpdateTransitionMatrices Samples transition matrices from a state sequence
% S, an input sequence I, a Dirichlet prior H and the param_set.burnin_counter of possible
% input symbols M.
K = length( H );

% get counts
if isempty( N );
    N = zeros(K-1,K-1,M);
    for i = 1:numel( S )
        my_S = S{ i };
        my_I = I{ i };
        for t = 1:length( my_I )
            N( my_S( t ) , my_S( t + 1 ) , my_I( t ) ) = 1 + ...
                N( my_S( t ) , my_S( t + 1 ) , my_I( t ) );
        end
    end
end

% sample Pi
Pi = zeros(K,K,M);
for m=1:M
    for k=1:K-1

        % add any fake counts, as long as they exist
        my_H = H;
        if hyper_set.use_faux_count_set
            try
                my_H = my_H + faux_count_set( k , : , m );
            end
        end
        
        % do the sampling
        if do_sample
            Pi(k, :, m) = sample_dirichlet( [ N(k,:,m) 0 ] + my_H );
        else
            sumN = sum( N(k,:,m) );
            if sumN > 0
                Pi(k, :, m) = N(k,:,m) / sumN;
            else
                Pi(k,:,m ) = my_H / sum( my_H );
            end
        end
    end
    if do_sample
        Pi( K , :, m ) = sample_dirichlet( my_H );
    else
        Pi( K , : , m ) = my_H / sum( my_H );
    end
end

% ----------------------------------------------------------------------- %
function [ Pi0 ] = UpdateStartDistribution( do_sample , S , H , hyper_set )
% samples the start distribution

K = size(H,2);
Pi0 = zeros( K , 1 );
N = zeros( K , 1 );

% get counts
for i = 1:numel( S )
    N( S{ i }( 1 ) ) = 1 + N( S{ i }( 1 ) );
end
% sample Pi0
if do_sample
    Pi0( : , 1 ) = sample_dirichlet( N );
else
    Pi0( : , 1 ) = N / sum( N );
end

% ----------------------------------------------------------------------- %
function model = extend_model( model , K , use_obs_output , use_reward_output )
extra_state_count = K - size( model.Pi, 1);

% extend transitions
model.Pi0 = [ model.Pi0 ; zeros( extra_state_count , 1 ) ];
model.Pi( (end+1):(end+extra_state_count) , : , : ) = ...
    repmat( model.Pi( end , : , : ) , ...
    [ extra_state_count 1 1 ] );
extra_state_count = ( K + 1 )- size( model.Pi , 2 );
model.Pi( : , (end):(end + extra_state_count ) , : ) = ...
    repmat( model.Pi( : , end , : ) / ( extra_state_count + 1 ) , ...
    [ 1 ( extra_state_count + 1 ) 1 ] );
model.Beta( end:( end + extra_state_count ) ) = model.Beta( end ) / ( extra_state_count + 1 );

% extend observations
if use_obs_output
    model.Phi( (end+1):(end+extra_state_count ) , : , : ) = ...
        repmat( model.Phi( end , : , : ) , ...
        [ extra_state_count 1 1 ] );
    model.Phi_flat( (end+1):(end+extra_state_count ) , : ) = ...
        repmat( model.Phi_flat( end , : ) , ...
        [ extra_state_count 1 1 ] );
end

% extend rewards
if use_reward_output
    model.PhiR( (end+1):(end+extra_state_count ) , : , : ) = ...
        repmat( model.PhiR( end , : , : ) , ...
        [ extra_state_count 1 1 ] );
    model.PhiR_flat( (end+1):(end+extra_state_count ) , : ) = ...
        repmat( model.PhiR_flat( end , : ) , ...
        [ extra_state_count 1 1 ] );
end