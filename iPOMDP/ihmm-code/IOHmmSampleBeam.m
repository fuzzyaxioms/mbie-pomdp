function [ S stats ] = IOHmmSampleBeam(Y, I, R, hyper_set, param_set )
% IHMMSAMPLEBEAM Samples states from the iHMM with multinomial output
% using the Beam sampler.
%
% [S, stats] = iHmmSampleBeam(Y, I , R , hyper_set , param_set ) uses the
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
%             * M: number of input symbols
%   - param_set: a structure that contains
%             * param_set.burnin_count: the param_set.burnin_counter of burnin iterations,
%             * param_set.model_count: the param_set.burnin_counter of samples to output,
%             * param_set.inter_iter_count: the param_set.burnin_counter of iterations between samples
%             * S0: is the initial assignment to the sequence; it should be
%               one longer than Y/I/R (s0, s1, s2..) vs. (y1,y2,...)
%             * sample.K: number of states
%             * model: struct with a Pi, Pi0, Phi, PhiR for initialisation
%             * sample_s: if true, state sequence done with ffbs, else fb
%             * sample_m: if true, samples model, else maximises
%
%   Output Parameters:
%   - S: is a cell array of sample structures where each sample contains the
%        hidden state sequence S, the param_set.burnin_counter of states K, the Beta, Pi,
%        Phi's used for that sample.
param_set.use_reward_output = ~isempty( R );
param_set.use_obs_output = true;
if isfield( param_set , 'train_with_obs' )
    param_set.use_obs_output = param_set.train_with_obs;
end

% ------------------------------ %
%     Initialize the sampler     %
% ------------------------------ %

% Cache random variables
random_cache_length = 5000;
random_cache_set = rand( 1 , random_cache_length );
random_cache_index = 1;

% --- Set initial state sequence --- %
episode_count = numel( Y );
sample.K = param_set.state_count;
if isfield( param_set , 'S0' )
    sample.S = param_set.S0;
else
    for episode_index = 1:episode_count
          % copied from iIOHMMSampleBeam.m -Shayan, because the existing
          % code wasn't working.
          sample.S{ episode_index } = ceil( sample.K * rand( 1 , length( I{ episode_index } ) + 1 ) );
%         sample.S{ episode_index } = rand( sample.K , length( Y{ episode_index } ) + 1 );
%         sample.S{ episode_index } = bsxfun( @rdivide, sample.S{ episode_index } , ...
%             sum( sample.S{ episode_index } ) );
    end
end

% --- Setup structures to store the output --- %
S = {};
stats.alpha0 = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));
stats.gamma = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));
stats.sumPi = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));
stats.running_mean = zeros(1,(param_set.burnin_count + (param_set.model_count-1)*param_set.inter_iter_count));

% --- Initialize hyper_set --- %
% (resample a few times as our inital guess might be off)
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
sample.Beta = ones( 1 , sample.K ) / sample.K;

% --- Sample the emission and transition probabilities --- %
if isfield( param_set , 'model' )
    Phi = param_set.model.Phi;
    Pi = param_set.model.Pi;
    Pi0 = param_set.model.Pi0;
    if isfield( param_set.model , 'PhiR' )
        PhiR = param_set.model.PhiR;
    else
        PhiR = [];
    end
else
    Pi = UpdateTransitionMatrices( true , sample.alpha0 * sample.Beta , [], ...
        sample.S , I , [] , [] , [] , [] , [] , sample.K , hyper_set.M );
    Pi0 = UpdateStartDistribution( true , sample.S , sample.alpha0 * sample.Beta );
    Phi = UpdateEmissionMatrices( true , sample.S , I , Y , ...
        sample.K , hyper_set.H , hyper_set.M , true );
    if param_set.use_reward_output
        PhiR = UpdateEmissionMatrices( true , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.M , false );
    else
        PhiR = [];
    end
end

% ------------------------------ %
%          Run Inference         %
% ------------------------------ %
iter = 1;
keep_running = true; burnin_done = false; burnin_count = param_set.burnin_count;
while keep_running
    
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
%     
%     % Safety check.
%     assert(sample.K == length(sample.Beta) - 1);
    
    % ---- Resample the hidden state sequence ---- %
    count_set.T = zeros( sample.K , sample.K , hyper_set.action_count );
    if param_set.use_obs_output
        count_set.Y = zeros( sample.K, length( hyper_set.H ) , hyper_set.action_count );
    end
    if param_set.use_reward_output
        count_set.R = zeros( sample.K, length( hyper_set.HR ) , hyper_set.action_count );
    end
    PiT = permute( Pi , [ 2 1 3 ] );
    for episode_index = 1:episode_count
        my_u = u{ episode_index };
        my_I = I{ episode_index };
        my_S = sample.S{ episode_index };
        if param_set.use_obs_output
            my_Y = Y{ episode_index };
        end
        if param_set.use_reward_output
            my_R = R{ episode_index };
        end
        T = length( my_S );
        dyn_prog = zeros( sample.K , T );
        
        % first step involves the prior distribution
        dyn_prog( : , 1 ) = Pi0( 1:sample.K ) > my_u( 1 );
        if param_set.use_reward_output
            dyn_prog( 1:sample.K , 1 ) = dyn_prog( 1:sample.K , 1 ) .* ...
                PhiR( 1:sample.K , my_R( 1 ) , my_I( 1 ) );
        end
        dyn_prog(:,1) = dyn_prog(:,1) / sum(dyn_prog(:,1));
        
        % later steps use the transitions
        for t=2:T
            A = PiT( 1:sample.K , 1:sample.K , my_I( t - 1 ) ) > my_u(t);
            dyn_prog( : , t ) = A * dyn_prog( : , t - 1 );
            if param_set.use_obs_output
                dyn_prog( 1:sample.K , t ) = dyn_prog( 1:sample.K , t ) .* ...
                    Phi( 1:sample.K , my_Y( t - 1 ) , my_I( t - 1 ) );
            end
            if param_set.use_reward_output && t~=T
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
                if param_set.use_obs_output
                    count_set.Y( my_S( t + 1 ) , my_Y( t ) , my_I( t ) ) = ...
                        count_set.Y( my_S( t + 1 ) , my_Y( t ) , my_I( t ) ) + 1;
                end
                if param_set.use_reward_output
                    count_set.R( my_S( t ) , my_R( t ) , my_I( t ) ) = ...
                        count_set.R( my_S( t ) , my_R( t ) , my_I( t ) ) + 1;
                end
            end
            sample.S{ episode_index } = my_S;
        end
    end
    
    % --- Parameter update --- %
    % Resample hyper-parameters given the transition probabilities.
    if isempty( count_set.T )
        count_set.T = compute_transition_counts( fb_stat_set , I , Y , R , ...
            Pi , Phi , PhiR , sample.K , hyper_set.M );
    end
    if param_set.sample_m
        [sample.Beta, sample.alpha0, sample.gamma] = HmmHyperSample(...
           sum( count_set.T , 3 ) , sample.Beta, sample.alpha0, sample.gamma, hyper_set, 20);
    end
    
    % Update model parameters
    Pi = UpdateTransitionMatrices( param_set.sample_m , sample.alpha0 * sample.Beta , count_set.T , ...
             [], I , Y , R , Pi , Phi , PhiR , sample.K , hyper_set.M );
    Pi0 = UpdateStartDistribution( param_set.sample_m , sample.S , sample.alpha0 * sample.Beta );
    Phi = UpdateEmissionMatrices( param_set.sample_m , sample.S , I , Y , ...
        sample.K , hyper_set.H , hyper_set.M , true , count_set.Y );
    if param_set.use_reward_output
        PhiR = UpdateEmissionMatrices( param_set.sample_m , sample.S , I , R , ...
            sample.K , hyper_set.HR , hyper_set.M , false , count_set.R );
    end
    
    % Save samples
    stats.alpha0(iter) = sample.alpha0;
    stats.gamma(iter) = sample.gamma;
    stats.sumPi(iter) = sum( Pi(:) );
    if iter == 2
        stats.running_mean(iter) = ( stats.gamma( iter ) - stats.gamma( iter - 1 ) > 0 );
    elseif iter > 2
        stats.running_mean(iter) = ( iter - 2 )/( iter - 1 ) * stats.running_mean( iter - 1 ) + ...
            ( stats.gamma( iter ) - stats.gamma( iter - 1 ) > 0 ) / ( iter - 1 );
        if ~burnin_done && iter > 10
            if sum( abs( stats.running_mean( iter-4:iter ) - .5 ) < 0.01 ) == 5
                burnin_count = iter;
                stats.burnin_count = burnin_count;
                burnin_done = true;
                disp(['Done burnin ' num2str( burnin_count )]);
            end
        end
    end
    if iter >= burnin_count && mod(iter - burnin_count, param_set.inter_iter_count) == 0
        sample.Pi = Pi;
        sample.Pi0 = Pi0;
        sample.Phi = Phi;
        sample.nssa = count_set.T;
        if param_set.use_reward_output
            sample.PhiR = PhiR;
        end
        S{end+1} = sample;
    end
    
    % update iters and check for convergence
    iter = iter + 1;
    if ~( param_set.sample_m || param_set.sample_s )
        if iter > 2
            model_diff = sum(abs( Phi(:) - prev_Phi(:) )) + ...
                sum( abs( Pi(:) - prev_Pi(:) )) + ...
                sum( abs( Pi0(:) - prev_Pi0(:) ));
            model_param_count = sample.K^2 * hyper_set.M + ...
                sample.K + sample.K * hyper_set.M * numel( hyper_set.H );
            if param_set.use_reward_output
                model_diff = sum(abs( PhiR(:) - prev_PhiR(:) )) + model_diff;
                model_param_count = model_param_count + sample.K + sample.K * hyper_set.M * numel( hyper_set.HR );
            end
            keep_running = ( model_diff / model_param_count > .00001 );
            % disp(['IOHmmSampleBeam: EM Converged at iteration ' num2str( iter
            % ) ]);
        end
    else
        keep_running = ( iter <= ( burnin_count + ...
            (param_set.model_count-1)*param_set.inter_iter_count ) );
    end
    prev_Phi = Phi; prev_Pi = Pi; prev_Pi0 = Pi0;
    if param_set.use_reward_output; prev_PhiR = PhiR; end;
end
    
% ----------------------------------------------------------------------- %
function [ Phi ] = UpdateEmissionMatrices( do_sample , S , I , Y, K, H , M , shift_forward , N )
%UpdateEmissionMatrices Samples the emission matrices from a given state
% sequence S, a corresponding observation vector Y, an input sequence I
% a Dirichlet prior H, and the param_set.burnin_counter of possible input symbols A for a K
% state HMM.
L = size( H , 2 );
Phi = zeros(K,L,M);


% compute empirical counts
if nargin == 8
    N = zeros(K,L,M);
    shift_t = 0;
    if shift_forward
        shift_t = 1;
    end
    for i = 1:numel( S )
        my_S = S{ i };
        my_Y = Y{ i };
        my_I = I{ i };
        for t = 1:length( my_Y )
            N( : , my_Y( t ) , my_I( t ) ) = my_S( : , t + shift_t ) + ...
                N( : , my_Y( t ) , my_I( t ) );
        end
    end
end

% sample Phi
for m=1:M
    for k=1:K
        if do_sample
            Phi(k,:,m) = sample_dirichlet( N(k,:,m) + H);
        else
            sumN = sum( N( k , : , m ) );
            if sumN > 0
                Phi(k,:,m) = N(k,:,m) / sumN;
            else
                Phi(k,:,m) = H / sum( H );
            end
        end
    end
end

% ----------------------------------------------------------------------- %
function [ Pi ] = UpdateTransitionMatrices( do_sample , H , N , fb_stat_set , I , Y , R , Pi , Phi , PhiR , K , M )
%UpdateTransitionMatrices Samples transition matrices from a state sequence
% S, an input sequence I, a Dirichlet prior H and the param_set.burnin_counter of possible
% input symbols M.

K = size(H,2);
if isempty( N )
    N = compute_transition_counts( fb_stat_set , I , Y , R , Pi , Phi , PhiR , K , M );
end

% sample Pi
Pi = zeros(K,K,M);
for m=1:M
    for k=1:K
        if do_sample
            Pi(k, :, m) = sample_dirichlet( N(k,:,m) + H );
        else
            sumN = sum( N(k,:,m) );
            if sumN > 0                
                Pi(k, :, m) = N(k,:,m) / sumN;
            else
                Pi(k,:,m ) = H / sum( H );
            end
        end
    end
end
        
% ----------------------------------------------------------------------- %
function [ Pi0 ] = UpdateStartDistribution( do_sample , S , H )
% samples the start distribution
    
K = size(H,2);
Pi0 = zeros( K , 1 );
N = zeros( K , 1 );

% get counts
for i = 1:numel( S )
    N( : , 1 ) = S{ i }( : , 1 ) + N( : , 1 );
end

% sample Pi0
if do_sample
    Pi0( : , 1 ) = sample_dirichlet( N + H' );
else
    if sum( N ) > 0
        Pi0( : , 1 ) = N / sum( N );
    else
        Pi0( : , 1 ) = H;
    end
end
Pi0 = Pi0 / sum( Pi0 );

% ----------------------------------------------------------------------- %
function N = compute_transition_counts( fb_stat_set , I , Y , R , Pi , Phi , PhiR , K , M )
N = zeros( K , K , M );
use_reward_model = ~isempty( PhiR );
PhiT = permute( Phi , [ 2 1 3 ] );
if use_reward_model
    PhiRT = permute( PhiR , [ 2 1 3 ] );
end
for episode_index = 1:numel( fb_stat_set )
    
    % get the variables for the episode
    my_I = I{ episode_index };
    if use_reward_model
        my_R = R{ episode_index };
    end
    if ~isempty( Y )
        my_Y = Y{ episode_index };
    end
    my_fb = fb_stat_set{ episode_index };
    T = length( my_I );
    
    % counting depends on whether we sampled or not
    if isfield( my_fb , 'forward_pass' )
        backward_pass = my_fb.backward_pass';
        for t = 1:T
            
            % get the forward and backward variables
            a = my_fb.forward_pass( : , t );
            b = backward_pass( t + 1 , : );
            
            % collect eta_ij (see rabiner)
            if isempty( Pi )
                dist = a * b;
            else
                dist = Pi( : , : , my_I( t ) );
                dist = bsxfun( @times , dist , a );
                dist = bsxfun( @times , dist , b );
                dist = bsxfun( @times , dist , PhiT( my_Y( t ) , : , my_I( t ) ) );
                if use_reward_model && t~= T
                    dist = bsxfun( @times , dist , PhiRT( my_R( t + 1 ) , : , my_I( t + 1 ) ) );
                end
            end
            dist = dist / sum( dist(:) );
            
            % add to N
            N( : , : , my_I( t ) ) = dist + N( : , : , my_I( t ) );
        end
        
    % if no forward/backward variables, use the sequence itself
    else
        my_fb_t = my_fb';
        for t = 1:T
            dist = my_fb( : , t ) * my_fb_t( t + 1 , : );
            dist = dist / sum( dist(:) );
            N( : , : , my_I( t ) ) = dist + N( : , : , my_I( t ) );
        end
    end
end

    
