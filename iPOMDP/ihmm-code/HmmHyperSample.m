function [sbeta, salpha0, sgamma, N, M] = HmmHyperSample(N, ibeta, ialpha0, igamma, hypers, numi )
% HMMHYPERSAMPLE resamples the hyperparameters of an infinite hmm.
%
% [sbeta, salpha0, sgamma, M] = ...
%   iHmmHyperSample(N, ibeta, ialpha0, igamma, hypers, numi )
%   resamples the hyperparameters given the transition counts N, the previous
%   hyperparameters ibeta, ialpha0, igamma and their respective
%   hyper-hyperparameters in the structure hypers (needs alpha0_a,
%   alpha0_b, gamma_a and gamma_b fields corresponding to gamma prior on
%   the hyperparameters). If the hyper-hyperparameters are not given,  the
%   estimated alpha0 and gamma will be the same as the input alpha0 and
%   gamma. numi is the number of times we run the Gibbs samplers for alpha0
%   and gamma (see HDP paper or Escobar & West); we recommend a value of
%   around 20. The function returns the new hyperparameters, the CRF counts
%   (N) and the sampled number of tables in every restaurant (M).
%
%   Note that the size of the resampled beta will be the same as the size
%   of the original beta.

% set number of states
K = length(ibeta);

% Compute N: state transition counts -- collapse all together, irrespective
% of the action (since the beta stick is the same for all actions)
N = sum( N , 3 );

% Compute M: number of similar dishes in each restaurant.
M = zeros(K);
for j=1:K
    for k=1:K
        if N(j,k) == 0
            M(j,k) = 0;
        else
            for l=1:N(j,k)
                M(j,k) = M(j,k) + (rand() < (ialpha0 * ibeta(k)) / (ialpha0 * ibeta(k) + l - 1));
            end
        end
    end
end

% Resample beta
ibeta = sample_dirichlet( sum(M,1) + igamma/K );

% Resample alpha
if isfield(hypers, 'alpha0')
    ialpha0 = hypers.alpha0;
else
    for iter = 1:numi
        w = randbeta_fast(ialpha0+1, sum(N,2));
        p = sum(N,2)/ialpha0; p = p ./ (p+1);
        s = binornd(1, p);
        ialpha0 = randg(hypers.alpha0_a + sum(sum(M)) - sum(s)) / (hypers.alpha0_b - sum(log(w)));
    end
end

% Resample gamma
if isfield(hypers, 'gamma')
    igamma = hypers.gamma;
else
    
    % for finite models, estimating the precision parameter is ugly: in our
    % case, the likelihood is G(gamma)*exp( gamma/K * sum_k log beta_k ) / G(
    % gamma/K )^K, where G is the gamma function.  for now use simple MH:
    ll = gammaln( igamma ) - K * gammaln( igamma/K ) + igamma/K * sum( log( ibeta ) );
    for iter = 1:numi
        gamma_new = randg( hypers.gamma_a / hypers.gamma_b );
        ll_new = gammaln( gamma_new ) - K * gammaln( gamma_new/K ) + gamma_new/K * sum( log( ibeta ) );
        if ll_new > ll
            ll = ll_new;
            igamma = gamma_new;
            % disp('mh accept in ihmmhypersample.m')
        end
    end
end

sbeta = ibeta;
salpha0 = ialpha0;
sgamma = igamma;

