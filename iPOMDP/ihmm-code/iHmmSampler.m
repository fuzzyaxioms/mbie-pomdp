function [S, stats] = iHmmSampler(obs, hypers, numb, nums, numi, inference)
% IHMMSAMPLER Samples states from the iHMM with multinomial output.
%
% [S, stats] = iHmmSampleBeam(Y, hypers, numb, nums, numi, inference) uses
% the beam sampling training algorithm for the infinite HMM.
%
%   Input Parameters:
%   - obs: can either be a cell array of training sequences of arbitrary
%          length; or a structure with nodes
%           - Y: cell array with training sequences of arbitrary length,
%           - I: cell array with input sequences (same lengths as Y's),
%           - M: the number of possible input symbols,
%   - hypers: a structure that describes the hyperparameters for the beam
%             sampler. If this structure contains alpha0 and gamma, it will
%             not resample these during sampling. If these are not
%             specified, one needs to specify hyperparameters for alpha0
%             and gamma (alpha0_a, alpha0_b, gamma_a, gamma_b). hypers
%             should also contain a prior for the emission alphabet in the
%             field H,
%   - numb: the number of burnin iterations,
%   - nums: the number of samples to output,
%   - numi: the number of sampling, iterations between two samples,
%   - inference (optional): is a structure that allows the user to
%                           customize the inference algorithm:
%       - S0 : is a cell array of initial assignment to the sequence;
%              a random sequence with values between 0 and 10 will be
%              used if this parameter is not given,
%       - K0 : the number of states to use when initializing the sampler,
%       - name : which algorithm to use for sampling; choices are
%                   - 'beam': uses the beam sampler
%                   - 'cgibbs': uses the collapsed Gibbs sampler
%                   - 'ehmm': uses the embedded HMM sampler (default Q = 10)
%       - Q : the number of states to consider for the embedded HMM sampler,
%       - debug : print debugging messages (off by default).
%
%
%   Output Parameters:
%   - S: is a cell array of sample structures where each sample contains the
%        hidden state sequence S, the number of states K, the Beta, Pi,
%        Phi's used for that sample.
%   - stats: is a structure that contains a variety of statistics for every
%            iteration of the sampler: K, alpha0, gamma, the size of the
%            trellis and the marginal likelihood.

if nargin < 6
    inference.K0 = 10;
    inference.name = 'ehmm';
    inference.Q = 10;
    inference.debug = 1;
else
    if ~isfield(inference, 'debug')
        inference.debug = 0;
    end
    if ~isfield(inference, 'name')
        inference.name = 'ehmm';
        inference.Q = 10;
    end
    if strcmp(inference.name, 'ehmm') && ~isfield(inference, 'Q')
        inference.Q = 10;
    end
end

% Some useful information ...
if isfield(obs, 'Y')
    Y = obs.Y;
    D = length(Y);                      % The number of sequences to train on.
    I = obs.I;
    M = obs.M;                              % Is the number of input symbols considered.
    if size(Y,1) ~= size(I,1)
        error('As many input sequences as observation sequences are expected.');
    end
else
    Y = obs;
    D = length(Y);                      % The number of sequences to train on.
    I{1} = ones(1,length(Y{1}));
    M = 1;
end

% Initialize the sampler.
sample.S = cell(D,1);
for d=1:D
    if isfield(inference, 'S0')
        if length(inference.S0{d}) ~= length(Y{d})
            error('Each sample initialization sequence must be as long as each output sequence.');
        end
        sample.S{d} = inference.S0{d};
    elseif isfield(inference, 'K0')
        sample.S{d} = ceil(rand(1,length(Y{d})) * inference.K0);
    else
        sample.S{d} = ceil(rand(1,length(Y{d})) * 10);
    end
end
sample.K = max(cellfun(@max, sample.S));

% Setup structures to store the output.
S = {};
stats.K = zeros(1,(numb + (nums-1)*numi));
stats.alpha0 = zeros(1,(numb + (nums-1)*numi));
stats.gamma = zeros(1,(numb + (nums-1)*numi));
stats.jll = zeros(1,(numb + (nums-1)*numi));

% Initialize hypers; resample a few times as our inital guess might be off.
if isfield(hypers, 'alpha0')
    sample.alpha0 = hypers.alpha0;
else
    sample.alpha0 = randgamma(hypers.alpha0_a) / hypers.alpha0_b;
end
if isfield(hypers, 'gamma')
    sample.gamma = hypers.gamma;
else
    sample.gamma = randgamma(hypers.gamma_a) / hypers.gamma_b;
end
for i=1:5
    sample.Beta = ones(1, sample.K+1) / (sample.K+1);
    [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);
end

% Sample the emission and transition probabilities.
sample.Phi = SampleEmissionMatrix( sample.S, Y, I, hypers.H, sample.K, M );
sample.Pi = SampleTransitionMatrix( sample.S, I, sample.alpha0 * sample.Beta, sample.K, M );

iter = 1;
if inference.debug
    disp(sprintf('Iteration 0: K = %d, alpha0 = %f, gamma = %f.', sample.K, sample.alpha0, sample.gamma));
end

while iter <= (numb + (nums-1)*numi)
    
    CheckSample(sample);
    
    tic

    % Run some sampler for one iteration.
    if strcmp(inference.name, 'beam')
        sample = iHmmIterationBeam(Y, I, sample, hypers, M);
    elseif strcmp(inference.name, 'cgibbs')
        sample = iHmmIterationGibbs(Y, I, sample, hypers, M);
    elseif strcmp(inference.name, 'ehmm')
        sample = iHmmIterationEmbedded(Y, I, sample, hypers, M, inference);
    else
        error('Unknown inference scheme (e.g. beam, cgibbs, ehmm: %s', inference.name);
    end
        
    % Save some stats.
    stats.alpha0(iter) = sample.alpha0;
    stats.gamma(iter) = sample.gamma;
    stats.K(iter) = sample.K;
    stats.jll(iter) = 0.0;%iHmmJointLogLikelihood(sample.S, Y, sample.Beta, sample.alpha0, hypers.H);
    stats.time(iter) = toc;
    
    if inference.debug
        disp(sprintf('Iteration: %d: K = %d, alpha0 = %f, gamma = %f, JL = %f.', ...
            iter, sample.K, sample.alpha0,sample. gamma, stats.jll(iter)));
    end

    % Store a sample if needed.
    if iter >= numb && mod(iter-numb, numi) == 0
        S{end+1} = sample;
    end

    % Prepare next iteration.
    iter = iter + 1;
end