function [sample, deleted] = iHmmIterationGibbs(Y, I, sample, hypers, M)
% IHMMITERATIONGIBBS Performs one iteration of the Gibbs sampler.
%
% [S, deleted] = iHmmIterationGibbs(Y, sample) performs one iteration of
% the Gibbs sampling training algorithm for the infinite IO-HMM.
%
%   Input Parameters:
%   - Y: training sequence of observations of arbitrary length,
%   - I: training sequence of actions of arbitrary length,
%   - sample: a sample structure with the following fields:
%       - S: an assignment of the hidden state sequence,
%       - K: the number of states that are used,
%       - Pi: a transition matrix,
%       - Phi: an emission matrix,
%       - Beta: the mother stick of the iHMM,
%   - hypers: a structure with hyperparameters (see iIOHmmSampleBeam for
%             details),
%   - M: the number of possible input symbols.
%
%   Output Parameters:
%   - sample: is a cell array of sample structures where each sample contains the
%        hidden state sequence S, the number of states K, the Beta, Pi,
%        Phi's used for that sample.
%   - deleted: the indices of the incomming states which were deleted.
% -------------------

% Some useful information ...
D = length(Y);                  % The number of sequences to train on.
L = length(hypers.H);

% Make state sequence longer if Y became longer.
% TODO might not be consistent with observations?
for d=1:D
    Delta = length(Y{d}) - length(sample.S{d});
    if Delta > 0
        [Send, Y] = iIOHmmPredict( ones(D,1), sample, hypers.H );
        sample.S{d}(end+1:end+Delta) = Send;
    end
end

CheckSample(sample);

% Initialize return variables.
oldK = sample.K;                    % store the number of incomming states.
deleted = [];                       % a list of incomming states that got deleted.

% Compute the empirical emission matrix:
%   E(i,j,l) = number of emmisions of symbol j from state i with input l.
E = zeros(sample.K, L, M);
for d=1:D
    for t=1:length(Y{d})
        E(sample.S{d}(t), Y{d}(t), I{d}(t)) = E(sample.S{d}(t), Y{d}(t), I{d}(t)) + 1;
    end
end

% Compute the empirical transition matrix:
%   N(i,j,l) = number of transition from state i to j with input l.
N = zeros(sample.K, sample.K, M);
for d=1:D
    N(1, sample.S{d}(1), I{d}(t)) = N(1, sample.S{d}(1), I{d}(t)) + 1;
    for t=2:length(Y{d})
        N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) = N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) + 1;
    end
end

for d=1:D
    for t=1:length(Y{d})
        % Discount the transition and emission counts for timestep t.
        E(sample.S{d}(t), Y{d}(t), I{d}(t)) = E(sample.S{d}(t), Y{d}(t), I{d}(t)) - 1;
        if t ~= 1
            N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) = N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) - 1;
        else
            N(1, sample.S{d}(t), I{d}(t)) = N(1, sample.S{d}(t), I{d}(t)) - 1;
        end
        if t ~= length(Y{d})
            N(sample.S{d}(t), sample.S{d}(t+1), I{d}(t)) = N(sample.S{d}(t), sample.S{d}(t+1), I{d}(t)) - 1;
        end

        % Compute the marginal probability for timestep t.
        r = ones(1, sample.K+1);
        for k=1:sample.K
            if t ~= 1
                r(k) = r(k) * ( N(sample.S{d}(t-1), k, I{d}(t)) + sample.alpha0 * sample.Beta(k) );
            else
                r(k) = r(k) * ( N(1, k, I{d}(t)) + sample.alpha0 * sample.Beta(k) );
            end

            if t ~= length(Y{d})
                if t > 1 && k ~= sample.S{d}(t-1)
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + sample.alpha0 );
                elseif t == 1 && k ~= 1
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + sample.alpha0 );
                elseif t > 1 && k == sample.S{d}(t-1) && k ~= sample.S{d}(t+1)
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + 1 + sample.alpha0 );
                elseif t > 1 && k == sample.S{d}(t-1) && k == sample.S{d}(t+1)
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + 1 + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + 1 + sample.alpha0 );
                elseif t == 1 && k == 1 && k ~= sample.S{d}(t+1)
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + 1 + sample.alpha0 );
                elseif t == 1 && k == 1 && k == sample.S{d}(t+1)
                    r(k) = r(k) * ( N(k, sample.S{d}(t+1), I{d}(t+1)) + 1 + sample.alpha0 * sample.Beta(sample.S{d}(t+1)) ) / ( sum(N(k, :, I{d}(t+1))) + 1 + sample.alpha0 );
                end
            end

            r(k) = r(k) * ( hypers.H(Y{d}(t)) + E(k, Y{d}(t), I{d}(t)) ) / ( sum(E(k,:, I{d}(t))) + sum(hypers.H) );
        end
        r(sample.K+1) = ( hypers.H(Y{d}(t)) / (sum(hypers.H)) ) * sample.alpha0 * sample.Beta(sample.K+1);
        if t ~= length(Y{d})
            r(sample.K+1) = r(sample.K+1) * sample.Beta(sample.S{d}(t+1));
        end

        % Resample s_t.
        r = r ./ sum(r);
        sample.S{d}(t) = 1 + sum(rand() > cumsum(r));

        % Update datastructures if we move to a new state.
        assert(size(N,1) == sample.K);
        assert(size(N,2) == sample.K);
        if sample.S{d}(t) > sample.K
            N(:, sample.S{d}(t),:) = 0;                  % We have a new state: augment data structures
            N(sample.S{d}(t), :,:) = 0;
            E(sample.S{d}(t), :,:) = 0;

            % Extend Beta. Standard stick-breaking construction stuff
            b = randbeta(1, sample.gamma);
            BetaU = sample.Beta(end);
            sample.Beta(end) = b * BetaU;
            sample.Beta(end+1) = (1-b)*BetaU;

            sample.K = sample.K + 1;
        end

        % Update emission and transition counts.
        E(sample.S{d}(t), Y{d}(t), I{d}(t)) = E(sample.S{d}(t), Y{d}(t), I{d}(t)) + 1;
        if t ~= 1
            N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) = N(sample.S{d}(t-1), sample.S{d}(t), I{d}(t)) + 1;
        else
            N(1, sample.S{d}(t), I{d}(t)) = N(1, sample.S{d}(t), I{d}(t)) + 1;
        end
        if t ~= length(Y{d})
            N(sample.S{d}(t), sample.S{d}(t+1), I{d}(t)) = N(sample.S{d}(t), sample.S{d}(t+1), I{d}(t)) + 1;
        end

        % Perform some coherency checks on the datastructures.
        assert(size(N,1) == sample.K);
        assert(size(N,2) == sample.K);
        assert(length(sample.Beta) == sample.K+1);
        assert(sum(sum(sum(N))) == length(Y{d}));
        assert(sum(sum(sum(E))) == length(Y{d}));
    end
end

% Safety check.
for d=1:D
    assert(~isnan(sum(sample.S{d}(t))));
end

% Clean up any unused states.
used = [];
for d=1:D
    used = [used unique(sample.S{d})];
end
zind = sort(setdiff(1:sample.K, unique(used)));
for i = length(zind):-1:1
    sample.Beta(end) = sample.Beta(end) + sample.Beta(zind(i));
    sample.Beta(zind(i)) = [];
    for d=1:D
        sample.S{d}(sample.S{d} > zind(i)) = sample.S{d}(sample.S{d} > zind(i)) - 1;
    end
    if zind(i) <= oldK
        deleted = [deleted zind(i)];
        oldK = oldK - 1;
    end
    sample.K = sample.K - 1;
end

% Resample Beta given the transition probabilities.
[sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);

% Resample the Phi's given the new state sequences.
sample.Phi = SampleEmissionMatrix(sample.S, Y, I, hypers.H, sample.K, M);

% Resample the transition probabilities.
sample.Pi = SampleTransitionMatrix(sample.S, I, sample.alpha0 * sample.Beta, sample.K, M );

CheckSample(sample);