function [sample, deleted] = iHmmIterationBeam(Y, I, sample, hypers, M)
% IHMMITERATIONBEAM Performs one iteration of the beam sampler.
%
% [S, deleted] = iHmmIterationBeam(Y, sample) performs one iteration of
% the beam sampling training algorithm for the infinite IO-HMM.
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

CheckSample(sample);

% Initialize return variables.
oldK = sample.K;                    % store the number of incomming states.
deleted = [];                       % a list of incomming states that got deleted.

% Sample the auxilary variables.
u = cell(D,1);
for d=1:D
    % Sample auxilary variables for the states which we sampled previously.
    for t=1:length(sample.S{d})
        if t == 1
            u{d}(t) = rand() * sample.Pi(1, sample.S{d}(t), I{d}(t));
        else
            u{d}(t) = rand() * sample.Pi(sample.S{d}(t-1), sample.S{d}(t), I{d}(t));
        end
    end
    % For new states, just set the slice to something small, so relatively
    % many new states are considered.
    for t=length(sample.S{d}):length(Y{d})
        u{d}(t) = min(u{d});
    end
end

% Extend Pi and Phi if necessary.
minu = min(cellfun(@min, u));
while max(max(sample.Pi(:, end, :))) > minu     % Break the Pi sticks some more.
    sample = ExpandTransitionMatrix(sample);
    sample.Phi(sample.K,:,:) = sample_dirichlet(hypers.H, M)';
end
sample.K = size(sample.Pi, 1);

% Make sure the last column of Pi and Beta make sure they sum to one.
for m=1:M
    sample.Pi(:,end,m) = 1.0 - sum(sample.Pi(:,1:end-1,m),2);
end
sample.Beta(end) = 1.0 - sum(sample.Beta(1:end-1));

% Resample the hidden state sequence.
for d=1:D
    dyn_prog = zeros( sample.K, length(Y{d}) );
    for t=1:length(Y{d})
        for k=1:sample.K
            dyn_prog(k,t) = 0;
            if t == 1
                if sample.Pi(1,k,I{d}(t)) > u{d}(t)
                    dyn_prog(k,t) = dyn_prog(k,t) + 1;
                end
            else
                for l=1:sample.K
                    if sample.Pi(l,k,I{d}(t)) > u{d}(t)
                        dyn_prog(k,t) = dyn_prog(k,t) + dyn_prog(l,t-1);
                    end
                end
            end
            dyn_prog(k,t) = sample.Phi(k, Y{d}(t), I{d}(t)) * dyn_prog(k, t);
        end

        dyn_prog(:,t) = dyn_prog(:,t) / sum(dyn_prog(:,t));
    end

    % Backtrack to sample a path through the HMM.
    if sum( dyn_prog(:,end) ) ~= 0.0 && isfinite( sum( dyn_prog(:,end) ) )
        sample.S{d}(length(Y{d})) = 1 + sum( rand() > cumsum( dyn_prog(:,end) ) );
        for t=length(Y{d})-1:-1:1
            r = dyn_prog(:,t) .* sample.Pi(:, sample.S{d}(t+1), I{d}(t+1));
            r = r ./ sum(r);
            sample.S{d}(t) = 1 + sum(rand() > cumsum(r));
        end

        % TODO Safety check remove
        assert(~isnan(sum(sample.S{d}(t))));
    else
        warning('BEAM:NoPathThroughTrellis', 'Wasted computation as there were no paths through the iHMM.');
    end
end

% Safety check.
for d=1:D
    assert(~isnan(sum(sample.S{d})));
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
    sample.Pi(:,zind(i),:) = [];
    sample.Pi(zind(i),:,:) = [];
    sample.Phi(zind(i),:,:) = [];
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
