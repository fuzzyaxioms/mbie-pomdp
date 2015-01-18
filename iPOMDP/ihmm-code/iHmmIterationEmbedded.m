function [sample, deleted] = iHmmIterationEmbedded(Y, I, sample, hypers, M, inference)
% IHMMITERATIONEMBEDDED Performs one iteration of the embedded HMM sampler.
%
% [S, deleted] = iHmmIterationEmbedded(Y, sample) performs one iteration of
% the embedded HMM sampler.
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
%   - M: the number of possible input symbols,
%   - inference: a structure specifying the settings for the inference
%                scheme.
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

% Start the embedded HMM sampler.
for d=1:D    
    T = length(Y{d});
    C = zeros(T, inference.Q);
    
    % Create the pool of variables.
    for t=1:T
        C(t,1) = sample.S{d}(t);
        for q=2:inference.Q
            C(t,q) = 1 + sum( rand() > cumsum( sample.Beta ) );
            if C(t,q) == sample.K+1
                sample = ExpandTransitionMatrix( sample );
                sample.Phi(sample.K,:,:) = sample_dirichlet(hypers.H, M)';
            end
        end
    end
    
    % Run the dynamic program.
    dyn_prog = zeros( inference.Q, length(Y{d}) );
    dyn_prog(:,1) = sample.Pi(1, C(1,:), I{d}(1))' .* sample.Phi(C(1,:), Y{d}(t), I{d}(t));
    dyn_prog(:,1) = dyn_prog(:,1) ./ sum(dyn_prog(:,1));
    for t=2:T
        dyn_prog(:,t) = sample.Pi(C(t-1,:), C(t,:), I{d}(t))' ...
                            * dyn_prog(:,t-1) ...
                            .* sample.Phi(C(t,:), Y{d}(t), I{d}(t)) ...
                            ./ sample.Beta(C(t,:))';
        dyn_prog(:,t) = dyn_prog(:,t) ./ sum(dyn_prog(:,t));
    end
    
    % Resample the state sequence.
    sample.S{d}(T) = C(T, 1 + sum( rand() > cumsum( dyn_prog(:,T) ) ));
    for t=T-1:-1:1
        r = dyn_prog(:,t) .* sample.Pi(C(t,:), sample.S{d}(t+1), I{d}(t+1));
        r = r ./ sum(r);
        sample.S{d}(t) = C(t, 1 + sum(rand() > cumsum(r)));
    end
    
    % TODO Safety check remove
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
