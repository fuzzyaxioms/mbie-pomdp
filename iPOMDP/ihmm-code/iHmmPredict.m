function [ S,Y ] = iHmmPredict( I, sample, H, K0 )
%IHMMPREDICT Samples a sequence from the IO-iHMM represented by sample
% given the input sequence I. H is the base measure from which to sample
% state parameters. This is needed in case the predicted sequences wanders
% off to new states. K0 is the state the sequence should start from.
%
% The method returns the hidden state sequence and observations which were
% sampled.

Pi = sample.Pi;
Phi = sample.Phi;

S = zeros(length(I),1);
Y = zeros(length(I),1);
S(1) = 1 + sum(rand() > cumsum( Pi(K0,:,I(1)) ));
Y(1) = 1 + sum(rand() > cumsum( Phi(K0,:,I(1)) ));
for t=2:length(I)
    S(t) = 1 + sum(rand() > cumsum( Pi(S(t-1),:,I(t)) ));
    Y(t) = 1 + sum(rand() > cumsum( Phi(S(t),:,I(t)) ));
    
    % Rebreak the stick if this is a new state.
    if S(t) == length(Pi)
        sample = ExpandTransitionMatrix(sample);
        for m=1:size(Phi,3)
        	Phi(S(t),:,m) = sample_dirichlet(H);
        end
    end
end
