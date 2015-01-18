function CheckSample( sample )
%CHECKSAMPLE Performs some checks to make sure that a sample is consistent.

% TODO get rid of cache!

D = length(sample.S);

% Make sure the number of parameters equals the number of states.
assert(size(sample.Phi,1) == sample.K);

% Make sure the number of rows in the transition matrix equals the number
% of states.
assert(size(sample.Pi,1) == sample.K);

% Make sure the number of columns in the transition matrix equals the
% number of states + the cache of the remaining weight.
assert(size(sample.Pi,2) == sample.K+1);

% Make sure the number of states equals the number of represented sticks in
% the base distribution + the cache of the remaining weight.
assert(sample.K+1 == length(sample.Beta));

% Make sure all transition probabilities are in [0,1].
assert(min(min(sample.Pi)) >= 0.0);
assert(max(max(sample.Pi)) <= 1.0);

% Make sure all states that we store are actually used.
maxS = max(cellfun(@max, sample.S));
assert(sample.K == maxS);