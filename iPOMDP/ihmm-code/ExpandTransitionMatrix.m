function [ sample ] = ExpandTransitionMatrix( sample )
%EXPANDTRANSITIONMATRIX Expands the representation of the transition matrix
% with one new state. The code handles both 2 or 3 dimensional matrices. It
% assumes that the last column is the sum of all mass in the infinite
% transition matrix.

oldK = size(sample.Pi, 2) - 1;
assert(length(sample.Beta) - 1 == oldK);
newK = oldK + 1;

% Break beta stick.
be = sample.Beta(newK);
bg = randbeta(1, sample.gamma);
sample.Beta(newK) = bg * be;
sample.Beta(newK+1) = (1-bg) * be;

% Add a row to transition matrix.
for m=1:size(sample.Pi, 3)
    sample.Pi(:,newK+1,m) = 0.0;
    sample.Pi(newK,:,m) = sample_dirichlet(sample.alpha0 * sample.Beta);
end

% Add a column to the transition matrix.
for m=1:size(sample.Pi, 3)
    pe = sample.Pi(1:oldK, newK, m);
    a = sample.alpha0 * repmat( sample.Beta(newK), oldK, 1 );
    b = sample.alpha0 * (1 - sum(sample.Beta(1:newK)));
    if min(a) < 1e-2 || min(b) < 1e-2       % This is an approximation when a or b are really small.
         pg = binornd(1, a./(a+b));
    else
        pg = randbeta( a, b );
    end
    sample.Pi(1:oldK, newK, m) = pg .* pe;
    sample.Pi(1:oldK, newK+1, m) = (1-pg) .* pe;
end

sample.K = sample.K + 1;
