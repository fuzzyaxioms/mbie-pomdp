function ll = collapsed_lmultinomial( count_set , alpha_set )
% function ll = collapsed_lmultinomial( count_set , alpha_set )
% integrates out the dirichlet distribution
ll = sum( gammaln( alpha_set + count_set ) ) - gammaln( sum( alpha_set + count_set ) ) ...
    + gammaln( sum( alpha_set ) ) - sum( gammaln( alpha_set ) );