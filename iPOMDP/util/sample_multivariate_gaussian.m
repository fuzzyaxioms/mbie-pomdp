function out = sample_multivariate_gaussian( mean_vector , cov_mat , n );
% function out = sample_multivariate_gaussian( mean_vector , cov_mat , n );
m = numel( mean_vector );

% last argument is how many
if nargin == 2
    n = 1;
end

% do the sampling
x = randn( m , n );
r = chol( cov_mat );
out = bsxfun( @plus , r * x , mean_vector ); 