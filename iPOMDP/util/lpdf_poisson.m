function ll = lpdf_poisson( x_set , lambda_set )
% function lpdf = lpdf_poisson( x_set , lambda_set )
%   - Outputs the log pdf of the poisson distribution for the each x,l pair
%     (must be of the same size!)
%   - NO checking for negative lambda, whole number x, or correcting large x
ll = x_set .* log( lambda_set ) - lambda_set - gammaln( x_set + 1 );
ll( ( lambda_set == 0 ) & ( x_set == 0 ) ) = 1; 
ll( ( lambda_set == 0 ) & ( x_set > 0 ) ) = 0; 