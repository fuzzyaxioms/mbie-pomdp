function [ ll mu_mu ] = collapsed_lgaussian( stat_set , base_set )
% function [ ll mu_mu ] = collapsed_lgaussian( stat_set , base_set )
% the stat set is [ sum x sum x^2 n ] per row
% the base set is [ mu0 sigma0 sigma ] overall (one row)
mu0 = base_set( 1 ); 
sigma0_sq = base_set( 2 )^2;
sigma_sq = base_set( 3 )^2;
sum_x = stat_set( : , 1 );
sum_x_sq = stat_set( : , 2 );
n = stat_set( : , 3 );
sigma_mu_sq = 1 ./ ( 1/sigma0_sq + n/sigma_sq );
mu_mu = ( mu0 / sigma0_sq + sum_x / sigma_sq ) .* sigma_mu_sq;

% do the computation
ll = -1/2 * log( 2 * pi * sigma0_sq ) ...
    - n/2 * log( 2 * pi * sigma_sq ) ...
    + 1/2 * log( 2 * pi * sigma_mu_sq ) ...
    - 1/2 * sum_x_sq / sigma_sq ...
    - 1/2 * mu0^2 / sigma0_sq ...
    + 1/2 * mu_mu.^2 ./ sigma_mu_sq;