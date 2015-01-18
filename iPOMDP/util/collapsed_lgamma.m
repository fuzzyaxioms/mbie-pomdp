function ll = collapsed_lgamma( stat_set , base_set )
% function ll = collapsed_lgamma( stat_set , base_set )
% the stat set is [ sum x sum log x n ] 
% the base set is [ a0 b0 a ]
a0 = base_set( 1 ); 
b0 = base_set( 2 );
a = base_set( 3 );
sum_x = stat_set( : , 1 );
sum_log_x = stat_set( : , 2 );
n = stat_set( : , 3 );

% do the computation
ll = a0 * log( b0 ) + ( a - 1 ) * sum_log_x + gammaln( a0 + n * a ) ...
    - gammaln( a0 ) - n * gammaln( a ) - ( a0 + n * a ) .* log( sum_x + b0 );