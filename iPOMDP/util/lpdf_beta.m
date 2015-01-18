function ll = lpdf_beta( p , a , b )
% function ll = lpdf_beta( p , a , b )
% p is a vector of probabilities

% compute the thingie
ll = gammaln( a + b ) - gammaln( a ) - gammaln( b ) + ...
    ( a - 1 ) * log( p ) + ( b - 1 ) * log( 1 - p );
