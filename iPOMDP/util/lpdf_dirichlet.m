function ll = lpdf_dirichlet( p , a )

% remove spots where a == 0
p = p( a > 0 );
a = a( a > 0 );

% compute the thingie
ll = gammaln( sum( a ) ) - sum( gammaln( a ) ) + ...
    sum( ( a - 1 ) .* log( p ) );


