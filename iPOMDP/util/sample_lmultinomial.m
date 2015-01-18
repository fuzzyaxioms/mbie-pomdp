function x = sample_lmultinomial( w , n )
% sample a value of x from the ( potentially unnormalized ) w
if nargin == 1
  n = 1;
end
if n > 0
  p = w - max( w ) - 1;
  p = exp( p );
  p = p / sum( p );
  for i = 1:n
      cump = cumsum( p );
      cump( end ) = 1 + eps;      
      x(i) = find( (rand * (1-eps)) < cump , 1 );
  end
else
  x = [];
end