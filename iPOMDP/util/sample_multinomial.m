function [ x l ] = sample_multinomial( w , n , my_rand )
% sample a value of x from the ( potentially unnormalized ) w
if nargin == 1
  n = 1;
  my_rand = rand;
elseif nargin == 2
  my_rand = rand;
end

% do the sampling
if n > 0
  p = w;
  p = p / sum( p );
  for i = 1:n
      cump = cumsum( p );
      cump( end ) = 1 + eps;      
      x(i) = find( ( my_rand * (1-eps)) < cump , 1 );
      l(i) = p(x(i));
  end
else
  x = [];
  l = [];
end