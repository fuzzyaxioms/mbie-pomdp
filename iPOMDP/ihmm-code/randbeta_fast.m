function x = randbeta_fast(a,b)
% function x = randbeta_fast(a,b)
% warning! a , b should be scalars!
% warning! is not safe for very small a , b
x = randg(a);
x = x./(x + randg(b));
