function x = randr(m, v)
% RANDRG Random numbers from rectified Gaussian density
%   p(x) = 1/C * exp(-1/2*(x-m)^2/v) * h(x)
%
% Usage
%   x = randrg(m,v) returns an array of random numbers chosen from the
%    rectified Gaussian density with parameters m and v.  The
%    size of x is the common size of arrays m and v.
%
% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk

% Allocate output array
x = zeros(size(m));

% Generate uniform random numbers
y = rand(size(m));

% Large negative mean and low variance, approximate by exponential density
j = -m./(sqrt(2*v));
k = j>26;

% Compute inverse cumulative density
x(k) = log(y(k))./(m(k)./v(k));
R = erfc(abs(j(~k)));
x(~k) = erfcinv(y(~k).*R-(j(~k)<0).*(2*y(~k)+R-2)).*sqrt(2*v(~k))+m(~k);

% correct any corner cases
zero_set = ( m == 0 ) & ( v == 0 );
x( zero_set ) = 0;

% Mikkel's deprecated randr
% % RANDR Random numbers from 
% %   p(x)=K*exp(-(x-m)^2/s-l'x), x>=0 
% %
% % Usage
% %   x = randr(m,s,l)
% 
% % Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
% 
% A = (l.*s-m)./(sqrt(2*s));
% a = A>26;
% x = zeros(size(m));
% 
% y = rand(size(m));
% x(a) = -log(y(a))./((l(a).*s-m(a))./s);
% 
% R = erfc(abs(A(~a)));
% x(~a) = erfcinv(y(~a).*R-(A(~a)<0).*(2*y(~a)+R-2)).*sqrt(2*s)+m(~a)-l(~a).*s;
% 
% x(isnan(x)) = 0;
% x(x<0) = 0;
% x(isinf(x)) = 0;
% x = real(x);
