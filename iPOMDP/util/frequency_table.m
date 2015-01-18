function [f v j] = freqTable( x )
% function [f v j] = freqTable( x )
% input: x is a vector of values
% output: 
%    f is the frequency of each value
%    v is the value 
%    v( j ) = x
[ v i j ] = unique( x );
f = zeros( size( v ) );
for ind = 1:length( x )
    f( j( ind ) ) = f( j( ind ) ) + 1;
end
