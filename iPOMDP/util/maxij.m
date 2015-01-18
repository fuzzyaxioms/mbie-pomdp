function [ m i j ] = maxij( A );
% [ m i j ] = maxij( A );
% gives row and column of max

[ mvec rvec ] = max( A ); 
[ m j ] = max( mvec );
i = rvec( j );