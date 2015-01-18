function ll = logsum( p , lq )
% function ll = logsum( p , lq )
% safely computes the quantity
%                        
%                    log( sum( p .* exp( lq ) )
%
% by first subtracting max( lq ) and then adding back the constant 

max_lq = max( lq );
ll = max_lq + log( sum( p .* exp( lq - max_lq ) ) );