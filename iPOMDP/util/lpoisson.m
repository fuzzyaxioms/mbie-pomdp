function ll = lpoisson( k , lambda )
% function ll = lpoisson( k , lambda )
ll = k * log( lambda ) - lambda - gammaln( k + 1 );