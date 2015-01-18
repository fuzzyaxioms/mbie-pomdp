function ls = log_sum( la )
% function ls = log_sum( la )
% computes the sum of a given la in a stable fashion
b = max( la );
ls = log( sum( exp( la - b ) ) ) + b;
