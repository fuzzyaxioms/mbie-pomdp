function ll_set = lpdf_unigaussian( value , mean_value , sigma )
% function ll_set = lpdf_unigaussian( value , mean_value , sigma )
ll_set = -1/2 * log( 2 * pi * sigma^2 ) - 1/2 * ( 1/sigma^2 ) * ( value(:) - mean_value(:) ).^2;
