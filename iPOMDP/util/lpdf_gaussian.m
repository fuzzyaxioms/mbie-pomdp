function ll = lpdf_gaussian( value , mean_vec , cov_mat )
% function ll = lpdf_gaussian( value , mean_vec , cov_mat )
% gives the log pdf of a multivariate gaussian; value and mean_vec are both
% converted to column vectors
value = value(:);
mean_vec = mean_vec(:);

% compute
try
    logdet_cov = logdet( cov_mat , 'chol' );
catch me
    disp( 'had to use general logdet!' );
    logdet_cov = real( logdet( cov_mat ) );
end
ll = -1/2 * numel( value ) * log( 2 * pi ) + ...
    -1/2 * logdet_cov + ...
    -1/2 * ( value - mean_vec )' * ( cov_mat \ ( value - mean_vec ) );

