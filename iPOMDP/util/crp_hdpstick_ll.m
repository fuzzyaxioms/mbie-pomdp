function ll = crp_hdpstick_ll( cnts, alpha, beta_stick )

  if length( cnts ) > length( beta_stick )
    cnts = cnts( 1:length( beta_stick ) );  
  else
      beta_stick = beta_stick( 1:length( cnts ) );
  end
  nzinds = ( cnts > 0 ) & ( beta_stick > 0 );
  cnts = cnts( nzinds );
  beta_stick = beta_stick( nzinds );

  if ( length(beta_stick) == 0 )
    ll = 0;

  else
    ll = gammaln( sum( alpha*beta_stick ) ) - ...
         sum( gammaln( alpha*beta_stick ) ) + ...
         sum( gammaln( alpha*beta_stick + cnts ) ) - ...
         gammaln( sum( alpha*beta_stick + cnts ) );
  end;

  if ( ~isfinite( ll ) )
    keyboard;
  end;

return;
