function probs = ll_to_prob( log_probs )
% Convert a vector of log-space probabilities to the renormalized
% version.
  probs = log_probs - max(log_probs);
  probs = exp( probs );
  probs = probs ./ sum( probs );
  
  if ( any( ~isfinite(probs) ) )
    fprintf('probability explosion!\n');
    keyboard;
  end;

return;
