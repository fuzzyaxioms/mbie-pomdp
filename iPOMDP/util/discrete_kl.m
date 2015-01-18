
function kl = discrete_kl( p , q )
% function kl = discrete_kl( p , q )
%    computes the kl( p || q )
q( p == 0 ) = [];
p( p == 0 ) = [];
kl = sum( p .* log( p ./ q ) );
