function h = discrete_entropy( p )
% function h = discrete_entropy( p )
% computes the entropy of p, where p is a discrete distribution (all values
% positive, sum to one), taking out places where p is zero
p( p == 0 ) = [];
h = -1 * sum( p .* log( p ) );