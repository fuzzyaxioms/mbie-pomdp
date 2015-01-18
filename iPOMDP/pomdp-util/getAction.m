function [a val maxInd] = getAction(bel, Vtable);
%   [a val maxInd] = getAction(bel, Vtable);
%   Vtable{ iter }.alphaList( :,i ) = unique alpha vector
%   Vtable{ iter }.alphaAction(i) = action for alpha i
%   bel is a ROW VECTOR or set of column vectors

% compute vals for given belief, return action with best score
vals = bel * Vtable{end}.alphaList;
[val maxInd] = max(vals,[],2);
a = Vtable{end}.alphaAction(maxInd);