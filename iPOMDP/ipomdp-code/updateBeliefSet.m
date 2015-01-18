function pomdpSet = updateBeliefSet( pomdpSet , action , obs )
% function pomdpSet = updateBeliefSet( pomdpSet , action , obs )
if iscell( pomdpSet )
    for i = 1:numel( pomdpSet )
        pomdpSet{ i }.bel = updateBelief( pomdpSet{ i }.pomdp , ...
            pomdpSet{ i }.bel, action, obs );
    end
end