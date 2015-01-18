function pomdpSet = resetBeliefSet( pomdpSet )
% function pomdpSet = resetBeliefSet( pomdpSet )
if iscell( pomdpSet )
    for i = 1:numel( pomdpSet );
        pomdpSet{ i }.bel = pomdpSet{ i }.pomdp.start';
    end
end