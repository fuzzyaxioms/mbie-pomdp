function [val vec] = getQ(bel, backupStats , a)
%   function val = getQ(bel, backupStats , a);
%   Vtable{ iter }.alphaList( :,i ) = unique alpha vector
%   Vtable{ iter }.alphaAction(i) = action for alpha i
%   bel is a COLUMN VECTOR or set of column vectors
if isfield( backupStats , 'Q' )
    [ val max_ind ] = max( backupStats.Q(:,:,a) * bel ); 
    vec = backupStats.Q( max_ind , : , a );
else
    aInd = find( backupStats.Vtable{end}.alphaAction == a );
    if isempty( aInd )
        disp('getQ.m: could not find any alpha vectors!')
        val = [];
        vec = [];
    else
        Qtable = backupStats.Vtable{end}.alphaList( : , aInd );
        [ val max_ind ] = max( bel * Qtable );
        vec = Qtable( : , max_ind );
    end
end

