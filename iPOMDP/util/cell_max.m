function m = cell_max( a )
% function m = cell_max( a )
% finds the largest element in a cell array for any row greater than offset
m = nanmax( cellfun( @array_max , a ) );
if isnan( m )
    m = [];
end

% ----------------------------------------------------------------------- %
function m = array_max( a )
if isempty( a )
    m = NaN;
else
    m = max( a(:) );
end