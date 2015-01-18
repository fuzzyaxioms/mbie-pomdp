function [ c unique_value ] = cell_unique( c , exclude_count )
% function c = cell_unique( c , exclude_count )
% converts all of the elements of c to unique indices except for
% 1:exclude_count, exclude_count = 0 if not specified.  ASSUMES THAT ALL
% CELLS HAVE THE SAME NUMBER OF ROWS.
if nargin == 1
    exclude_count = 0;
end

% determine whether there are factors not to exclude
factor_count = size( c{ 1 } , 1 ) - exclude_count;
if factor_count == 0
    unique_value = [];
    return
end

% flatten and unique everything but 1:exclude count
flat_c = cell2mat( c );  
flat_c = flat_c( ( exclude_count + 1 ):end , : );
[ unique_value unique_id unique_set ] = unique( flat_c( : ) );
unique_flat_c = reshape( unique_set , factor_count , [] );

% copy back
start_index = 1;
for trial_index = 1:numel( c )
    end_index = start_index + size( c{ trial_index } , 2 ) - 1;
    c{ trial_index }( ( exclude_count + 1 ):end , : ) = ...
        unique_flat_c( : , start_index:end_index );
    start_index = end_index + 1;
end