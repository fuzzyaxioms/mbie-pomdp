function value = query_simple_hash( h , key )
% function value = query_simple_hash( h , key )
% returns false if nothing present

% get the index
[ hash_size hash_depth ] = size( h  );
key_ind = mod( key , hash_size ) + 1;

% check if has is occupied
if h( key_ind , 1 ) == key
    value = h( key_ind , 2 );
else
    value = [];
    col = 3;
    while col < hash_depth && h( key_ind , col ) ~= key 
        col = col + 2;
    end
    if col < hash_depth
        value = h( key_ind , col + 1 );
    end
end