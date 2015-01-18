function h = insert_simple_hash( h , key , value )
% function h = insert_simple_hash( h , key , value )
% uses just a mod function for insertions, chains if space is occupied

% get index
hash_size = size( h , 1 );
key_ind = mod( key , hash_size ) + 1;

% check if has is occupied
if h( key_ind , 1 ) < 0 || h( key_ind , 1 ) == key
    h( key_ind , 1 ) = key;
    h( key_ind , 2 ) = value;
else
    col = 3;
    while col < size( h , 2 ) && h( key_ind , col ) >= 0 && h( key_ind , col ) ~= key 
        col = col + 2;
    end
    h( key_ind , col ) = key;
    h( key_ind , col + 1 ) = value; 
end