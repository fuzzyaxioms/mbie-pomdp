function [ x_mid y_mid y_se ] = block( x , y , window_size )
% function blockplot( x , y , window_size )
% plots by windowing
min_x = min( x );
max_x = max( x );

% go through each window
block_index = 1;
start_x = min_x + ( block_index - 1 ) * window_size;
end_x = start_x + window_size - 1;
while ( ( end_x < max_x ) || ( block_index <= 1 ) )
    y_set = y( ( x >= start_x ) & ( x < end_x ) );
    y_mid( block_index ) = mean( y_set );
    y_se( block_index ) = std( y_set ) / sqrt( length( y_set ) );    
    start_x = min_x + ( block_index - 1 ) * window_size;
    end_x = start_x + window_size - 1;
    x_mid( block_index ) = ( start_x + end_x ) / 2;
    block_index = block_index + 1;
end

