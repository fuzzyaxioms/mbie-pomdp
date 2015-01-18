function [ x_block y_center y_std ] = block_plot( data , block_size )
% Data should be in the form of r x n, where r is the number of repetitions
% and n is the length of the trial. (NaNs okay if the trials are of
% different lengths.)  Block size is the chunk size. Note: both reps and
% blocks get considered the same in the block averaging, may or may not be
% statistically appropriate!!
%
% returns: x's for the middle of the blocks, y_center is the mean, y_std is
% the standard error (NOT standard dev)

% cut into blocks
x_block = []; y_center = []; y_upper = []; y_lower = []; y_std = [];
block_start = 1;
max_trial_length = size( data , 2 );
while block_start < max_trial_length
    block_end = min( block_start + block_size , max_trial_length );
    x_block( end + 1 ) = ( block_end + block_start / 2 );
    y_block = data( : , block_start:block_end );
    y_block = y_block(:);
    y_center( end + 1 ) = nanmean( y_block );
    y_std( end + 1 ) = nanstd( y_block ) / sqrt( numel( y_block ) );
    block_start = block_end + 1;
end

% plot for 95% confidence interval of the mean
figure( 1 ); clf; hold on;
errorbar( x_block , y_center , 2 * y_std , 'linewidth' , 3 );
