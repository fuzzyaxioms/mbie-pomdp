function colorplot( x , y , use_median , figure_index , color )
% function colorplot( x , y , use_median , figure_index , color )
%  * plots x vs. y, y should be [ n rows of p reps ] format
%  * use_median = true plots median and IQR of y; use_median = false plots
%    mean and standard error of y (note: not standard deviation!)
%  * color can be a 3-vector (RGB) or 4-vector (RGB-alpha)

% make sure that data was inputted
if isempty( y )
    return
end

% default values
if nargin == 2
    use_median = false;
    figure_index = 1;
    color = [1 0 0];
elseif nargin == 3
    figure_index = 1;
    color = [1 0 0];
elseif nargin == 4
    color = [1 0 0];
end

% check length of color vector
if length( color ) == 3
    alpha = 1;
else
    alpha = color( end );
    color = color( 1:3 );
end

% get the right figure
figure( figure_index ); hold on;

% get the data
if use_median
    boty = prctile( y' , 25 );
    topy = prctile( y' , 75 );
    midy = prctile( y' , 50 );
else
    midy = mean( y' );
    stderr = std( y' ) / sqrt( size( y , 2 ) );
    boty = midy - stderr;
    topy = midy + stderr;
end

% make the plot
my_x = [ x x(end:-1:1) ];
my_y = [ boty topy(end:-1:1) ];
fill( my_x , my_y , color , 'edgecolor' , color , 'facealpha' , alpha );
plot( x , midy , 'color' , .5 * color , 'linewidth' , 4 )
