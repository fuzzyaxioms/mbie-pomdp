function mi = MutualInformation( data )
% data has two columns of categorical variables; we estimate the marginals
% and the joint use these to compute an empirical mutual information

% find out how many elements are in each set
xSet = unique( data( : , 1 ) );
ySet = unique( data( : , 2 ) );

% go through the data and count things
data = sortrows( data );
nx = zeros( 1 , length( xSet ) );
ny = zeros( 1 , length( ySet ) );
nxy = zeros( length( xSet ) , length( ySet ) );
xStart = 0;
for x = 1:length( xSet )
    
    % update the marginal on x
    xEnd = find( data( : , 1 ) == xSet( x ) , 1 , 'last' );
    if size( xEnd , 1 ) == 0
        xEnd = xStart;
    end
    nx( x ) = xEnd - xStart;
        
    % loop through y values
    yStart = 0;
    for y = 1:length( ySet )

        % update the marginal on y
        yEnd = find( data( (xStart+1):xEnd , 2 ) == ySet( y ) , 1 , 'last' );
        if size( yEnd , 1 ) == 0
            yEnd = yStart;
        end
        ny( y ) = ny( y ) + yEnd - yStart;

        % update the joint
        nxy( x , y ) = yEnd - yStart;
        
        % for the next iteration
        yStart = yEnd;
    end
    
    % for the next iteration
    xStart = xEnd;
end

% convert counts into probabilities
px = nx / sum( nx );
py = ny / sum( ny );
pxy = nxy / sum( nxy( : ) );

% compute mutual information
tmat = pxy .* log( pxy ./ ( px' * py ) );
tmat( find( pxy == 0 ) ) = 0;
mi = sum( tmat(:) );