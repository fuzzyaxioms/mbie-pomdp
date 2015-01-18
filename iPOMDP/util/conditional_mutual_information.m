function cmi = ConditionalMutualInformation( data )
% data has three columns of categorical variables; the first variable
% is the conditioning variable

% find out how many elements are in conditioning set
xSet = unique( data( : , 1 ) );

% go through the data and count/compute mi's
data = sortrows( data );
nx = zeros( 1 , length( xSet ) );
mi = zeros( 1 , length( xSet ) );
xStart = 0;
for x = 1:length( xSet )
    
    % update the marginal on x
    xEnd = find( data( : , 1 ) == xSet( x ) , 1 , 'last' );
    if size( xEnd , 1 ) == 0
        xEnd = xStart;
    end
    nx( x ) = xEnd - xStart;
    
    
    % compute conditional mutual information
    mi( x ) = MutualInformation( data( (xStart+1):xEnd , 2:3 ) );
    
    % for the next iteration
    xStart = xEnd;
    
end

% convert counts into probabilities
px = nx / sum( nx );

% compute conditional mutual information
cmi = sum( px .* mi );