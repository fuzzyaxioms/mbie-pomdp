function fsc = create_image_fsc

fsc.node_count = 12;
fsc.action_count = 10;
fsc.obs_count = 5;
fsc.start = zeros( 1 , fsc.node_count ); 
fsc.start( 1 ) = 1;

fsc.policy = zeros( fsc.node_count , fsc.action_count );
fsc.transition = zeros( fsc.node_count , fsc.node_count , fsc.obs_count );

% first state: apply the filter, transfer down a scale
fsc.policy( 1 , 1 ) = 1;
fsc.transition( 2 , 1 , 1 ) = 1;
fsc.transition( 3 , 1 , 2 ) = 1;
fsc.transition( 4 , 1 , 3 ) = 1;
fsc.transition( 5 , 1 , 4 ) = 1;
fsc.transition( 1 , 1 , 5 ) = 1;

% next states, 2-5: everyone needs to change the scale and transition to
% change filter/apply filter states
fsc.policy( 2 , 5 ) = 1;
fsc.policy( 3 , 6 ) = 1;
fsc.policy( 4 , 7 ) = 1;
fsc.policy( 5 , 8 ) = 1;
fsc.transition( 6 , 2:5 , : ) = 1;

% do another filter/apply
fsc.policy( 6 , 4 ) = 1;
fsc.transition( 7 , 6 , : ) = 1;
fsc.policy( 7 , 1 ) = 1;
fsc.transition( 8 , 7 , 1 ) = 1;
fsc.transition( 9 , 7 , 2 ) = 1;
fsc.transition( 10 , 7 , 3 ) = 1;
fsc.transition( 11 , 7 , 4 ) = 1;
fsc.transition( 7 , 7 , 5 ) = 1;

% next states, 8-11: same change scale and transition down
fsc.policy( 8 , 5 ) = 1;
fsc.policy( 9 , 6 ) = 1;
fsc.policy( 10 , 7 ) = 1;
fsc.policy( 11 , 8 ) = 1;
fsc.transition( 12 , 8:11 , : ) = 1;
fsc.policy( 12 , 2 ) = 1;
fsc.transition( 1 , 12 , : ) = 1;
