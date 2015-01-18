function index = get_value_set_index( node , state , state_count )
    index = state + ( node - 1 ) * state_count;
