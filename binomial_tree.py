import numpy as np
from tqdm import tqdm

def get_rate_at_node(initial_rate, up_move, down_move, time, state):
    """S_i(j) is the value of non-dividend / coupon paying security
    at time i and state j

    This function returns the no-arbitrage rate for node i at state j.
    
    >>> get_rate_at_node(.06, 1.25, .9, 4, 3)
    0.105469

    >>> get_rate_at_node(.06, 1.25, .9, 5, 5)
    0.18

    >>> get_rate_at_node(.06, 1.25, .9, 3, 1)
    0.06
    """
    if time < state:
        raise ValueError('State (i) cannot be greater than time (j)')
    
    return (up_move ** state) * down_move ** (time -state) * initial_rate

def build_rates_tree(initial_rate, up_move, down_move, Time):
    """S_i(j) is the value of non-dividend / coupon paying security
    at time i and state j

    This function builds the complete binomial lattice
    
    >>> tree = build_rates_tree(.06, 1.25, .9, 5)
    >>> tree[4,3]
    0.105469

    >>> tree = build_rates_tree(.06, 1.25, .9, 5)
    >>> tree[5,1]
    0.0492075

    >>> tree = build_rates_tree(.06, 1.25, .9, 5)
    >>> tree[0,0]
    .06

    >>> tree = build_rates_tree(.06, 1.25, .9, 5)
    >>> np.sum(sum(tree[5,:]))
    0.562843931

    >>> tree = build_rates_tree(.06, 1.25, .9, 5)
    >>> tree[1,1]
    0.075000
    
    """
    
    Time += 1

    tree = np.full((Time,Time), np.NaN)

    tree[0,0] = initial_rate

    for time in tqdm(range(Time), "Build rate tree"):  # i
        for state in range(time + 1):  #j

            tree[time, state] = get_rate_at_node(initial_rate,
                                            up_move,
                                            down_move,
                                            time,
                                            state)

    return tree

def build_prices_tree(initial_rate, up_move, down_move, Time):

    """Price at T is 100, work backwards using abritrage-free binomial tree

    >>> price_tree = build_prices_tree(.06, 1.25, .9, 4) 
    >>> price_tree[0,0]
    77.21774

    >>> price_tree = build_prices_tree(.06, 1.25, .9, 4) 
    >>> price_tree[2,2]
    83.08

    """

    price_tree = np.full((Time+1, Time+1), np.NaN)

    price_tree[Time, :] = 100.00

    rates_tree = build_rates_tree(initial_rate, up_move, down_move, Time)

    for time in tqdm(range(Time-1, -1, -1)):
        for step in range(time, -1, -1):

            rate = rates_tree[time, step]

            discount_factor = 1/(1 + rate)
            
            prior_up_price = price_tree[time+1,step+1]

            prior_down_price = price_tree[time+1,step]

            price = ((0.5 * prior_up_price) + (0.5 * prior_down_price)) * discount_factor

            price_tree[time,step] = price
    
    return price_tree
            
def get_spot_rate(short_rate, up_move, down_move, periods):
    """Returns the spot rate generated from a no-arbitrage binomial tree

    Bond Price = 100/(1+spot)^periods

    >>> get_spot_rate(.06, 1.25, .9, 4)
    .06676

    """

    bond_price = build_prices_tree(short_rate, up_move, down_move, periods)[0,0]

    spot_rate = ((100/bond_price)**(1/periods)) - 1

    return spot_rate

