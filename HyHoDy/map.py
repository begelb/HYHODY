import numpy as np
import CMGDB

# function that given an initial impact velocity and phase, and the data of initial and next conditions, returns the next impact velocity and phase
def return_map(input, initial_data, next_data):
    init_impact_velocity = input[0]
    init_impact_phase = input[1]
    tolerance_velocity = 0.04 #0.02 
    tolerance_phase = 0.005 #0.0025

    # Find indices within the tolerance
    idx = np.where(
        (np.abs(initial_data[:, 0] - init_impact_velocity) <= tolerance_velocity) &
        (np.abs(initial_data[:, 1] - init_impact_phase) <= tolerance_phase))[0][0]
    
    # Return the next impact velocity and phase
    return [next_data[idx, 0], next_data[idx, 1]]