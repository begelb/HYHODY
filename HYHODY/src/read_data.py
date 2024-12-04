# read the data from data folder

import numpy as np
import os

# read data from .dat file
def read_data(file_name):
    file_path = os.path.join('data', file_name)
    data = np.loadtxt(file_path)
    return data
