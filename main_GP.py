from HYHODY.src.read_data import read_data
import numpy as np
import CMGDB
import matplotlib
import math
import time
from HYHODY.src.map import return_map
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import math
import csv
import os
import GPy

''' Read the data files '''

initial_data = read_data('vp_ic11all.dat')
next_data = read_data('vp_next11all.dat')

init_impact_velocity = initial_data[:, 0]
init_impact_phase = initial_data[:, 1]
X = initial_data[:, [0, 1]]
Y = next_data[:, [0, 1]]

# randomly subsample data points from X and the corresponding data from Y
num_data = 2000
idx = np.random.choice(X.shape[0], num_data, replace=False)
X = X[idx]
Y = Y[idx]

next_impact_velocity = next_data[:, 0]
next_impact_phase = next_data[:, 1]

#print a list of the unique values of the init_impact_velocity
# print('Unique values of the initial impact velocity:')
# print(np.unique(init_impact_velocity))

# # print a list of the unique values of the init_impact_phase
# print('Unique values of the initial impact phase:')
# print(np.unique(init_impact_phase))

def f(initial_cond):
    return return_map(initial_cond, initial_data, next_data)

lower_bounds_init = [np.min(init_impact_velocity), np.min(init_impact_phase)]
upper_bounds_init = [np.max(init_impact_velocity), np.max(init_impact_phase)]

# print lower and upper bounds for the next data
lower_next_bounds = [np.min(next_impact_velocity), np.min(next_impact_phase)]
upper_next_bounds = [np.max(next_impact_velocity), np.max(next_impact_phase)]

lower_bounds = [min(lower_bounds_init[0], lower_next_bounds[0]) - 0.1, min(lower_bounds_init[1], lower_next_bounds[1]) - 0.1]
upper_bounds = [max(upper_bounds_init[0], upper_next_bounds[0]), max(upper_bounds_init[1], upper_next_bounds[1])]
print('Lower bounds:', lower_bounds)
print('Upper bounds:', upper_bounds)

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

''' Gaussian process using mean '''
# Define a Gaussian process
def GP(X_train, Y_train):
    # fit Gaussian Process with dataset X_train, Y_train
    # define kernel to be matern kernel
    kernel = Matern(nu = 1.5)
    #kernel = RBF(0.5, (0.01, 2)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X_train, Y_train)
    return gp

trial = 0

gp = GP(X, Y)

def f(X):
    return gp.predict([X])[0]

# Define box map for f
def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

# Define the parameters for CMGDB

subdiv_min = 15
subdiv_max = 20
init_subdivision = 8
subdivision_limit = 10000

morse_fname = 'morse_sets_Scikit.csv'

model = CMGDB.Model(subdiv_min, subdiv_max, init_subdivision, subdivision_limit, lower_bounds, upper_bounds, F)

plt.ioff()
morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)

graph = CMGDB.PlotMorseGraph(morse_graph)

# if folder does not exist, create it
if not os.path.exists(f'results/new_data11/matern/{num_data}_dps'):
    os.makedirs(f'results/new_data11/matern/{num_data}_dps')
graph.render(f'results/new_data11/matern/{num_data}_dps/{num_data}_dps_trial_{trial}_morse_graph', format='png', cleanup=True)

print('plotting')
matplotlib.use('Agg')
CMGDB.PlotMorseSets(morse_graph, fig_fname=f'results/new_data11/matern/{num_data}_dps/{num_data}_dps_trial_{trial}_morse_sets')