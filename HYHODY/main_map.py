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
import dytop.Grid as Grid
import dytop.CMGDB_util as CMGDB_util

''' Read the data files '''

initial_data = read_data('vp_ic11all.dat')
next_data = read_data('vp_next11all.dat')

init_impact_velocity = initial_data[:, 0]
init_impact_phase = initial_data[:, 1]

X = initial_data[:, [0, 1]]
Y = next_data[:, [0, 1]]

next_impact_velocity = next_data[:, 0]
next_impact_phase = next_data[:, 1]

lower_bounds_init = [np.min(init_impact_velocity), np.min(init_impact_phase)]
upper_bounds_init = [np.max(init_impact_velocity), np.max(init_impact_phase)]
print('Lower init bounds: ', lower_bounds_init)
print('Upper init bounds ', upper_bounds_init)

lower_next_bounds = [np.min(next_impact_velocity), np.min(next_impact_phase)]
upper_next_bounds = [np.max(next_impact_velocity), np.max(next_impact_phase)]
print('Lower next bounds ', lower_next_bounds)
print('Upper next bounds ', upper_next_bounds)

lower_bounds = [min(lower_bounds_init[0], lower_next_bounds[0]) - 0.1, min(lower_bounds_init[1], lower_next_bounds[1]) - 0.1]
upper_bounds = [max(upper_bounds_init[0], upper_next_bounds[0]), max(upper_bounds_init[1], upper_next_bounds[1])]
print('Lower bounds:', lower_bounds)
print('Upper bounds:', upper_bounds)

data = np.concatenate((X,Y),axis=1)

subdiv_limit = 10000

# k is the padding
for k in [0.75]:
    for sb in range(10, 20): # subdivision
        grid = Grid.Grid(lower_bounds, upper_bounds, sb)

        id2image = grid.id2image(data)

        # Define box map for f
        MG_util = CMGDB_util.CMGDB_util()
        K=[k, k]
        def F(rect):
            return MG_util.F_data(rect, id2image, grid.point2cell, K)

        subdiv_init = sb
        subdiv_min = sb
        subdiv_max = sb
        model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                            lower_bounds, upper_bounds, F)

        plt.ioff()

        try:
            morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)

            graph = CMGDB.PlotMorseGraph(morse_graph)

            # if folder does not exist, create it
            if not os.path.exists(f'results/map/new_data11//sblimit20000/k_{k}'):
                os.makedirs(f'results/map/new_data11//sblimit20000/k_{k}')
            graph.render(f'results/map/new_data11//sblimit20000/k_{k}/sb_{sb}_morse_graph', format='png', cleanup=True)

            print('plotting')
            matplotlib.use('Agg')
            print('saving at ' + f'results/map/new_data11//sblimit20000/k_{k}/sb_{sb}_morse_sets.png')
            CMGDB.PlotMorseSets(morse_graph, fig_fname=f'results/map/new_data11//sblimit20000/k_{k}/sb_{sb}_morse_sets.png')
        
        except:
            print('error')

