import numpy as np
import matplotlib.pyplot as plt
import CMGDB_utils
from HyHoDy.grid import cluster_data
import matplotlib

# Load data
init_data_file2 = 'data/vel_phase_ic.dat'
next_data_file2 = 'data/vel_phase_next.dat'

init_data2 = np.loadtxt(init_data_file2)
next_data2 = np.loadtxt(next_data_file2)

X2 = init_data2[:, [0, 1]]
Y2 = next_data2[:, [0, 1]]

# Mod the phase variable of X by 6.2828711
X2[:, 0] = X2[:, 0] % (6.2828711)

lower_bounds = [0, 0]
upper_bounds = [6.2828711, 1.25]
discont_thresh = 0.2


F_BoxMap = CMGDB_utils.BoxMapData(X2, Y2)

def F(rect):
    Y0 = F_BoxMap.interpolate(rect)
    print('Y0: ', Y0)
    cluster_dicts = cluster_data(Y0, lower_bounds, upper_bounds, discontinuity_threshold=discont_thresh, method='Gaussian', separate_two_points=False)
    F_rect = []
    for cluster_dict in cluster_dicts:
        cluster_pts = np.array(cluster_dict['cluster_pts'])
        bdry_dists = cluster_dict['bdry_dist']
        print('cluster_pts: ', cluster_pts)
        Y_l_bounds = list(np.min(cluster_pts, axis=0))
        Y_u_bounds = list(np.max(cluster_pts, axis=0))
        if bdry_dists != None:
            if bdry_dists[0] < bdry_dists[1]:
                Y_l_bounds[0] = lower_bounds[0]
            else:
                Y_u_bounds[0] = upper_bounds[0]
        F_rect.append(Y_l_bounds + Y_u_bounds)
    return F_rect

grid_size = [200, 200]
model = CMGDB_utils.Model(lower_bounds, upper_bounds, grid_size, F, map_type='MultiBoxMap')
morse_graph6, morse_decomp6, vertex_mapping6, cubical_complex6 = CMGDB_utils.ComputeMorseGraph(model)

CMGDB_utils.PlotMorseGraph(morse_graph6, cmap=matplotlib.cm.jet)
CMGDB_utils.PlotMorseSets(morse_graph6, morse_decomp6, vertex_mapping6, cubical_complex6, cmap=matplotlib.cm.jet)