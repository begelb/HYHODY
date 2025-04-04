import numpy as np
from HyHoDy.grid import Boxes, cluster_data
import matplotlib.pyplot as plt
    
init_data_file = 'data/vp_ic11all.dat'
next_data_file = 'data/vp_next11all.dat'

init_data = np.loadtxt(init_data_file)
next_data = np.loadtxt(next_data_file)


lower_bounds = [0, 0]
upper_bounds = [6.2828542917900005, 1.25]
boxes = Boxes(lower_bounds, upper_bounds, 20, phase_periodic=True)

X = init_data[:, [0, 1]]
Y = next_data[:, [0, 1]]

Y[:, 0] = Y[:, 0] % 6.2828243669441

next_data = np.column_stack((Y, next_data[:, 2]))

plt.xlim(lower_bounds[0], upper_bounds[0])
plt.ylim(lower_bounds[1], upper_bounds[1])

# ''' Example 1: One cluster in the middle of the cylinder '''
# a, b = 1.2565708583580002, 1.5078850300296003
# c, d = 1.2000000000000002, 1.2500000000000002

# filtered_Y = Y[(X[:, 0] >= a) & (X[:, 0] <= b) & (X[:, 1] >= c) & (X[:, 1] <= d)]
# cluster_dicts = cluster_data(filtered_Y, lower_bounds, upper_bounds, discontinuity_threshold=0.2, method = 'Gaussian')
# print('Number of clusters: ', len(cluster_dicts))

# for cluster_dictionary in cluster_dicts:
#     print('Boundary distance: ', cluster_dictionary['bdry_dist'])
#     cluster_pts = np.array(cluster_dictionary['cluster_pts'])
#     print('Cluster points: ', cluster_pts)
#     plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color = np.random.rand(3,))
#     plt.show()

''' Example 2: One cluster that crosses the periodic boundary '''

a, b = 0.0, 0.25131417167160003
c, d = 1.1500000000000001, 1.2000000000000002

filtered_Y = Y[(X[:, 0] >= a) & (X[:, 0] <= b) & (X[:, 1] >= c) & (X[:, 1] <= d)]
cluster_dicts = cluster_data(filtered_Y, lower_bounds, upper_bounds, discontinuity_threshold=0.2, method = 'Gaussian')
print('Number of clusters: ', len(cluster_dicts))

for cluster_dictionary in cluster_dicts:
    print('Boundary distance: ', cluster_dictionary['bdry_dist'])
    cluster_pts = np.array(cluster_dictionary['cluster_pts'])
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color = np.random.rand(3,))
plt.show()
    
''' Example 3: Two clusters '''
a, b = 0.753, 1.004
c, d = 0.40000136, 0.45000127999999995

filtered_Y = Y[(X[:, 0] >= a) & (X[:, 0] <= b) & (X[:, 1] >= c) & (X[:, 1] <= d)]
cluster_dicts = cluster_data(filtered_Y, lower_bounds, upper_bounds, discontinuity_threshold=0.2, method = 'Gaussian')
print('Number of clusters: ', len(cluster_dicts))

for cluster_dictionary in cluster_dicts:
    print('Boundary distance: ', cluster_dictionary['bdry_dist'])
    cluster_pts = np.array(cluster_dictionary['cluster_pts'])
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color = np.random.rand(3,))
plt.show()

