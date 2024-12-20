from src.read_data import read_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

initial_data = read_data('vp_ic11all.dat')
next_data = read_data('vp_next11all.dat')

init_impact_phase = initial_data[:, 0]
init_impact_velocity = initial_data[:, 1]

X = initial_data[:, [0, 1]]
Y = next_data[:, [0, 1]]

next_impact_phase = next_data[:, 0]
next_impact_velocity = next_data[:, 1]

lower_bounds_init = [np.min(init_impact_phase), np.min(init_impact_velocity)]
upper_bounds_init = [np.max(init_impact_phase), np.max(init_impact_velocity)]
print('Lower init bounds: ', lower_bounds_init)
print('Upper init bounds ', upper_bounds_init)

lower_next_bounds = [np.min(next_impact_phase), np.min(next_impact_velocity)]
upper_next_bounds = [np.max(next_impact_phase), np.max(next_impact_velocity)]
print('Lower next bounds ', lower_next_bounds)
print('Upper next bounds ', upper_next_bounds)

lower_bounds = [min(lower_bounds_init[0], lower_next_bounds[0]) - 0.1, min(lower_bounds_init[1], lower_next_bounds[1]) - 0.1]
upper_bounds = [max(upper_bounds_init[0], upper_next_bounds[0]), max(upper_bounds_init[1], upper_next_bounds[1])]
print('Lower bounds:', lower_bounds)
print('Upper bounds:', upper_bounds)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Initial Phase')
ax.set_ylabel('Initial Velocity')
ax.set_zlabel('Return Phase')

# add the z=0 plane
x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
x, y = np.meshgrid(x, y)
z = np.zeros(x.shape)

#ax.plot_surface(x, y, z, alpha=0.5)

# Mask for positive and negative values
positive_mask = next_impact_phase > 0
negative_mask = next_impact_phase <= 0

# Scatter plot
ax.scatter(X[:,0][positive_mask], X[:,1][positive_mask], next_impact_phase[positive_mask], c='blue', s = 0.1, label='Positive return phase')
ax.scatter(X[:,0][negative_mask], X[:,1][negative_mask], next_impact_phase[negative_mask], c='purple', s = 0.1, label='Negative return phase')

plt.legend(markerscale=10)
ax.view_init(elev=60, azim=100)
plt.show()
plt.close()