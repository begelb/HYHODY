import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
import matplotlib.patches as patches
from ripser import ripser
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

def generate_contrast_colormap(n_colors=40, emphasize_first=5):
    colors = []
    
    # First approach: For the first few colors, use distinct colormaps
    distinct_cmaps = [plt.cm.viridis, plt.cm.plasma, plt.cm.cool, plt.cm.autumn, plt.cm.tab10, 
                      plt.cm.Set1, plt.cm.Dark2, plt.cm.Set3]
    
    # Get the first few highly distinct colors
    if emphasize_first > 0:
        # Use distinct color points from different colormaps
        for i in range(min(emphasize_first, len(distinct_cmaps))):
            cmap = distinct_cmaps[i]
            # Sample from different parts of each colormap
            point = 0.1 + (0.8 * (i % 3) / 3)  # Vary within the colormap too
            colors.append(cmap(point))
            
        # If we need more distinct colors than we have colormaps
        if emphasize_first > len(distinct_cmaps):
            # Use tab10/Set1 which have distinct categorical colors
            categorical_cmap = plt.cm.tab10
            for i in range(emphasize_first - len(distinct_cmaps)):
                idx = i % 10  # tab10 has 10 distinct colors
                colors.append(categorical_cmap(idx / 10))
    
    # For the remaining colors, use a perceptually uniform colormap with decreasing contrast
    remaining = n_colors - len(colors)
    if remaining > 0:
        # Create non-linear spacing that starts more spread out
        spacing = np.power(np.linspace(0, 1, remaining), 0.7)  # Power < 1 emphasizes early differences
        
        # Use the viridis colormap for the remaining colors
        colors.extend([plt.cm.viridis(point) for point in spacing])
    
    # Convert to hex codes
    hex_colors = [mcolors.to_hex(color) for color in colors]
    
    return hex_colors

class Cylinder:
    def __init__(self, lower_bounds, upper_bounds):
        self.x_min = lower_bounds[0]
        self.x_max = upper_bounds[0]
        self.y_min = lower_bounds[1]
        self.y_max = upper_bounds[1]

    def cylinder_first_coordinate_distance(self, x0_in, x1_in):
        x0 = min(x0_in, x1_in)
        x1 = max(x0_in, x1_in)
        z = x0 - self.x_min + self.x_max - x1
        if x1 - x0 <= z:
            data_wrap_Bool = False
            return x1 - x0, data_wrap_Bool
        else:
            data_wrap_Bool = True
            return z, data_wrap_Bool
        
    def cylinder_euclidean_distance(self, x0, y0, x1, y1):
        d, data_wrap_Bool = self.cylinder_first_coordinate_distance(x0, x1)
        return d, np.sqrt(d ** 2 + (y0 - y1) ** 2), data_wrap_Bool
    
    def cylinder_distance_matrix(self, data_array):
        num_points = len(data_array)

        distance_matrix = np.zeros((num_points, num_points))
        data_wrap_Bool_matrix = np.zeros((num_points, num_points), dtype=bool)
        x_diff_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                x_diff, distance_entry, data_wrap_Bool_entry = self.cylinder_euclidean_distance(data_array[i, 0], data_array[i, 1], data_array[j, 0], data_array[j, 1])
                distance_matrix[i, j] = distance_entry
                data_wrap_Bool_matrix[i, j] = data_wrap_Bool_entry
                x_diff_matrix[i, j] = x_diff
        return distance_matrix + distance_matrix.T, data_wrap_Bool_matrix, x_diff_matrix
    

''' Persistence '''  
def death_time_ratio(data, cylinder):
    data_array = np.array(data)
    distance_matrix, wrap_matrix, x_diff_matrix = cylinder.cylinder_distance_matrix(data_array)
    result = ripser(distance_matrix, maxdim=0, distance_matrix = True)
    diagram = result['dgms'][0]
    lengths = [death - birth for birth, death in diagram if death != np.inf]
    max_num = max(lengths)
    second_max = lengths[-2]

    return second_max/max_num, wrap_matrix, x_diff_matrix

''' Analyze f(X) to define mv map '''

def get_num_clusters(ratio, discontinuity_threshold):
    if ratio <= discontinuity_threshold:
        return 2
    else:
        return 1
    
def get_2_means_labels(data):
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    # print('Warning: k-means was done using the Euclidean distance. This will be a bug if there are multiple clusters on the cylinder, and one cluster wraps around.')
    return labels


def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

def get_Gaussian_mixture_labels(data, n_iterations=8):
    # clustering is done five times and the best cluster is returned according to the silhouette score
    best_labels = None
    for i in range(n_iterations):
        gmm = GaussianMixture(n_components=2) 
        labels = gmm.fit_predict(data)
        sil=silhouette_score(data, labels, metric='euclidean')
       # print('silhouette score: ', sil)
        if best_labels is None or sil > best_silhouette:
            best_labels = labels
            best_silhouette = sil
    return best_labels

def get_submatrix(cluster_labels, id, matrix):
    cluster_indices = np.where(cluster_labels == id)[0]
    return matrix[np.ix_(cluster_indices, cluster_indices)]

# def get_boundary_distances(x_diff_matrix, data, cylinder):
#     flattened_matrix = x_diff_matrix.flatten()
#     max_id = np.argmax(flattened_matrix)
#     i, j = np.unravel_index(max_id, x_diff_matrix.shape)
#     x0 = data[i][0]
#     x1 = data[j][0]
#     x_small = min(x0, x1)
#     x_large = max(x0, x1)
#     print('x_small: ', x_small)
#     print('x_large: ', x_large)
#     left_bdry_dist = x_small - cylinder.x_min
#     right_bdry_dist = cylinder.x_max - x_large
#     return [left_bdry_dist, right_bdry_dist]

def get_boundary_distances(data, cylinder): # assumes that data is coming from one cluster
    x_max = np.max(data[:, 0])
    x_min = np.min(data[:, 0])
    left_bdry_dist = x_max - cylinder.x_min
    right_bdry_dist = cylinder.x_max - x_min
    return [left_bdry_dist, right_bdry_dist]

def get_clusters(data, method):
    if method == 'kmeans':
        cluster_labels = get_2_means_labels(data)
    elif method == 'Gaussian':
        cluster_labels = get_Gaussian_mixture_labels(data)
    else:
        raise NotImplementedError("The choices for method are kmeans or Gaussian")
    cluster_0 = data[cluster_labels == 0]
    cluster_1 = data[cluster_labels == 1]
    clusters = [cluster_0, cluster_1]
    return clusters, cluster_labels

def get_cluster_data_dicts(data, num_clusters, wrap_matrix, x_diff_matrix, lower_bounds, upper_bounds, method):
    cylinder = Cylinder(lower_bounds, upper_bounds)
    if num_clusters == 1:
        # case when cluster crosses the periodic boundary
        if np.any(wrap_matrix.flatten()):
          #  boundary_distance = get_boundary_distances(x_diff_matrix, data, cylinder)
            clusters, cluster_labels = get_clusters(data, method) # cluster the points on the rectangle using 2-means (Euclidean metric)

            dictionary_list = []
            for i, cluster in enumerate(clusters):
               # print('cluster: ', cluster)
               # cluster_x_diff_matrix = get_submatrix(cluster_labels, id=i, matrix=x_diff_matrix)
                boundary_distance = get_boundary_distances(cluster, cylinder)

                cluster_dict = {
                    'cluster_pts' : cluster,
                    'bdry_dist' : boundary_distance
                }
                dictionary_list.append(cluster_dict)
            return dictionary_list
        
        # case when there is one cluster, and it does not cross the periodic boundary
        else:
            boundary_distance = None
            clusters = data

            cluster_dict = {
                    'cluster_pts' : clusters,
                    'bdry_dist' : boundary_distance
                }
            
            return [cluster_dict]
        
    # two clusters
    elif num_clusters == 2:
        clusters, cluster_labels = get_clusters(data, method)

        dictionary_list = []
        for i, cluster in enumerate(clusters):
            cluster_wrap_matrix = get_submatrix(cluster_labels, id=i, matrix=wrap_matrix)
           # cluster_x_diff_matrix = get_submatrix(cluster_labels, id=i, matrix=x_diff_matrix)
            if np.any(cluster_wrap_matrix.flatten()):
                boundary_distance = get_boundary_distances(data, cylinder)
            else:
                boundary_distance = None

            cluster_dict = {
                'cluster_pts' : cluster,
                'bdry_dist' : boundary_distance
            }
            dictionary_list.append(cluster_dict)
        return dictionary_list
        
    else:
        raise NotImplementedError("The function get_num_clusters currently only returns 1 or 2")

def cluster_data(data, lower_bounds, upper_bounds, discontinuity_threshold=0.3, method='Gaussian', separate_two_points=False):
    data = np.array(data)
    if len(data) == 0:
        raise Exception("Empty box")
    
    # If given a single point, return that point as a cluster
    if len(data) == 1:
        cluster_dict = {
                'cluster_pts' : data,
                'bdry_dist' : None
            }
        return [cluster_dict]
    
    # If given two points, return them as two distinct clusters
    if len(data) == 2:
        if separate_two_points: # if separate_two_points is True, return two clusters, one containing each point
            dictionary_list = []
            for i in range(2):
                cluster_dict = {
                    'cluster_pts' : np.array([data[i]]),
                    'bdry_dist' : None
                }
                dictionary_list.append(cluster_dict)
            return dictionary_list
        else: # if separate_two_points is False, return a single cluster with both points
            cluster_dict = {
                'cluster_pts' : data,
                'bdry_dist' : None
            }
        return [cluster_dict]
    
    else:
        cylinder = Cylinder(lower_bounds, upper_bounds)
        ratio, wrap_matrix, x_diff_matrix = death_time_ratio(data, cylinder)
        num_clusters = get_num_clusters(ratio, discontinuity_threshold)
        cluster_dictionary_list = get_cluster_data_dicts(data, num_clusters, wrap_matrix, x_diff_matrix, lower_bounds, upper_bounds, method)
    return cluster_dictionary_list


''' Analysis of f(X) over a space discretization '''
class Boxes:
    # initialize a grid with a certain number of subdivisions to make rectangles
    def __init__(self, lower_bounds, upper_bounds, num_subdivisions, phase_periodic=False):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.cylinder = Cylinder(self.lower_bounds, self.upper_bounds)
        self.phase_periodic = phase_periodic
        self.x_min = lower_bounds[0]
        self.x_max = upper_bounds[0]
        self.y_min = lower_bounds[1]
        self.y_max = upper_bounds[1]
        self.num_subdivisions = num_subdivisions
        self.x_step = (self.x_max - self.x_min) / num_subdivisions
        self.y_step = (self.y_max - self.y_min) / num_subdivisions
        self.rectangles = []
        x_pairs = []
        y_pairs = []
        for i in range(num_subdivisions):
            box_x_min = self.x_min + i * self.x_step
            box_x_max = box_x_min + self.x_step
            x_pairs.append((box_x_min, box_x_max))
        for i in range(num_subdivisions):
            box_y_min = self.y_min + i * self.y_step
            box_y_max = box_y_min + self.y_step
            y_pairs.append((box_y_min, box_y_max))
        for i in range(num_subdivisions):
            rect_x_min, rect_x_max = x_pairs[i]
            for j in range(num_subdivisions):
                rect_y_min, rect_y_max = y_pairs[j]
                self.rectangles.append(Rect(rect_x_min, rect_x_max, rect_y_min, rect_y_max))

    def make_data_dict(self, init_data, next_data):
        rect_dict = {}
        for rect in self.rectangles:
            rect_dict[rect] = []
            for i, point in enumerate(init_data):
                if rect.contains(point[0], point[1]):
                    rect_dict[rect].append(next_data[i])
        return rect_dict
    
    def make_persistence_dict(self, init_data, next_data):
        rect_dict = self.make_data_dict(init_data, next_data)
        pers_dict = {}
        for rect in rect_dict:
          #  print('working on ', rect)
            if len(rect_dict[rect]) > 1: # if more than one data point, do persistence
                pers_dict[rect], wrap_matrix, x_diff_matrix = death_time_ratio(rect_dict[rect], self.cylinder)
                cluster_dicts = cluster_data(rect_dict[rect], self.lower_bounds, self.upper_bounds, discontinuity_threshold=0.3)
                # for cluster_dictionary in cluster_dicts:
                  #  print(cluster_dictionary['bdry_dist'])

            else:
                pers_dict[rect] = None
        return pers_dict
    
    def make_return_type_dict(self, init_data, next_data):
        rect_dict = self.make_data_dict(init_data, next_data)
        return_type_dict = {}
        for rect in rect_dict:
            third_elements = [arr[2] for arr in rect_dict[rect]]
            # number of unique third_elements
            num_unique_returns = len(set(third_elements))
            return_type_dict[rect] = num_unique_returns
        return return_type_dict
    
    # perform linkage(data, method='ward') on every rectangle
    def make_linkage(self, init_data, next_data, method):
        rect_dict = self.make_data_dict(init_data, next_data)
        linkage_dict = {}
        for rect in rect_dict:
            if len(rect_dict[rect]) > 0:
                linkage_dict[rect] = linkage(rect_dict[rect], method)
            else:
                linkage_dict[rect] = None
        return linkage_dict
    

    # get the last height of every linkage for all the rectangles
    def get_last_height(self, init_data, next_data, method):
        linkage_dict = self.make_linkage(init_data, next_data, method)
        heights = {}
        for rect in linkage_dict:
            if linkage_dict[rect] is not None:
                heights[rect] = linkage_dict[rect][:, 2][-1]
            else:
                heights[rect] = None
        return heights

    def plot(self, init_data, next_data, method):
        fig, ax = plt.subplots()
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        if method == 'persistence':
            heights = self.make_persistence_dict(init_data, next_data)
            plt.title('Ratio of second largest to largest finite death time')
        
        elif method == 'return_type':
            heights = self.make_return_type_dict(init_data, next_data)
            plt.title('Number of unique return types')

        else:
            heights = self.get_last_height(init_data, next_data, method)

        valid_heights = [h for h in heights.values() if h is not None]
        
        if not valid_heights:
            print("No valid heights to plot.")
            return

        if method == 'persistence':
            max_height = max(valid_heights)
            for rect, height in heights.items():
                if height is not None:
                    color = plt.cm.viridis(height / max_height)
                    rect_patch = patches.Rectangle((rect.x_min, rect.y_min),
                                                rect.width, rect.height,
                                                color=color)
                    ax.add_patch(rect_patch)

        elif method == 'return_type':
            for rect, height in heights.items():
                if height is not None:
                    if height == 1:
                        color = '#332288'
                    elif height == 2:
                        color = '#DDCC77'
                    rect_patch = patches.Rectangle((rect.x_min, rect.y_min),
                                                rect.width, rect.height,
                                                color=color)
                    ax.add_patch(rect_patch)
                else:
                    print('height is None')

        if method == 'persistence':
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_height))
            sm.set_array([]) 
            plt.colorbar(sm, ax=ax)
        elif method == 'return_type':
            # Define fixed color mapping
            color_map = {1: '#332288', 2: '#DDCC77'}

            # Create legend handles
            legend_handles = [
                patches.Patch(color=color, label=f"{rtype} type" if rtype == 1 else f"{rtype} types")
                for rtype, color in color_map.items()]

            # Add legend to the plot
            ax.legend(handles=legend_handles, loc='lower right')
        plt.xlabel(r'$\psi$ (phase)')
        plt.ylabel(r'$\dot{Z}$ (velocity)')

        plt.show()
        plt.close()



class Rect:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid rectangle coordinates")
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = x_max - x_min
        self.height = y_max - y_min

    def __repr__(self):
        return f"Rect({self.x_min}, {self.x_max}, {self.y_min}, {self.y_max})"

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def plot(self, color):
        fig, ax = plt.subplots()
        rect_patch = patches.Rectangle((self.x_min, self.y_min),
                                       self.width, self.height,
                                       color=color)
        ax.add_patch(rect_patch)