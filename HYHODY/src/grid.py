import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
import matplotlib.patches as patches
from ripser import ripser
from ripser import Rips

class Boxes:
    # initialize a grid with a certain number of subdivisions to make rectangles
    def __init__(self, lower_bounds, upper_bounds, num_subdivisions):
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
                #print(Rect(rect_x_min, rect_x_max, rect_y_min, rect_y_max))

    def make_data_dict(self, init_data, next_data):
        rect_dict = {}
        for rect in self.rectangles:
            rect_dict[rect] = []
            for i, point in enumerate(init_data):
                if rect.contains(point[0], point[1]):
                    rect_dict[rect].append(next_data[i])
        # print the first element of the dictionary
        #print(next(iter(rect_dict.values())))
        return rect_dict
    
    def do_persistence(self, data):
        data_array = np.array(data)
        # produce unique data_array
        data_array_unique = np.unique(data_array, axis=0)

        if len(data_array_unique) == 1:
            print('unique data array flag')
            return 0
        
        result = ripser(data_array, maxdim=0)
        diagram = result['dgms'][0]

        lengths = [death - birth for birth, death in diagram if death != np.inf]

        max_num = max(lengths)
        second_max = lengths[-2]

        return second_max/max_num
    
    def make_persistence_dict(self, init_data, next_data):
        rect_dict = self.make_data_dict(init_data, next_data)
        pers_dict = {}
        for rect in rect_dict:
            if len(rect_dict[rect]) > 1: # if more than one data point, do persistence
                pers_dict[rect] = self.do_persistence(rect_dict[rect])
            else:
                pers_dict[rect] = None
        return pers_dict
    
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
            # make title
            plt.title('Ratio of second largest to largest finite barcode length')
        else:
            heights = self.get_last_height(init_data, next_data, method)
        # Filter out None values
        valid_heights = [h for h in heights.values() if h is not None]
        
        if not valid_heights:
            print("No valid heights to plot.")
            return

        max_height = max(valid_heights)
        # k = 0
        for rect, height in heights.items():
            # if k % 2 == 0:
            #     color = 'white'
            # else:
            #     color = 'black'
            if height is not None:
                # make color grayscale color
                color = plt.cm.viridis(height / max_height)
                rect_patch = patches.Rectangle((rect.x_min, rect.y_min),
                                            rect.width, rect.height,
                                            color=color)
                ax.add_patch(rect_patch)

        # add colorbarx
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_height))
      #  sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_height))
        plt.colorbar(sm)
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

    # function for printing the rectangle
    def __repr__(self):
        return f"Rect({self.x_min}, {self.x_max}, {self.y_min}, {self.y_max})"

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    # plot the square filled in a certain color

    def plot(self, color):
        fig, ax = plt.subplots()
        rect_patch = patches.Rectangle((self.x_min, self.y_min),
                                       self.width, self.height,
                                       color=color)
        ax.add_patch(rect_patch)

    # def plot(self, color):
    #     plt.fill(
    #         [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min], 
    #         [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min], 
    #         color
    #     )