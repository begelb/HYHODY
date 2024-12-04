import numpy as np
import matplotlib.pyplot as plt

class Example:
    def __init__(self, lower_bounds, upper_bounds, f):
        self.f = f
        self.x_min = lower_bounds[0]
        self.x_max = upper_bounds[0]
        self.y_min = lower_bounds[1]
        self.y_max = upper_bounds[1]
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

    # evaluate f on a grid of points
    def make_grid_data(self, resolution):
        x = np.linspace(self.x_min, self.x_max, resolution)
        y = np.linspace(self.y_min, self.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                u, v = self.f(X[i, j], Y[i, j])
                U[i, j] = u
                V[i, j] = v

        init_data = np.column_stack((X.flatten(), Y.flatten()))
        next_data = np.column_stack((U.flatten(), V.flatten()))
        return init_data, next_data

    def plot_data(self, init_data, next_data):
        plt.scatter(init_data[:, 0], init_data[:, 1], c='purple', s=0.1, label='Initial Data')
        plt.scatter(next_data[:, 0], next_data[:, 1], c='orange', s=0.1, label='Next Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(markerscale=10)
        plt.show()

    def plot(self, resolution=100):
        """
        Plots the contour of a function f: R^2 -> R^2.
        
        Parameters:
        - f: function that takes two inputs (x, y) and returns two outputs (u, v)
        - x_range: tuple (x_min, x_max) defining the x domain
        - y_range: tuple (y_min, y_max) defining the y domain
        - resolution: number of points along each axis
        
        Returns:
        - None: Displays the contour plots
        """
        x_range = (self.x_min, self.x_max)
        y_range = (self.y_min, self.y_max)
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute f over the grid
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                u, v = self.f(X[i, j], Y[i, j])
                U[i, j] = u
                V[i, j] = v

        # Plot contours
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot U (first component of f)
        c1 = axes[0].contourf(X, Y, U, levels=50, cmap='plasma')
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title('First Component of f')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot V (second component of f)
        c2 = axes[1].contourf(X, Y, V, levels=50, cmap='plasma')
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title('Second Component of f')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.tight_layout()
        plt.show()

        
            