import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plot():

    @staticmethod
    def plot_frame(marker_data_frame):
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        data = np.array([marker_data_frame[marker] for marker in marker_data_frame.keys()])

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=16, marker='o')

        # Layout
        max_data = np.nanmax(data, axis=0)
        min_data = np.nanmin(data, axis=0)
        mid_data = (max_data + min_data) * 0.5
        range = max_data - min_data
        max_range = np.nanmax(range) * 0.5
        ax.set_xlim(mid_data[0] - max_range, mid_data[0] + max_range)
        ax.set_ylim(mid_data[1] - max_range, mid_data[1] + max_range)
        ax.set_zlim(mid_data[2] - max_range, mid_data[2] + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return fig, ax

    @staticmethod
    def quiver(ax, origin, vector, color):
        ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color=color, length=0.2)

    @staticmethod
    def draw_R(ax, origin, x_axis, y_axis, z_axis):
        Plot.quiver(ax, origin, x_axis, "red")
        Plot.quiver(ax, origin, y_axis, "green")
        Plot.quiver(ax, origin, z_axis, "blue")
