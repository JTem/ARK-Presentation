import ipywidgets as widgets
import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import HTML

from DualQuaternionQuinticBlends.LineGenerator import LineGenerator
from DualQuaternionQuinticBlends.ArcGenerator import ArcGenerator
from DualQuaternionQuinticBlends.DQQBTrajectoryGenerator import DQQBTrajectoryGenerator

np.set_printoptions(precision=2, suppress=True, linewidth=200, formatter={'float': '{:8.3f}'.format})

def create_3d_plot(qr = Quaternion(1,0,0,0)):
    
        plt.ioff()
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.header_visible = False
        fig.canvas.layout.min_height = '400px'
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor('white')
    
        for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()

        start_point = [0, 0, 0]
        R_base = qr.asRotationMatrix()*1.5
        draw_frame(ax, start_point, R_base)

        return fig, ax


def draw_frame(ax, start_point, R):
    
        x_axis = ax.quiver(*start_point, *R[:,0], arrow_length_ratio = 0.1, linewidth = 1, color='r')
        y_axis = ax.quiver(*start_point, *R[:,1], arrow_length_ratio = 0.1, linewidth = 1, color='g')
        z_axis = ax.quiver(*start_point, *R[:,2], arrow_length_ratio = 0.1, linewidth = 1, color='b')
        return x_axis, y_axis, z_axis

def create_slider(name, start_val, min_val, max_val):
        slider_width = '98%'

        slider = widgets.FloatSlider(orientation='horizontal',description=name, value=start_val, min=min_val, max=max_val, step = 0.01, layout={'width': slider_width})
        return slider