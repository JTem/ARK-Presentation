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
        slider_width = '60%'

        slider = widgets.FloatSlider(orientation='horizontal',description=name, value=start_val, min=min_val, max=max_val, step = 0.01, layout={'width': slider_width})
        return slider
    
    
    
def plot_trajectory_evaluation(trajectory, time_vec):
    time = [0]
    vel_norm_list = [0]
    acc_norm_list = [0]
    jerk_norm_list = [0]

    ang_vel_norm_list = [0]
    ang_acc_norm_list = [0]
    ang_jerk_norm_list = [0]

    quaternion_list = []

    x = []
    y = []
    z = []

    for t in time_vec:
        time.append(t)
        pos, vel, acc, jerk, quaternion, ang_vel, ang_acc, ang_jerk = trajectory.evaluate(t)

        if t == time_vec[-1]:
            vel_norm_list.append(0)
            acc_norm_list.append(0)
            jerk_norm_list.append(0)

            ang_vel_norm_list.append(0)
            ang_acc_norm_list.append(0)
            ang_jerk_norm_list.append(0)
        else:
            vel_norm_list.append(np.linalg.norm(vel))
            acc_norm_list.append(np.linalg.norm(acc))
            jerk_norm_list.append(np.linalg.norm(jerk))

            ang_vel_norm_list.append(np.linalg.norm(ang_vel))
            ang_acc_norm_list.append(np.linalg.norm(ang_acc))
            ang_jerk_norm_list.append(np.linalg.norm(ang_jerk))

        quaternion_list.append(quaternion)

        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        
        
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    text_size = 12
    plt.rcParams['font.size'] = text_size  # Adjust this value as needed
    plt.rcParams['axes.labelsize'] = text_size  # For x and y labels
    plt.rcParams['xtick.labelsize'] = text_size  # For x tick labels
    plt.rcParams['ytick.labelsize'] = text_size  # For y tick labels
    plt.rcParams['legend.fontsize'] = text_size  # For legend


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    ax1.plot(time, vel_norm_list, label = r'$v \left[ \frac{m}{s} \right]$')
    ax1.plot(time, acc_norm_list, label = r'$a \left[ \frac{m}{s^2} \right]$')
    ax1.set_yticks(np.arange(0, 1 + 0.5, 0.5))
    ax1.legend()

    ax2.plot(time, ang_vel_norm_list, label = r'$\omega \left[ \frac{rad}{s} \right]$')
    ax2.plot(time, ang_acc_norm_list, label = r'$\dot{\omega} \left[ \frac{rad}{s^2} \right]$')
    ax2.set_yticks(np.arange(0, 2 + 1, 1))
    ax2.legend()

    ax3.plot(time, jerk_norm_list, label =r'$j \left[ \frac{m}{s^3} \right]$')
    ax3.plot(time, ang_jerk_norm_list, label = r'$\ddot{\omega} \left[ \frac{rad}{s^3} \right]$')
    ax3.set_yticks(np.arange(0, 15 + 5, 5))
    ax3.set_xlabel(r'$\textrm{time } \left[ s \right]$')
    ax3.legend()

    # Place the legends outside the plot on the right side
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Enable grid
    ax1.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax2.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax3.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)

    # Adjust the layout
    plt.tight_layout()

    # To avoid clipping of the legend, adjust the subplot parameters
    #plt.subplots_adjust(right=0.8)  # Adjust this value based on your layout

    plt.show() 

def plot_dual_quaternion_evaluation(trajectory, time_vec):
    dqr_list = []
    dqr_dot_list = []
    dqr_ddot_list = []

    dqd_list = []
    dqd_dot_list = []
    dqd_ddot_list = []
    time_vec = time_vec[:-1]
    
    for t in time_vec:

        dq, dq_dot, dq_ddot = trajectory.evaluateDQ(t)

        dqr_list.append(dq.real.asVector().flatten())
        dqr_dot_list.append(dq_dot.real.asVector().flatten())
        dqr_ddot_list.append(dq_ddot.real.asVector().flatten())

        dqd_list.append(dq.dual.asVector().flatten())
        dqd_dot_list.append(dq_dot.dual.asVector().flatten())
        dqd_ddot_list.append(dq_ddot.dual.asVector().flatten())


    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 9))
    fig.suptitle(r"UDQ pose and derivatives of the online interpolation Algorithm")

    ax1.plot(time_vec, dqr_list, label = ["$w_r$", "$x_r$", "$y_r$", "$z_r$"])

    ax2.plot(time_vec, dqr_dot_list, label = ["$\dot{w}_r$", "$\dot{x}_r$", "$\dot{y}_r$", "$\dot{z}_r$"])

    ax3.plot(time_vec, dqr_ddot_list, label = ["$\ddot{w}_r$", "$\ddot{x}_r$", "$\ddot{y}_r$", "$\ddot{z}_r$"])
    
    ax4.plot(time_vec, dqd_list, label = ["$w_d$", "$x_d$", "$y_d$", "$z_d$"])

    ax5.plot(time_vec, dqd_dot_list, label = ["$\dot{w}_d$", "$\dot{x}_d$", "$\dot{y}_d$", "$\dot{z}_d$"])

    ax6.plot(time_vec, dqd_ddot_list, label = ["$\ddot{w}_d$", "$\ddot{x}_d$", "$\ddot{y}_d$", "$\ddot{z}_d$"])

    ax1.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax2.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax3.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax4.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax5.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
    ax6.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)

    # Place the legends outside the plot on the right side
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax5.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax6.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # # Adjust the layout and display the plot
    plt.tight_layout()
    
     # To avoid clipping of the legend, adjust the subplot parameters
    #plt.subplots_adjust(right=0.8)  # Adjust this value based on your layout
    
    plt.show()

        


