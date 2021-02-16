
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, plotting
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc

#----------------------------------------------------------------------------#
#                         Generate Grid and Obstacles                        #
#----------------------------------------------------------------------------#

end_goal_x      =   9     # position of initial and end point
end_goal_y      =   9
initial_pos_x   =   0
initial_pos_y   =   0
xlim_min        =   -0.5  # xlim and ylim of plots
xlim_max        =   10.5
ylim_min        =   -2
ylim_max        =   16
n               =   10    # size of square grid



N       =  10             # number of shooting points of every ocp

path_horizon = 3
obs_horizon  = 100

obstacles_option = 1
path_option = 1

path_x = np.linspace(0, 9, 10)

path_y = [ 0.0, 5.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
         
Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 1)
u = np.linspace(0,1,100)
global_path = interpolate.splev(u, Bspline_obj)


npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints)
                       
path_x = global_path[0]
path_y = global_path[1]
    
plt.figure()

for i in range(0,len(path_x)):
    pos_x = path_x[i]
    pos_y = path_y[i]
   
    global_path_x, global_path_y , Bspline_obj      =  create_global_path_mpc(path_option,pos_x,pos_y, path_horizon, N)
    occupied_positions_x, occupied_positions_y      =  create_obstacles_mpc(obstacles_option, pos_x, pos_y, obs_horizon)
    
    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    shifted_feasiblebubbles_x = []
    shifted_feasiblebubbles_y = []
    for i in range (0, len(shifted_midpoints_x)):
            shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
            shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))

    plt.plot(occupied_positions_x,occupied_positions_y,'bo', markersize = 1.5)
    # plt.plot(shifted_midpoints_x,shifted_midpoints_y, 'gx', markersize = 5)
    plt.plot(shifted_feasiblebubbles_x,shifted_feasiblebubbles_y, 'ro', markersize = 0.2)
        
    plt.plot(pos_x, pos_y, 'bx', markersize = 10) 
    plt.plot(global_path_x,global_path_y,'g--' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.001)














