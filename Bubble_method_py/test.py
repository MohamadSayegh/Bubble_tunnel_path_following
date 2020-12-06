
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from Bubble_tunnel_generation_v2 import generate_bubbles_v2, plotting_v2
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


''' 
possible combinations of the two options below: 1/1 2/1

option 1/1 works 

option 2/1 doesnt work 

''' 

obstacles_option = 1 
path_option = 1

path_x = np.linspace(0, 9, 37)

path_y = [ 0.    ,  2.0   , 4.0   , 6.0   , 8.0   , 9.8   , 9.8   ,  9.8  ,  9.8   ,  9.8  ,  9.8   ,
           9.8   ,  9.0   , 8.0   , 3.0   , 2.0   , 1.0   , 0.1   ,  0.1   , 0.1   ,  0.1   , 0.1   ,
           0.1   ,  0.1   , 0.1   , 4.0   , 5.0   , 6.0   , 9.0   ,  9.0   , 9.0   ,  9.0   , 9.0   ,
           9.0   ,  9.0   , 9.0   , 9.0 ]
        

npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints)
                           
plt.figure()

for i in range(0,len(path_x)):
    pos_x = path_x[i]
    pos_y = path_y[i]
    global_path_x, global_path_y = create_global_path_mpc(path_option,pos_x,pos_y)
    occupied_positions_x, occupied_positions_y = create_obstacles_mpc(obstacles_option, pos_x, pos_y)
    
    if (occupied_positions_x.size != 0):
        shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_v2(global_path_x,global_path_y,occupied_positions_x,occupied_positions_y)
    
        shifted_feasiblebubbles_x = []
        shifted_feasiblebubbles_y = []
        for i in range (0, len(shifted_midpoints_x)):
                shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
                shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))


    if (occupied_positions_x.size != 0):
        plt.plot(occupied_positions_x,occupied_positions_y,'bo', markersize = 1)
        plt.plot(shifted_midpoints_x,shifted_midpoints_y, 'gx', markersize = 5)
        plt.plot(shifted_feasiblebubbles_x,shifted_feasiblebubbles_y, 'ro', markersize = 0.5)
        
    plt.plot(pos_x, pos_y, 'bx', markersize = 10) 
    plt.plot(global_path_x,global_path_y,'g-' )
    plt.plot()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.01)



                
# fig = plt.figure()
# ax = plt.subplot(1, 1, 1)
# ax.set_xlim([xlim_min,xlim_max])
# ax.set_ylim([ylim_min,ylim_max])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')

# for i in range(0,len(path_x)):
#     pos_x = path_x[i]
#     pos_y = path_y[i]
#     global_path_x, global_path_y = create_global_path_mpc(path_option,pos_x,pos_y)
#     occupied_positions_x, occupied_positions_y = create_obstacles_mpc(obstacles_option, pos_x, pos_y)
    
#     if (occupied_positions_x.size != 0):
#         shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_v2(global_path_x,global_path_y,occupied_positions_x,occupied_positions_y)
    
#         shifted_feasiblebubbles_x = []
#         shifted_feasiblebubbles_y = []
#         for i in range (0, len(shifted_midpoints_x)):
#                 shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
#                 shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))


#     if (occupied_positions_x.size != 0):
#         ax.plot(occupied_positions_x,occupied_positions_y,'bo', markersize = 1)
#         ax.plot(shifted_midpoints_x,shifted_midpoints_y, 'gx', markersize = 5)
#         ax.plot(shifted_feasiblebubbles_x,shifted_feasiblebubbles_y, 'ro', markersize = 0.5)
        
#     ax.plot(pos_x, pos_y, 'bx', markersize = 10) 
#     ax.plot(global_path_x,global_path_y,'g-' )







