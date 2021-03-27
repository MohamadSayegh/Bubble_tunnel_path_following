
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import  generate_bubbles_v2




end_goal_x      =   9     # position of initial and end point
end_goal_y      =   9
initial_pos_x   =   0
initial_pos_y   =   0
xlim_min        =   -1  # xlim and ylim of plots
xlim_max        =   12
ylim_min        =   -1
ylim_max        =   12
n               =   10    # size of square grid



N       =  10             # number of shooting points of every ocp

path_horizon = 100
obs_horizon  = 100

obstacles_option = 3 
path_option = 3


global_path_x, global_path_y , Bspline_obj      =  create_global_path_mpc(path_option,0,0,N, 500)
occupied_positions_x, occupied_positions_y      =  create_obstacles_mpc(obstacles_option, 0, 0, 500)
midpoints_x, midpoints_y, radii                 =  generate_bubbles_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)

  
    
npoints =  100  #numbr of points of every circle
ts      =  np.ones(npoints)
    
feasiblebubbles_x = []
feasiblebubbles_y = []

for i in range (0, len(midpoints_x)):
    
        length = radii[i]  
        
        point = (midpoints_x[i] - length)*ts
        feasiblebubbles_x.append(point)
        line = np.linspace(midpoints_y[i] - length, midpoints_y[i] + length, npoints)
        feasiblebubbles_y.append(line)
        
        point = (midpoints_x[i] + length)*ts
        feasiblebubbles_x.append(point)
        line = np.linspace(midpoints_y[i] - length, midpoints_y[i] + length, npoints)
        feasiblebubbles_y.append(line)
        
        line = np.linspace(midpoints_x[i] - length, midpoints_x[i] + length, npoints)
        feasiblebubbles_x.append(line)
        point = (midpoints_y[i] + length)*ts
        feasiblebubbles_y.append(point)
        
        line = np.linspace(midpoints_x[i] - length, midpoints_x[i] + length, npoints)
        feasiblebubbles_x.append(line)
        point = (midpoints_y[i] - length)*ts
        feasiblebubbles_y.append(point)
        
        
        


    
plt.figure()
plt.plot(global_path_x, global_path_y, 'b-')
plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
plt.legend(['original path','Midpoints', 'Obstacles', 'Feasible squares'])
plt.title('The shifted feasible Bubbles')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


    
# plt.figure()
# plt.plot(global_path_x, global_path_y, 'b-')
# plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
# plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
# plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
# plt.legend(['original path','Midpoints', 'Obstacles', 'Feasible squares'])
# plt.title('The shifted feasible Bubbles')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([2.9,6.9])
# plt.ylim([-0.1,2.9])



