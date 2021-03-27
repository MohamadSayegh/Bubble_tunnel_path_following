import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import math  
from numpy import cos, sin, pi
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, plotting, get_bubbles_mpc_loop, generate_bubbles_mpc_v3
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import create_tunnel, plotting_v2



    
obstacles_option  = 1
path_option       = 1


global_end_goal_x       =    9     #position of initial and end point
global_end_goal_y       =    9
initial_pos_x           =    0
initial_pos_y           =    0
xlim_min                =   -2     #xlim and ylim of plotsR
xlim_max                =    12
ylim_min                =   -2
ylim_max                =    12

# ----------- draw ellipse
npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points

obs_horizon       = 40
path_horizon      = 2      # less than 3 causes problems (jumps overs the original path)
 
N       = 10
    
while initial_pos_x < 8 or initial_pos_y < 8 :
   
    

    
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)
      
    
    initial_pos_x = global_path_x[5]
    initial_pos_y = global_path_y[5]
    
    
    midpoints_x, midpoints_y, radii_x, radii_y = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
      
    ellipse_x = []
    ellipse_y = []
    for i in range(0, len(midpoints_x)):      
        ellipse_x.append(midpoints_x[i] + radii_x[i]*cos(ts) )
        ellipse_y.append(midpoints_y[i] + radii_y[i]*sin(ts) ) 
    
    plt.figure(dpi=300)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([-1,12])
    plt.ylim([-1,12])
    plt.plot(midpoints_x, midpoints_y, 'g--')
    plt.plot(ellipse_x,ellipse_y,'b.', markersize = 1)
    plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
    plt.pause(0.01)
                            

            







