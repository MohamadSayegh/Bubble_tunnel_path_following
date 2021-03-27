


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



#------------- use sqaures or circles as bubbles ?
use_squares = False
# use_squares = True


# option 2,2 works with N = 5 T = 10
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

   

obs_horizon       = 10
path_horizon      = 2      # less than 3 causes problems (jumps overs the original path)

Nsim    = 100             #max allowed iterations   
N       = 5
dt      = 2               #reducing less than 2 gives constraint violations



#------------- Initialize OCP

ocp = Ocp(T = N*dt)       



#---------------- Initialize grid, occupied positions and bubbles



while initial_pos_x < 8 or initial_pos_y < 8:
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon, N)
    
    midpointx, midpointy, radiusx, radiusy = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    
    npoints =  500  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
        
    ellipse_x =  midpointx + radiusx*cos(ts) 
    ellipse_y =  midpointy + radiusy*sin(ts)  
    
    
    plt.figure(dpi=300)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.plot(global_path_x, global_path_y, 'g--')
    plt.plot(midpointx, midpointy,'bx')
    plt.plot(ellipse_x,ellipse_y,'b.', markersize = 1)
    plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
    plt.legend(['original path','Obstacles', 'Feasible tunnel', 'OCP solution'])
    plt.title('OCP Solutin with given path and tunnel')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig('OCP Solution', dpi=300)
    plt.pause(0.01)
    
    initial_pos_x = global_path_x[10]
    initial_pos_y = global_path_y[10]
    
