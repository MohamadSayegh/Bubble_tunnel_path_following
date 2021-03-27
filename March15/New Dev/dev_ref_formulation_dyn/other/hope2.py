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


def is_inside_rectangle(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]):
      return 1
   else:
      return 0
       


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


while initial_pos_x < 8 or initial_pos_y < 8 :
   
    
    obs_horizon       = 30
    path_horizon      = 2      # less than 3 causes problems (jumps overs the original path)
     
    N       = 5
    
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon, N)
      
    initial_pos_x = global_path_x[10]
    initial_pos_y = global_path_y[10]
    
    npoints =  500  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
    
    
    if (occupied_positions_x.size != 0): #if there are obstacles
        
        acceptable_radius    = 1
        
        occ   = np.array([occupied_positions_x,occupied_positions_y]).T
        tree  = spatial.KDTree(occ)
    
        point = np.array([global_path_x[0],global_path_y[0]])   #point on the path
                        
        idxs = tree.query(point, 2)
        nearest_index = idxs[1][1]
        nearest_point = occ[nearest_index]
        radius1 = np.sqrt(np.sum(np.square(point - nearest_point))) 
        radius1 = 0.9*radius1
        
        
        bl_set = []
        tr_set = []
        
        dx = 0.1
        dy = 0.1
        
        bl_x  = point[0] - dx 
        bl_y  = point[1] - dy
        bl    = np.array([bl_x,bl_y])
        
        tr_x  = point[0] + dx 
        tr_y  = point[1] + dy
        tr    = np.array([tr_x,bl_y])
        
        
        for i in range(1,10):
            
            bl_x = bl_x - dx 
            bl_y = bl_y - dy
            bl   = np.array([bl_x,bl_y])
            
            for j in range(1,10):
                
                tr_x  = tr_x + dx 
                tr_y  = tr_y + dy
                tr    = np.array([tr_x,bl_y])
                    
                is_inside = 0
                
                for i in range(0, len(occupied_positions_x)):
                    
                    ox = occupied_positions_x[i]
                    oy = occupied_positions_y[i]
                    p  = np.array([ox,oy])
                    
                    is_inside = is_inside +  is_inside_rectangle(bl, tr, p) 
                    
                if is_inside == 0: 
                    bl_set.append(bl)
                    tr_set.append(tr)
                
                    plt.figure(dpi=300)
                    plt.xlabel('x [m]')
                    plt.ylabel('y [m]')
                    plt.xlim([-1,12])
                    plt.ylim([-1,12])
                    plt.plot(global_path_x, global_path_y, 'g--')
                    plt.plot(point[0], point[1],'kx')
                    plt.plot(bl_x,bl_y,'rx')
                    plt.plot(tr_x,tr_y,'bx')
                    plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
                    plt.pause(0.01)
                            

            







