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


def is_inside_ellipse( x, y, xp, yp, a, b): 
  
    is_inside = 0

    ellipse = (x-xp)**2/a**2 + (y-yp)**2/b**2
  
    if (ellipse < 1): 
        is_inside = 1
        
    return is_inside
        
def find_path(global_path_x, global_path_y, xp , yp , radiusx ,radiusy, N):
    
    index = 0
    for i in range(0, len(global_path_x)):
        e = (global_path_x[i]-xp)**2/radiusx**2 + (global_path_y[i]-yp)**2/radiusy**2
        if e > 1:
            break
        else:
            index = i
    #----------- N points of path
    Bspline_obj, u = interpolate.splprep([global_path_x[0:index],global_path_y[0:index]], u = None, s = 0)
    u = np.linspace(0,1,N)
    global_path = interpolate.splev(u, Bspline_obj)
    global_path_x_new = np.array(global_path[0])
    global_path_y_new = np.array(global_path[1])
    
    return global_path_x_new, global_path_y_new

            
    


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
                        
        
        #--------------- for choosing the bubble radius ----------------------------------------------
        

        idxs = tree.query(point, 2)
        nearest_index = idxs[1][1]
        nearest_point = occ[nearest_index]
        radius = np.sqrt(np.sum(np.square(point - nearest_point))) 

        
        
        if abs(nearest_point[0] - point[0]) < 0.2:
            long_axis_y = False
            
        elif abs(nearest_point[1] - point[1]) < 0.2:
            long_axis_y = True         
        else:
            long_axis_y = True
            
        # print(long_axis_y)
        
        #----------------- shifting midpoint 
        
        shifted_radius = radius
        shifted_point = point
        
        # new_radius = []
        # new_radius.append(radius)
        
        # if (radius < acceptable_radius):
    
        #     deltax      = 0.2*(point[0] - nearest_point [0])
        #     deltay      = 0.2*(point[1] - nearest_point [1])
        #     new_point   = point
            
        #     for ss in range(0,5):
                
        #         new_rad = 0
                
        #         new_point   = np.array( [new_point[0] + deltax , new_point[1] + deltay ])
                
        #         idxs2 = tree.query(new_point, 2)
        #         nearest_index2 = idxs2[1][1]
        #         nearest_point2 = occ[nearest_index2]
        
        #         new_rad = np.sqrt(np.sum(np.square(new_point - nearest_point2))) 
                
        #         if new_rad >= new_radius[-1]:
        #             new_radius.append(new_rad)
        #             shifted_radius = new_radius[-1]
        #             shifted_point  = new_point
        #             nearest_point  = nearest_point2
        #             if shifted_radius > acceptable_radius:
        #                 break
          
        
        #----------------- Ellipse second radius -----------------------------------------

        shifted_radius = 0.9*shifted_radius
        radius1 = shifted_radius
        radius2 = shifted_radius
        rad = shifted_radius
        point = shifted_point
        
        
        while True:
                
            rad = rad + 0.1
                                            
            is_inside = 0
            
            for i in range(0, len(occupied_positions_x)):
                
                ox = occupied_positions_x[i]
                oy = occupied_positions_y[i]
            
                if long_axis_y == True:
                    is_inside = is_inside + is_inside_ellipse( point[0], point[1], ox, oy, radius1, rad )
                else:
                    is_inside = is_inside + is_inside_ellipse( point[0], point[1], ox, oy, rad, radius1 )
             
                
            if is_inside > 0:
                # print("is_inside")
                break
            else:
                if rad > 10:
                    break
                else:      
                    radius2 = rad
                        
        if long_axis_y == True:      
            radiusx = radius1
            radiusy = radius2 
        else:
            radiusx = radius2
            radiusy = radius1 
            
            
            
        ellipse_x =  point[0] + radiusx*cos(ts) 
        ellipse_y =  point[1] + radiusy*sin(ts)  
        
        global_path_x, global_path_y = find_path(global_path_x, global_path_y, point[0] , point[1] , radiusx ,radiusy, N)    
        
        plt.figure(dpi=300)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim([-1,12])
        plt.ylim([-1,12])
        plt.plot(global_path_x, global_path_y, 'g--')
        plt.plot(point[0], point[1],'bx')
        plt.plot(ellipse_x,ellipse_y,'b.', markersize = 1)
        plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
        plt.pause(0.01)
                            

            







