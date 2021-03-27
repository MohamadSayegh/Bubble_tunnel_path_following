
"""
December 11 2020 

"""



import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import math  
from numpy import cos, sin, pi
from scipy import interpolate
  


def generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
    if (occupied_positions_x.size != 0): #if there are obstacles
        
        index  = 0
        path_length = len(global_path_x)
        # edge_point  = False
        
        midpoints_x = []
        midpoints_y = []
        radii_x     = []
        radii_y     = []
        

        while (index < path_length): #iterate on all points of the path
        

            # if (index == 1 or index == path_length-1): edge_point = True
            
            occ   = np.array([occupied_positions_x,occupied_positions_y]).T
            tree  = spatial.KDTree(occ)
    
            point = np.array([global_path_x[index],global_path_y[index]])   #point on the path
                            
            
            #--------------- for choosing the bubble radius ----------------------------------------------
            
    
            idxs = tree.query(point, 2)
            nearest_index = idxs[1][1]
            nearest_point = occ[nearest_index]
            radius = 0.9*np.sqrt(np.sum(np.square(point - nearest_point))) 
    
            
            
            if abs(nearest_point[0] - point[0]) < 0.2:
                long_axis_y = False
                
            elif abs(nearest_point[1] - point[1]) < 0.2:
                long_axis_y = True         
            else:
                long_axis_y = True
                
    
            #----------------- Ellipse second radius -----------------------------------------
    
            radius1 = radius
            radius2 = radius
            rad     = radius
            
            
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
                    if rad > 3*radius:
                        break
                    else:      
                        radius2 = rad
                            
            if long_axis_y == True:            
                radiusx = radius1
                radiusy = radius2
            else:
                radiusx = radius2
                radiusy = radius1
                
                
            #--------------- for choosing the next point on the path -------------------------
        
            indexp = index
            new_point_inside_bubble = True
            distance = 0
            while new_point_inside_bubble:
                indexp = indexp + 1
                
                if (indexp >= path_length):
                    index = path_length
                    break
                
                new_midpoint = np.array([global_path_x[indexp],global_path_y[indexp]])   #point on the path
                
                distance = (np.sum(np.square(point - new_midpoint)))
                         
                if long_axis_y == True:  
                    if distance >= 0.5*radiusy**2:
                        new_point_inside_bubble = False
                        index = indexp
                else:
                    if distance >= 0.5*radiusx**2:
                        new_point_inside_bubble = False
                        index = indexp
                    
                    
            
            #------------------ append data -----------------------------------------------
            
            midpoints_x.append(point[0])
            midpoints_y.append(point[1])
            radii_x.append(radiusx)
            radii_y.append(radiusy)
            
        
        return midpoints_x, midpoints_y, radii_x, radii_y

        
    else: #no obstacles
    
        
        midpoints_x = global_path_x
        midpoints_y = global_path_y
        radii_x     = np.linspace(2,3,len(global_path_x))
        radii_y     = np.linspace(2,3,len(global_path_y))
                                                                 
        return midpoints_x, midpoints_y, radii_x, radii_y
            
            



def generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
    if (occupied_positions_x.size != 0): #if there are obstacles
        
        acceptable_radius    = 1
        index                = 0
        
        path_length = len(global_path_x);   
        
        #initialization of arrays
        point                       = []
        shifted_midpoints_x         = []
        shifted_midpoints_y         = []
        shifted_radii               = []
        
        occ = np.array([occupied_positions_x,occupied_positions_y]).T
        tree = spatial.KDTree(occ)
        
        edge_point = False
            
        while (index < path_length): #iterate on all points of the path
        
        
            if (index == 1 or index == path_length-1): edge_point = True
    
            point = np.array([global_path_x[index],global_path_y[index]])   #point on the path
                            
            #--------------- for choosing the bubble radius ----------------------------------------------
            
           
            idxs = tree.query(point, 2)
            nearest_index = idxs[1][1]
            nearest_point = occ[nearest_index]
            radius = np.sqrt(np.sum(np.square(point - nearest_point))) 
            radius = 0.8*radius
            
            #--------------- for choosing the next point on the path -------------------------
            
            indexp = index
            new_point_inside_bubble = True
            distance = 0
            while new_point_inside_bubble:
                indexp = indexp + 1
                if (indexp >= path_length):
                    index = path_length
                    break
                new_midpoint = np.array([global_path_x[indexp],global_path_y[indexp]])   #point on the path
                
                distance = (np.sum(np.square(point - new_midpoint)))
                         
        
                if distance >= radius**2:
                    new_point_inside_bubble = False
                    index = indexp
                              
            
            #---------------------- for shifting the midpoints -------------------------------
            shifted_radius = radius
            shifted_point = point
            
            new_radius = []
            new_radius.append(radius)
            
            if (radius < acceptable_radius) and (not edge_point):
        
                deltax      = 0.2*(point[0] - nearest_point [0])
                deltay      = 0.2*(point[1] - nearest_point [1])
                new_point   = point
                
                for ss in range(0,5):
                    
                    new_rad = 0
                    
                    new_point   = np.array( [new_point[0] + deltax , new_point[1] + deltay ])
                    
                    idxs2 = tree.query(new_point, 2)
                    nearest_index2 = idxs2[1][1]
                    nearest_point2 = occ[nearest_index2]
            
                    new_rad = np.sqrt(np.sum(np.square(new_point - nearest_point2))) 
                    
                    if new_rad >= new_radius[-1]:
                        new_radius.append(new_rad)
                        shifted_radius = new_radius[-1]
                        shifted_point  = new_point
                        if shifted_radius > acceptable_radius:
                            break
                                                  
            shifted_midpoints_x.append(shifted_point[0]) #the point becomes the midpoint of the bubble
            shifted_midpoints_y.append(shifted_point[1])
            shifted_radii.append(shifted_radius)
            
    else: #no obstacles
        
        shifted_midpoints_x = global_path_x
        shifted_midpoints_y = global_path_y
        shifted_radii = np.linspace(2,3,len(global_path_x))
            
            
    return shifted_midpoints_x, shifted_midpoints_y, shifted_radii


    
  
    
def plotting(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
                xlim_min, xlim_max, ylim_min, ylim_max,\
                shifted_midpoints_x, shifted_midpoints_y, shifted_radii):
    
    
    npoints =  500  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
        
    shifted_feasiblebubbles_x = []
    shifted_feasiblebubbles_y = []
    for i in range (0, len(shifted_midpoints_x)):
            shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
            shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))

       
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path','shifted Midpoints', 'Occupied Positions', 'shifted Feasible Bubbles'])
    plt.title('The shifted feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path','shifted Midpoints', 'Occupied Positions', 'shifted Feasible Bubbles'])
    plt.title('The shifted feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([1,3])
    plt.ylim([8,11])
 
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path','shifted Midpoints', 'Occupied Positions', 'shifted Feasible Bubbles'])
    plt.title('The shifted feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([2.9,6])
    plt.ylim([-0.1,3])
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'r-')
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.legend(['original global path','Occupied Positions'])
    plt.title('Global Path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
  
    
  

def get_bubbles_mpc_loop(global_path_x,global_path_y, x_sol_ref, y_sol_ref,\
                         occupied_positions_x, occupied_positions_y,\
                         xlim_min, xlim_max, ylim_min, ylim_max, midpoints_x,\
                         midpoints_y, radii, use_squares,\
                         x_hist, y_hist, x_sol, y_sol,i):
    
    npoints =  100  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints)
    
    if use_squares == False:
        shifted_feasiblebubbles_x = []
        shifted_feasiblebubbles_y = []
        for k in range (0, len(midpoints_x)):
                shifted_feasiblebubbles_x.append(midpoints_x[k] + radii[k]*np.cos(ts))
                shifted_feasiblebubbles_y.append(midpoints_y[k] + radii[k]*np.sin(ts))
    

        return shifted_feasiblebubbles_x, shifted_feasiblebubbles_y

    if use_squares == True:
            
        npoints =  200
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
                plt.figure(dpi=300)
                
        return feasiblebubbles_x, feasiblebubbles_y
                
    


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
    Bspline_obj, u = interpolate.splprep([global_path_x[0:index-1],global_path_y[0:index-1]], u = None, s = 0)
    u = np.linspace(0,1,N)
    global_path = interpolate.splev(u, Bspline_obj)
    global_path_x_new = np.array(global_path[0])
    global_path_y_new = np.array(global_path[1])
    
    return global_path_x_new, global_path_y_new, index
        


    
  
    
  
    
  