
"""
march 11 2021

"""

import numpy as np
from scipy import spatial
from scipy import interpolate
  


def generate_bubbles_mpc_ellipses(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
       
    
    if (occupied_positions_x.size != 0): #if there are obstacles
        index  = 0
        path_length = len(global_path_x)
        # edge_point  = False
        acceptable_radius  = 0.5
        
        midpoints_x = []
        midpoints_y = []
        radii_x     = []
        radii_y     = []
        
        point = np.array([global_path_x[index],global_path_y[index]]) 
        
    
        while (index < path_length): #iterate on all points of the path
        
    
            # if (index == 1 or index == path_length-1): edge_point = True
            
            occ   = np.array([occupied_positions_x,occupied_positions_y]).T
            tree  = spatial.KDTree(occ)
    
            old_point = point
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
            
            if radius < acceptable_radius:
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
            else:
                radiusx = radius
                radiusy = radius
                
                
                
            #--------------- for choosing the next point on the path -------------------------
        
            indexp = index
            new_point_inside_bubble = True
            distance = 0
            while new_point_inside_bubble:
                indexp = indexp + 1
                
                if (indexp >= path_length):
                    index = path_length
                    break
                
                new_midpoint = np.array([global_path_x[indexp],global_path_y[indexp]])  
                
                distance = (np.sum(np.square(point - new_midpoint)))
                         
                if long_axis_y == True:  
                    if distance >= 0.8*radiusy**2:
                        new_point_inside_bubble = False
                        index = indexp
                else:
                    if distance >= 0.8*radiusx**2:
                        new_point_inside_bubble = False
                        index = indexp
                    
                    
            #---------------------- for shifting the midpoints -------------------------------
            shifted_radius = radiusx
            
            new_radius = []
            new_radius.append(radiusx)
            
            
            if (radiusx < acceptable_radius) and (radiusy < acceptable_radius):
        
                deltax      = 0.5*(point[0] - nearest_point [0])
                deltay      = 0.5*(point[1] - nearest_point [1])
                new_point   = point
                
                
                for ss in range(0,10):
                    
                    new_rad = 0
                    
                    new_point   = np.array( [new_point[0] + deltax , new_point[1] + deltay ])
                  
                    inside_line = check_inside_line(occupied_positions_x, occupied_positions_y, old_point, new_point)
                            
                    if inside_line == False:
                        
                        idxs2 = tree.query(new_point, 2)
                        nearest_index2 = idxs2[1][1]
                        nearest_point2 = occ[nearest_index2]
                
                        new_rad = np.sqrt(np.sum(np.square(new_point - nearest_point2))) 
                    
                    if new_rad >= new_radius[-1]:
                        new_radius.append(new_rad)
                        radiusx = new_radius[-1]
                        if radiusx > radiusy :
                            radiusy = radiusx
                        point  = new_point
                        if shifted_radius > acceptable_radius:
                            break
            
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
    
    
    
    


def generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
    if (occupied_positions_x.size != 0): #if there are obstacles
        
        acceptable_radius    = 1
        index                = 0
        
        path_length = len(global_path_x);   
        
        #initialization of arrays
        point = np.array([global_path_x[index],global_path_y[index]]) 
        
        shifted_midpoints_x         = []
        shifted_midpoints_y         = []
        shifted_radii               = []
        
        occ = np.array([occupied_positions_x,occupied_positions_y]).T
        tree = spatial.KDTree(occ)
        
        edge_point = False
            
        while (index < path_length): #iterate on all points of the path
        
        
            if (index == 1 or index == path_length-1): edge_point = True
            
            old_point = point
    
            point = np.array([global_path_x[index],global_path_y[index]])   #point on the path
                            
            #--------------- for choosing the bubble radius ----------------------------------------------
            
           
            idxs = tree.query(point, 2)
            nearest_index = idxs[1][1]
            nearest_point = occ[nearest_index]
            radius = np.sqrt(np.sum(np.square(point - nearest_point))) 
            radius = 0.99*radius
            
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
        
                deltax      = 0.3*(point[0] - nearest_point [0])
                deltay      = 0.3*(point[1] - nearest_point [1])
                new_point   = point
                
                
                for ss in range(0,10):
                    
                    new_rad = 0
                    
                    new_point   = np.array( [new_point[0] + deltax , new_point[1] + deltay ])
                  
                    inside_line = check_inside_line(occupied_positions_x, occupied_positions_y, old_point, new_point)
                            
                    if inside_line == False:
                        
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
        


def check_inside_line(occupied_positions_x, occupied_positions_y, point, new_point):
    
    inside_line = False
    corner_new_point_x = new_point[0]
    corner_new_point_y = new_point[1] 
    corner_point_x = point[0] 
    corner_point_y = point[1]
        
    for i in range(0, len(occupied_positions_x)):
        if occupied_positions_x[i] >= corner_point_x and occupied_positions_x[i] <= corner_new_point_x:
            if occupied_positions_y[i] <= corner_point_y and occupied_positions_y[i] >= corner_new_point_y:
                inside_line = True  
            elif occupied_positions_y[i] >= corner_point_y and occupied_positions_y[i] <= corner_new_point_y:
                inside_line = True  
        if occupied_positions_x[i] <= corner_point_x and occupied_positions_x[i] >= corner_new_point_x:
            if occupied_positions_y[i] <= corner_point_y and occupied_positions_y[i] >= corner_new_point_y:
                inside_line = True  
            elif occupied_positions_y[i] >= corner_point_y and occupied_positions_y[i] <= corner_new_point_y:
                inside_line = True  

            
    return inside_line
    
  
    
  
    
  