"""
December 11 2020 

"""



import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


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
            radius = 0.9*radius
            
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
  
    
  
    
  
    
  
    
  
        
        
        
    
    
