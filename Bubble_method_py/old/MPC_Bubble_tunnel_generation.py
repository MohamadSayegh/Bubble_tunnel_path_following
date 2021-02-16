"""
December 8 2020 

"""

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt



def generate_bubbles_mpc(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y,N):
    
   
    if (occupied_positions_x.size != 0): #then preform bubble generation
        
        acceptable_radius    = 0.5
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
        
        indxs = []
        indxs.append(index)
        
        max_radius = 1
            
        while (index < path_length): #iterate on all points of the path

            
            #--------------- dont shift edge points -----------------------------------
            if (index == 0 or index == path_length-1): 
                edge_point = True
    
            #------------------   #point on the path ----------------------------------
            
            point = np.array([global_path_x[index],global_path_y[index]]) 
                            
            #--------------- for choosing the bubble radius -----------------------------------
            
           
            idxs = tree.query(point, 2)
            nearest_index = idxs[1][1]
            nearest_point = occ[nearest_index]
            radius = np.sqrt(np.sum(np.square(point - nearest_point))) 
            if radius > max_radius:
                radius = max_radius
                 

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
                    indxs.append(index)
            
                    
                    
        #---------------------- for shifting the midpoints -------------------------------
        shifted_radius = radius
        shifted_point = point
        
        new_radius = []
        new_radius.append(radius)
        
        if (radius < acceptable_radius) and (not edge_point):
    
            deltax      = (point[0] - nearest_point [0])
            deltay      = (point[1] - nearest_point [1])
            new_point   = point
            
            for ss in range(0,3):
                
                new_rad = 0
                
                new_point   = np.array( [new_point[0] + deltax , new_point[1] + deltay ])
                
                idxs2 = tree.query(new_point, 2)
                nearest_index2 = idxs2[1][1]
                nearest_point2 = occ[nearest_index2]
        
                new_rad = np.sqrt(np.sum(np.square(new_point - nearest_point2))) 
                print(new_radius)
                
                if new_rad >= new_radius[-1]:
                    new_radius.append(new_rad)
                    shifted_radius = new_radius[-1]
                    shifted_point  = new_point
                    if shifted_radius > acceptable_radius:
                        break
                    
        shifted_midpoints_x.append(shifted_point[0]) #the point becomes the midpoint of the bubble
        shifted_midpoints_y.append(shifted_point[1])
        shifted_radii.append(shifted_radius)
            
        # mod_global_path_x = global_path_x[indxs]
        # mod_global_path_y = global_path_y[indxs]
        

        # if  len(shifted_midpoints_x) < N:
        
        #     start = indxs[-1] + 1 
        #     end   = N - len(indxs) + start
            
        #     add_x = global_path_x[start:end]
        #     add_y = global_path_y[start:end]
        
        #     shifted_midpoints_x =   np.concatenate((shifted_midpoints_x, add_x))
        #     shifted_midpoints_y =   np.concatenate((shifted_midpoints_y, add_y))
                
        #     mod_global_path_x   =   np.concatenate((mod_global_path_x, add_x))
        #     mod_global_path_y   =   np.concatenate((mod_global_path_y, add_y))
        #     add_radii           =   np.linspace(1,2,abs(start-end))
        #     shifted_radii       =   np.concatenate(( shifted_radii, add_radii ))
                       
    #     else:
            
    #         shifted_midpoints_x =   shifted_midpoints_x[0:N]
    #         shifted_midpoints_y =   shifted_midpoints_y[0:N]
    #         shifted_radii       =   shifted_radii[0:N]
    #         mod_global_path_x   =   mod_global_path_x[0:N]
    #         mod_global_path_y   =   mod_global_path_y[0:N]
           
                       
    # else:        
    #     mod_global_path_x       =   global_path_x[0:N]
    #     mod_global_path_y       =   global_path_y[0:N]
    #     shifted_midpoints_x     =   mod_global_path_x
    #     shifted_midpoints_y     =   mod_global_path_y
    #     shifted_radii           =   np.linspace(1,2,N)
        
    return shifted_midpoints_x, shifted_midpoints_y, shifted_radii, global_path_x, global_path_y


    
  
    
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
  
    
  
    
  
    
  
    
  
        
        
        
    
    
