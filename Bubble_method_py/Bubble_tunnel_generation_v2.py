"""
December 3 2020 

modifications on previous method: 
    
    1) local planner will prevent from crossing walls = no need to account for it
    
    2) only take what we need: radii and midpoints of shifted bubbles
    
    3) take x,y instead of path parameter s

after these changes new mthod v2 has the smallet execution time: 
    
time needed for generating of bubbles / new method / in seconds:  2.5656816959381104
time needed for generating of bubbles / new method v2 / in seconds:  0.4679415225982666
time needed for generating of bubbles / old method / in seconds:  0.8760976791381836

    4) use kdtree instead to improve distance computations
       side effect = distance value have some errors

after using kd-tree 

time needed for generating of bubbles / new method / in seconds:  2.4924418926239014
time needed for generating of bubbles / new method v2 / in seconds:  0.07679271697998047
time needed for generating of bubbles / old method / in seconds:  0.894556999206543


one important modifications: the global path has to be refined (200 points now)
for the bubbles to be connecting (bcz using index instead of path parameter)
    
"""



import numpy as np
from scipy import interpolate
from scipy import spatial



def generate_bubbles_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    

    obstacles_iteration_limit  = len(occupied_positions_x)
    
    x                    = 0  #initial
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
        
    while (index < path_length): #iterate on all points of the path

        distance_obs = []
        point = np.array([global_path_x[index],global_path_y[index]])   #point on the path
                  
        
        #--------------- for choosing the bubble radius ----------------------------------------------
        
       
        idxs = tree.query(point, 2)
        nearest_index = idxs[1][1]
        nearest_point = occ[nearest_index]
        radius = np.sqrt(np.sum(np.square(point - nearest_point))) 
        
        
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
        distance_obs_2 = []
        if radius < acceptable_radius:
    
            deltax      = (point[0] - nearest_point [0])
            deltay      = (point[1] - nearest_point [1])
            new_point   = np.array( [point[0] + deltax , point[1] + deltay ])
   
            idxs2 = tree.query(new_point, 2)
            nearest_index2 = idxs2[1][1]
            nearest_point2 = occ[nearest_index2]
    
            new_radius = np.sqrt(np.sum(np.square(new_point - nearest_point2)))                
            
            if new_radius >= radius:
                shifted_radius = new_radius
                shifted_point  = new_point
                
                            
        shifted_midpoints_x.append(shifted_point[0]) #the point becomes the midpoint of the bubble
        shifted_midpoints_y.append(shifted_point[1])
        shifted_radii.append(shifted_radius)
        
    return shifted_midpoints_x, shifted_midpoints_y, shifted_radii


    
  
    
  
    
  
    
  
    
  
    
  
        
        
        
    
    
