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
from scipy import spatial
import matplotlib.pyplot as plt



def generate_bubbles_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
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
        if (radius < acceptable_radius) and (not edge_point):
    
            deltax      = (point[0] - nearest_point [0])
            deltay      = (point[1] - nearest_point [1])
            new_point   = np.array( [point[0] + deltax , point[1] + deltay ])
   
            idxs2 = tree.query(new_point, 2)
            nearest_index2 = idxs2[1][1]
            nearest_point2 = occ[nearest_index2]
    
            new_radius = np.sqrt(np.sum(np.square(new_point - nearest_point2))) 
            new_radius = 0.9*new_radius               
            
            if new_radius >= radius:
                shifted_radius = new_radius
                shifted_point  = new_point
                
                            
        shifted_midpoints_x.append(shifted_point[0]) #the point becomes the midpoint of the bubble
        shifted_midpoints_y.append(shifted_point[1])
        shifted_radii.append(shifted_radius)
        
    return shifted_midpoints_x, shifted_midpoints_y, shifted_radii


    
  
    
def plotting_v2(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
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
  
    
  
    
  
    
  
    
  
        
        
        
    
    
