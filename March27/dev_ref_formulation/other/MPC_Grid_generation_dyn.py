import numpy as np
from scipy import interpolate
# from rockit import *
from scipy import spatial
# from random import seed
# from random import randint


def create_obstacles_mpc(obstacles_option,pos_x,pos_y,obs_horizon,dyn_obs_x,dyn_obs_y):
    

    if obstacles_option == 4:  #feasible corner
    
        # seed(1)
        
    	# center1x = randint(-1, 9)
        # center1y = randint(-1, 9)
        
        ts =  np.linspace(-np.pi, np.pi, 100) 
        
        centerx = 1
        centery = 5
        radius  = 1.2
        
        occupied_positions_x =  centerx + radius*np.cos(ts)
        occupied_positions_y =  centery + radius*np.sin(ts)
        
        centerx = 2
        centery = 2
        radius  = 1
        
        occupied_positions_x =  np.concatenate((occupied_positions_x, centerx + radius*np.cos(ts)))
        occupied_positions_y =  np.concatenate((occupied_positions_y, centery + radius*np.sin(ts)))
        
        centerx = dyn_obs_x
        centery = dyn_obs_y
        radius  = 0.5
        
        occupied_positions_x =  np.concatenate((occupied_positions_x, centerx + radius*np.cos(ts)))
        occupied_positions_y =  np.concatenate((occupied_positions_y, centery + radius*np.sin(ts)))
        
        
    #get all the obstacle points within a certain range of the position of the robot, also no obstacles should be behind eachother (like real data)

    
    obstacles = np.array([occupied_positions_x, occupied_positions_y])
    indx = np.where( (obstacles[0] - pos_x)**2 + (obstacles[1] - pos_y)**2 <= obs_horizon )
            
    occupied_positions_in_range_x = occupied_positions_x[indx]
    occupied_positions_in_range_y = occupied_positions_y[indx]
    
    
    
    return occupied_positions_in_range_x , occupied_positions_in_range_y



def create_global_path_mpc(path_option,pos_x,pos_y,path_horizon, N):
    
 
    if path_option == 4:  #circle

        ts =  np.linspace(np.pi, 0, 500) #for creating circles points
        
        center_x = 5
        center_y = 0
        radius   = 5
        path_x = center_x + radius*np.cos(ts)
        path_y = center_y + radius*np.sin(ts)
        
        Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 0)
        u = np.linspace(0,1,500)
        global_path = interpolate.splev(u, Bspline_obj)
        path_x = np.array(global_path[0])
        path_y = np.array(global_path[1])
        path   = np.array([global_path[0], global_path[1]]).T
        
        #-------------------- find closest path point to position ------------
        
        position = np.array([pos_x,pos_y])
        tree = spatial.KDTree(path)
        idxs = tree.query(position, 2)
        nearest_index = idxs[1][1]
         
        new_path_x = path_x[nearest_index:len(path_x)]
        new_path_y = path_y[nearest_index:len(path_x)]
        
        new_global_path = np.array([new_path_x, new_path_y])
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2 + (new_global_path[1] - pos_y)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]

       
                 
                 
            

       
    return new_path_x_2, new_path_y_2, Bspline_obj
        
        
        
        
