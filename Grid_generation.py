import numpy as np
from scipy import interpolate


def create_obstacles(obstacles_option):
    
    if obstacles_option == 1:  #feasible corner

        #add obstacles as lines
        
        #line at y = 10
        occupied_positions_x = np.linspace(0,9,200)
        occupied_positions_y = 10*np.ones(200)
        
        #line at x = 3
        occupied_positions_x = np.concatenate((occupied_positions_x, 3*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,8,50)))
        
        #line at x = 7
        occupied_positions_x = np.concatenate((occupied_positions_x, 7*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,8,50)))
        
        #line at y = 0
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,7,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 0*np.ones(100)))
        
        #line at y = 8
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(7,9,50)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 8*np.ones(50)))
        
        

    if obstacles_option == 2: #infeasible corner first type

        #add obstacles as lines
        
        #line at y = 10
        occupied_positions_x = np.linspace(0,9,200)
        occupied_positions_y = 10*np.ones(200)
        
        #line at x = 3
        occupied_positions_x = np.concatenate((occupied_positions_x, 3*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,8,50)))
        
        #line at x = 7
        occupied_positions_x = np.concatenate((occupied_positions_x, 7*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,8,50)))
        
        #line at y = 0
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,7,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 0*np.ones(100)))
        
        #line at y = 8
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(7,9,50)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 8*np.ones(50)))
        
        #line at y = 4
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(4,7,50)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 4*np.ones(50)))
        
    
    return occupied_positions_x, occupied_positions_y

def create_global_path(path_option):
    
    if path_option == 1:

        # Manually design the main points of the path
        path_x = np.linspace(0, 9, 20)
        path_y = [ 0.0, 5.0, 10.0, 11.0, 11.0, 11.0, 11.0, 6.0, 0.1, 0.1,
                   0.1, 0.1, 1.0, 1.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0 ]
        
        
        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
        Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 1)
        u = np.linspace(0,1,100)
        global_path = interpolate.splev(u, Bspline_obj)
        
        
        return Bspline_obj, global_path
        
        
        
        
