
import numpy as np
from scipy import interpolate
# from rockit import *
from scipy import spatial
# from random import seed
# from random import randint


def create_obstacles_mpc(obstacles_option,pos_x,pos_y,obs_horizon):
    
    # pos_x and pos_y are th positions of the robot at every time the function is called
    
    if obstacles_option == 1:  #feasible corner

        #add obstacles (global) obstacles as lines
        
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
        

        
    
    elif obstacles_option == 2:  #feasible corner


        #line at y = 10
        occupied_positions_x = np.linspace(0,9,200)
        occupied_positions_y = 10*np.ones(200)
        
        #line at x = 3
        occupied_positions_x = np.concatenate((occupied_positions_x, 3*np.ones(100)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,8,100)))
        
        #line at y = 8
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,9,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 8*np.ones(100)))
        
        ts =  np.linspace(-np.pi, np.pi, 100) 
        
        centerx = 2
        centery = 2
        radius  = 1
        
        occupied_positions_x =  np.concatenate((occupied_positions_x, centerx + radius*np.cos(ts)))
        occupied_positions_y =  np.concatenate((occupied_positions_y, centery + radius*np.sin(ts)))
        
        centerx = 0.9
        centery = 8
        radius  = 0.9
        
        occupied_positions_x =  np.concatenate((occupied_positions_x, centerx + radius*np.cos(ts)))
        occupied_positions_y =  np.concatenate((occupied_positions_y, centery + radius*np.sin(ts)))
        
        
        
    
    elif obstacles_option == 3:  #feasible corner


        #line at x = 3
        occupied_positions_x =  3*np.ones(50)
        occupied_positions_y =  np.linspace(0,7,50)

        #line at x = 6
        occupied_positions_x = np.concatenate((occupied_positions_x, 4.6*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(0,7,50)))
        
        #line at y = 0
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,4.6,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 0*np.ones(100)))

        #line at y = 7
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(4.6,10,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 7*np.ones(100)))
        
        #line at y = 7
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(2,3,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 7*np.ones(100)))
        
        #line at y = 10
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(0,10,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 10*np.ones(100)))
        
        
        
    elif obstacles_option == 4:  #feasible corner
    
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
        
        centerx = 6
        centery = 3
        radius  = 1.5
        
        occupied_positions_x =  np.concatenate((occupied_positions_x, centerx + radius*np.cos(ts)))
        occupied_positions_y =  np.concatenate((occupied_positions_y, centery + radius*np.sin(ts)))
        
        #line at y = 4
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3.5,4.5,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 4.5*np.ones(100)))
        
        #line at y = 3
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3.5,4.5,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 3.5*np.ones(100)))
        
        #line at x = 3
        occupied_positions_x = np.concatenate((occupied_positions_x, 3*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(3.5,4.5,50)))
        
        #line at x = 4
        occupied_positions_x = np.concatenate((occupied_positions_x, 4.5*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(3.5,4.5,50)))
        
        
        #line at y = 4
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,4,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 4*np.ones(100)))
        
        #line at y = 3
        occupied_positions_x = np.concatenate((occupied_positions_x,   np.linspace(3,4,100)))
        occupied_positions_y = np.concatenate((occupied_positions_y, 3*np.ones(100)))
        
        #line at x = 3
        occupied_positions_x = np.concatenate((occupied_positions_x, 3*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(3,4,50)))
        
        #line at x = 4
        occupied_positions_x = np.concatenate((occupied_positions_x, 4*np.ones(50)))
        occupied_positions_y = np.concatenate((occupied_positions_y,   np.linspace(3,4,50)))
        
        
        

  


    
        
        
        
    
    #get all the obstacle points within a certain range of the position of the robot, also no obstacles should be behind eachother (like real data)

    
    obstacles = np.array([occupied_positions_x, occupied_positions_y])
    indx = np.where( (obstacles[0] - pos_x)**2 + (obstacles[1] - pos_y)**2 <= obs_horizon )
            
    occupied_positions_in_range_x = occupied_positions_x[indx]
    occupied_positions_in_range_y = occupied_positions_y[indx]
    
    
    
    return occupied_positions_in_range_x , occupied_positions_in_range_y



def create_global_path_mpc(path_option,pos_x,pos_y,path_horizon):
    
    if path_option == 1:


        path_x = np.linspace(0, 9, 37)

        path_y = [ 0.    ,  2.0   , 4.0   , 6.0   , 8.0   , 9.8   , 9.8   ,  9.8  ,  9.8   ,  9.8  ,  9.8   ,
                    9.8   ,  9.0   , 8.0   , 3.0   , 2.0   , 1.0   , 0.2   ,  0.2   , 0.2   ,  0.2   , 0.2   ,
                    0.2   ,  0.2   , 0.2   , 4.0   , 5.0   , 6.0   , 9.0   ,  9.0   , 9.0   ,  9.0   , 9.0   ,
                    9.0   ,  9.0   , 9.0   , 9.0 ]
        
        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
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
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]
    
     
    if path_option == 10:

        # Manually design the main points of the path
        path_x = np.linspace(0, 9, 37)
        
        path_y = [ 0.    ,  2.0   , 4.0   , 6.0   , 8.0   , 9.8   , 9.8   ,  9.8  ,  9.8   ,  9.8  ,  9.8   ,
                   9.8   ,  9.0   , 8.0   , 3.0   , 2.0   , 1.0   , 0.2   ,  0.2   , 0.2   ,  0.2   , 0.2   ,
                   0.2   ,  0.2   , 0.2   , 4.0   , 5.0   , 6.0   , 9.0   ,  9.0   , 9.0   ,  9.0   , 9.0   ,
                   9.0   ,  9.0   , 9.0   , 9.0 ]
        
        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
        Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 0)
        u = np.linspace(0,1,200)
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
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]
        
        new_path_x_2 = np.append(pos_x, new_path_x_2)
        new_path_y_2 = np.append(pos_y, new_path_y_2)
    
        
        Bspline_obj, u = interpolate.splprep([new_path_x_2, new_path_y_2], u = None, s = 0)
        u = np.linspace(0,1,2*len(new_path_x_2))
        global_path = interpolate.splev(u, Bspline_obj)
        new_path_x_2  = np.array(global_path[0])
        new_path_y_2  = np.array(global_path[1])
        
    
    elif path_option == 2:

        # Manually design the main points of the path
        path_x = np.linspace(0, 9, 10)
        
        path_y = [ 0.0, 5.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        
        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
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
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]
        
        
    elif path_option == 3:

        path_x = np.linspace(0, 9, 37)
        
        #  path_x =  array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
        #                   2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  , 5.25,
        #                   5.5 , 5.75, 6.  , 6.25, 6.5 , 6.75, 7.  , 7.25, 7.5 , 7.75, 8.  ,
        #                   8.25, 8.5 , 8.75, 9.  ])
        
        path_y = [ 0.    ,  4.0   , 8.0   , 9.0   , 9.0   , 9.0   , 9.0   ,  9.0  ,  9.0   ,  9.0  ,  9.0   ,
                   9.0   ,  8.5   , 2.0   , 1.0   , 1.0   , 1.0   , 2.0   ,  4.5   ,  7.5   ,  7.5   , 7.5   ,
                   7.5   ,  7.5   , 7.5   , 7.5   , 7.5   , 7.5   , 7.5   ,  7.5   ,  7.5   , 7.5   , 7.5   ,  
                   7.5   ,  7.5   , 8.0   , 9.0   ]
        
        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
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
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]
    
    elif path_option == 4:  #circle

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
        
        
    if path_option == 5:


        path_x = np.linspace(0, 9, 6)
        path_y = [0., 3., 9., 1., 5., 9.]

        # Interpolate the path using a spline
        # this interpolation will now be considered the 'original global path'
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
        
        indxs2  = np.where( ( (new_global_path[0] - pos_x)**2) < path_horizon**2 )
        
        new_path_x_2 = new_path_x[indxs2]
        new_path_y_2 = new_path_y[indxs2]

       
                 
                  
    return new_path_x_2, new_path_y_2, Bspline_obj
        
        
