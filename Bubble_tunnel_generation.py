import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *


def generate_bubbles(global_path,Bspline_obj,occupied_positions_x,occupied_positions_y):
    
    #----------------------------------------------------------------------------#
    #                           Creating the Bubbles                             #
    #----------------------------------------------------------------------------#
    
    npoints =  500  #numbr of points of every circle
    ts      =  np.linspace(0, 2*pi, npoints) #for creating circles points
    
    
    obstacles_iteration_limit  = len(occupied_positions_x)
    
    s                    = 0  #initial
    s_step               = 0.01 #the step in path parameter
    acceptable_radius    = 1
    
    #initialization of arrays
    point                       = []
    midpoints_x                 = []
    midpoints_y                 = []
    feasiblebubbles_x           = []
    feasiblebubbles_y           = []
    radii                       = []
    shifted_midpoints_x         = []
    shifted_midpoints_y         = []
    shifted_feasiblebubbles_x   = []
    shifted_feasiblebubbles_y   = []
    shifted_radii               = []
    
    
    while(s < 1): #iterate on all points of the path
        
        midpoint_feasible = False
        distance_obs = []
        point = np.array(interpolate.splev(s, Bspline_obj)) #point on the path
        
        #--------------- for choosing the bubble radius
        for obsi in range(0, obstacles_iteration_limit): #iterate on all obstacle points
            
            obspoint = np.array([occupied_positions_x[obsi],occupied_positions_y[obsi]]) #one obstacle point
            distance_obs.append(np.sqrt(np.sum(np.square(point - obspoint))))  #eucledian distance
            
        obspoints = np.array([distance_obs, occupied_positions_x, occupied_positions_y]).T
        sorted_obspoints = obspoints[np.argsort(obspoints[:, 0])]
        
        # the minimum distance becomes the radius of the bubble 
        # can also attenuate with 0.9
        radius = 0.9*sorted_obspoints[0][0]
        
        
        #--------------- for choosing the next path parameter s
        sp = s
        new_point_inside_bubble = True
        distance = 0
        while new_point_inside_bubble:
            sp = sp + 0.001
            new_midpoint = np.array(interpolate.splev(sp, Bspline_obj))
    
            if distance >= radius:
                new_point_inside_bubble = False
                s = sp
                
            distance = np.sqrt(np.sum(np.square(point - new_midpoint)))
            
                     
        #--------------- for skipping unfeasible bubbles (behind the wall)
        while not midpoint_feasible:
            next_point = np.array(interpolate.splev(s, Bspline_obj))
            inside_x = np.where((occupied_positions_x > point[0]) & (occupied_positions_x < next_point[0]))
            
            if np.size(inside_x) != 0:
                occ_y = occupied_positions_y[inside_x]
                inside_y = np.where((occ_y > point[1]) & (occ_y < next_point[1]))
                
                if np.size(inside_y) != 0:
                    s = s + s_step
                else:
                    midpoint_feasible = True
            else: 
                midpoint_feasible = True
    
        
        #---------------- for shifting the midpoints
        shifted_radius = radius
        shifted_point = point
        distance_obs_2 = []
        if radius < acceptable_radius:
    
            obspoint    = sorted_obspoints[0][1:3] #the closest obstalce
            deltax      = (point[0] - obspoint[0])
            deltay      = (point[1] - obspoint[1])
            new_point   = np.array( [point[0] + deltax , point[1] + deltay ])
            
            for obsi in range(0, obstacles_iteration_limit): #iterate on all obstacle points
                obspoint = np.array([occupied_positions_x[obsi],occupied_positions_y[obsi]])
                distance_obs_2.append(np.sqrt(np.sum(np.square(new_point - obspoint))))   
                new_radius = np.sort(distance_obs_2)[0]
            
            if new_radius >= radius:
                shifted_radius = new_radius
                shifted_point  = new_point
    
                    
            
                    
        midpoints_x.append(point[0]) #the point becomes the midpoint of the bubble
        midpoints_y.append(point[1])
        feasiblebubbles_x.append(point[0] + radius*np.cos(ts))
        feasiblebubbles_y.append(point[1] + radius*np.sin(ts))
        radii.append(radius)
        
        shifted_midpoints_x.append(shifted_point[0]) #the point becomes the midpoint of the bubble
        shifted_midpoints_y.append(shifted_point[1])
        shifted_feasiblebubbles_x.append(shifted_point[0] + shifted_radius*np.cos(ts))
        shifted_feasiblebubbles_y.append(shifted_point[1] + shifted_radius*np.sin(ts))
        shifted_radii.append(shifted_radius)
        
    return feasiblebubbles_x, feasiblebubbles_y, shifted_feasiblebubbles_x,\
        shifted_feasiblebubbles_y, midpoints_x, midpoints_y, radii, shifted_midpoints_x,\
            shifted_midpoints_y, shifted_radii


def plotting(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
             xlim_min, xlim_max, ylim_min, ylim_max,feasiblebubbles_x, feasiblebubbles_y,\
             shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, midpoints_x, midpoints_y, radii,\
             shifted_midpoints_x, shifted_midpoints_y, shifted_radii):
           
    plt.figure()
    plt.plot(np.linspace(initial_pos_x,end_goal_x,len(radii)), radii)
    plt.plot(np.linspace(initial_pos_x,end_goal_x,len(radii)), shifted_radii)
    plt.ylabel('radii magnitude in m')
    plt.xlabel('x [m]')
    plt.legend(['Before shifting','After shifting'])
    plt.title('Radii of Bubbles before and after shifting the bubbles')
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx')
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.legend(['Global Reference Trajectory','Midpoints of the bubbles', 'Occupied Positions'])
    plt.title('The feasible bubbles midpoints and the global trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'yx', markersize= 0.5)
    plt.legend(['original path','Midpoints', 'Occupied Positions', 'Feasible Bubbles'])
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    # plt.xlim([2.9,7.1])
    # plt.ylim([-0.1,9])
    
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
    # plt.xlim([2.9,7.1])
    # plt.ylim([-0.1,9])
    
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'yx', markersize= 0.5)
    plt.legend(['original path','Midpoints', 'Occupied Positions', 'Feasible Bubbles'])
    plt.title('The feasible Bubbles when the path crosses the wall')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([0.5,3.5])
    plt.ylim([9,12])
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'yx', markersize= 0.5)
    plt.legend(['original path','Midpoints', 'Occupied Positions', 'Feasible Bubbles'])
    plt.title('The feasible Bubbles when the path is close to wall')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([3.5,5.5])
    plt.ylim([-0.5,1])
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 5)
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'gx',markersize= 5)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'yx', markersize= 0.5)
    plt.legend(['original path','Midpoints', 'Shifted Midpoints','Occupied Positions', 'Feasible Bubbles'])
    plt.title('The feasible Bubbles when the path crosses the wall')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([2,5])
    plt.ylim([-0.5,4])
    
    plt.figure()
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 5)
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'gx',markersize= 5)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.legend(['original path','Midpoints', 'Shifted Midpoints','Occupied Positions', 'Feasible Bubbles'])
    plt.title('The feasible Bubbles when the path crosses the wall')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([2.8,7])
    plt.ylim([-0.2,5])
        
        
        
    
    
