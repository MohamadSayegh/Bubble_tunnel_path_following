"""
Feb 26 2021

"""



import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

def generate_ellipses(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
    acceptable_radius    = 0.5
    index                = 0
    
    path_length = len(global_path_x);   
    
    #initialization of arrays
    points_x                    = []
    points_y                    = []
    point                       = []
    cx                          = []
    cy                          = []
    shifted_radii               = []
    
    occ = np.array([occupied_positions_x,occupied_positions_y]).T
    tree = spatial.KDTree(occ)
    
    edge_point = False
        
    while (index < path_length): #iterate on all points of the path
    
    
        if (index == 1 or index == path_length-1): edge_point = True

        point = np.array([global_path_x[index],global_path_y[index]])   #point on the path
             
        #--------------- for choosing the first ellipse radius --------------------------
        
        idxs = tree.query(point, 2)
        nearest_index = idxs[1][1]
        nearest_point = occ[nearest_index]
        radius = np.sqrt(np.sum(np.square(point - nearest_point))) 
        
        #--------------- for choosing the next point on the path ----------------------
        
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
    
            deltax      = 0.5*(point[0] - nearest_point [0])
            deltay      = 0.5*(point[1] - nearest_point [1])
            new_point   = point
            
            for ss in range(0,10):
                
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
        
        points_x.append(point[0]) 
        points_y.append(point[1])
        cx.append(shifted_point[0] - point[0])
        cy.append(shifted_point[1] - point[1])
        shifted_radii.append(shifted_radius)
        
        
def generate_bubbles_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y):
    
    
    acceptable_radius    = 0.5
    index                = 0
    
    path_length = len(global_path_x);   
    
    #initialization of arrays
    points_x                    = []
    points_y                    = []
    point                       = []
    cx                          = []
    cy                          = []
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
    
            deltax      = 0.5*(point[0] - nearest_point [0])
            deltay      = 0.5*(point[1] - nearest_point [1])
            new_point   = point
            
            for ss in range(0,10):
                
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
        
        points_x.append(point[0]) 
        points_y.append(point[1])
        cx.append(shifted_point[0] - point[0])
        cy.append(shifted_point[1] - point[1])
        shifted_radii.append(shifted_radius)
 
    

    return points_x, points_y, cx, cy, shifted_radii




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
    
            deltax      = 0.5*(point[0] - nearest_point [0])
            deltay      = 0.5*(point[1] - nearest_point [1])
            new_point   = point
            
            for ss in range(0,10):
                
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
        
    return shifted_midpoints_x, shifted_midpoints_y, shifted_radii


    
  
    
def plotting_v2(global_path, occupied_positions_x, occupied_positions_y,\
                xlim_min, xlim_max, ylim_min, ylim_max,\
                shifted_midpoints_x, shifted_midpoints_y, shifted_radii):
    
    
    npoints =  500  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
        
    shifted_feasiblebubbles_x = []
    shifted_feasiblebubbles_y = []
    for i in range (0, len(shifted_midpoints_x)):
            shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
            shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))

       
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.savefig('The sfeasible Bubbles', dpi=300)
    
    
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([1,3])
    plt.ylim([8,11])
    plt.savefig('The feasible Bubbles 2', dpi=300)
 
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([2.9,6])
    plt.ylim([-0.1,3])
    plt.savefig('The feasible Bubbles 3', dpi=300)
    
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'r-')
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.legend(['original global path','Obstacles'], loc = "best")
    plt.title('Global Path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.savefig('Global Path', dpi=300)
    
    

def plotting_v2_squares(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
                xlim_min, xlim_max, ylim_min, ylim_max, midpoints_x, midpoints_y, radii):
    
    
    npoints =  500  #numbr of points of every circle
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
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.savefig('The sfeasible Bubbles', dpi=300)
    
    
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([1,3])
    plt.ylim([8,11])
    plt.savefig('The feasible Bubbles 2', dpi=300)
 
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'b-')
    plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 3)
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
    plt.legend(['original path',' Bubbles Midpoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
    plt.title('The feasible Bubbles')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([2.9,6])
    plt.ylim([-0.1,3])
    plt.savefig('The feasible Bubbles 3', dpi=300)
    
    plt.figure(dpi=300)
    plt.plot(global_path[0], global_path[1], 'r-')
    plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
    plt.legend(['original global path','Obstacles'], loc = "best")
    plt.title('Global Path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.savefig('Global Path', dpi=300)
    
    
    
def is_inside(circle_x, circle_y, radius, x, y):
    
    d = (x-circle_x)**2 + (y-circle_y)**2
    
    if (d < radius**2):
        return 1;
    else:
        return 0;
    
    
def is_inside_squares(center_x, center_y, length, x, y):

    d1 = abs(x - center_x) 
    d2 = abs(y - center_y)
    
    if (d1 < length and d2 < length):
        return 1;
    else:
        return 0;


def create_tunnel(midpoints_x,midpoints_y,radii):
    
    
    
    tunnel_x = [];
    tunnel_y = [];
    inside = 0
    minx = -2; 
    maxx = 12;
    
    
    # create feasible  bubbles
    npoints =  700  #numbr of points of every circle
    ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
        
    feasiblebubbles_x = []
    feasiblebubbles_y = []
    for i in range (0, len(midpoints_x)):
            feasiblebubbles_x.append(midpoints_x[i] + radii[i]*np.cos(ts))
            feasiblebubbles_y.append(midpoints_y[i] + radii[i]*np.sin(ts))
    


    
    for bubble_index in range(0,len(midpoints_x)):
        for point_index in range (0,npoints):
            
            pointx = feasiblebubbles_x[bubble_index][point_index]
            pointy = feasiblebubbles_y[bubble_index][point_index]
            
            for bubble_index_2 in range(0,len(midpoints_x)):
                inside = inside + is_inside(midpoints_x[bubble_index_2], midpoints_y[bubble_index_2], radii[bubble_index_2], pointx, pointy)
                
            if(pointx < minx or pointx > maxx): # only tunnel points between start and end 
                inside = inside + 1;
            
            if(inside == 0):
                tunnel_x.append(pointx)
                tunnel_y.append(pointy)
                
            inside = 0

    return tunnel_x, tunnel_y
    
  
    
  
def create_tunnel_sqaures(midpoints_x,midpoints_y,radii):
    
    
    
    tunnel_x = [];
    tunnel_y = [];
    inside = 0
    minx = -2; 
    maxx = 12;
    
        
    npoints =  500  #numbr of points of every circle
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

    
    for bubble_index in range(0,len(midpoints_x)):
        for point_index in range (0,npoints):
            
            pointx = feasiblebubbles_x[bubble_index][point_index]
            pointy = feasiblebubbles_y[bubble_index][point_index]
            
            for bubble_index_2 in range(0,len(midpoints_x)):
                inside = inside + is_inside_squares(midpoints_x[bubble_index_2], midpoints_y[bubble_index_2], radii[bubble_index_2], pointx, pointy)
                
            if(pointx < minx or pointx > maxx): # only tunnel points between start and end 
                inside = inside + 1;
            
            if(inside == 0):
                tunnel_x.append(pointx)
                tunnel_y.append(pointy)
                
            inside = 0

    return tunnel_x, tunnel_y    
  

        
    
  
    
