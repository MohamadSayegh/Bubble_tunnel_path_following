import numpy as np
from scipy.spatial.distance import cdist
from scipy import interpolate


# def defineDiscreteBubbles(tck,occupied_positions):
    
    
#     radii = []
#     closest_points = []
#     ts = np.linspace(0, 2*np.pi, 100)
#     feasiblebubbles_x = []
#     feasiblebubbles_y = []
#     midpoints_x = []
#     midpoints_y = []
#     midpoint_x = 0.0
#     midpoint_y = 0.0
#     max_radius = 5
#     radius = 0
    
#     s_1 = 0.0
#     s_2 = 0.01
#     point_1 = interpolate.splev(s_1, tck)
#     point_2 = interpolate.splev(s_2, tck)
#     dist = ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5
#     ref_dist = (s_2-s_1)/dist
    
#     ## Determine the closest occupied point and distance with reference to all the discrete points of the reference trajectory

#     end_of_spline_reached = False
#     s = 0.0

#     while not end_of_spline_reached:
#         if s >= 0.90:
#             s = 1.0
#             end_of_spline_reached = True
       
        
#         point = interpolate.splev(s, tck)
#         closest_point = occupied_positions[cdist([point], occupied_positions).argmin()]
#         smallest_distance = (((point[0] - closest_point[0])**2) + ((point[1] - closest_point[1])**2))**0.5

       
#         closest_points.append([closest_point[0], closest_point[0]])


#         new_radius_feasible = True
#         max_radius_reached = True
        
#         if smallest_distance < max_radius:
#             max_radius_reached = False
#             radius = smallest_distance
#             midpoint_x = point[0]
#             midpoint_y = point[1]

#             while ((new_radius_feasible) and (not max_radius_reached)):

#                 delta_x = point[0] - closest_point[0]
#                 delta_y = point[1] - closest_point[1]

#                 midpoint_x = point[0] + (1/2)*delta_x
#                 midpoint_y = point[1] + (1/2)*delta_y

#                 radius = (2.9/2.0) * smallest_distance

#                 closest_point = occupied_positions[cdist([[midpoint_x, midpoint_y]], occupied_positions).argmin()]
#                 new_smallest_distance = (((midpoint_x - closest_point[0])**2) + ((midpoint_y - closest_point[1])**2))**0.5

#                 if new_smallest_distance < radius:
#                     new_radius_feasible = False

#                 else:
#                     smallest_distance = new_smallest_distance
#                     point[0] = midpoint_x
#                     point[1] = midpoint_y
#                     if radius >= max_radius:
#                         max_radius_reached = True
        
#             if new_radius_feasible:
#                 radii.append(radius)
#                 midpoints_x.append(midpoint_x)
#                 midpoints_y.append(midpoint_y)
#                 smallest_distance = radius

#             else:
#                 radii.append(smallest_distance)
#                 midpoints_x.append(point[0])
#                 midpoints_y.append(point[1])
#                 midpoint_x = point[0]
#                 midpoint_y = point[1]
#                 radius = smallest_distance

#         else:
#             radii.append(smallest_distance)
#             midpoints_x.append(point[0])
#             midpoints_y.append(point[1])
#             midpoint_x = point[0]
#             midpoint_y = point[1]
#             radius = smallest_distance

#         feasiblebubbles_x.append(midpoint_x + radius*np.cos(ts))
#         feasiblebubbles_y.append(midpoint_y + radius*np.sin(ts))
        
#         if radius >= 4:
#             s = s + ref_dist*radius
#         elif ref_dist*radius <= 0.10:
#             s = s + ref_dist*radius
#         else:
#             s = s + 0.01
            
#     return feasiblebubbles_x, feasiblebubbles_y, radii, midpoints_x, midpoints_y



def defineDiscreteBubbles(tck,occupied_positions_x,occupied_positions_y):
        
    ## Determine the closest occupied point and distance with reference to all the discrete points of the reference trajectory
    s = 0.0
    j = 0
    k = 0
    
    distance = 0
    radii = []
    closest_points = []
    ts = np.linspace(0, 2*np.pi, 500)
    feasiblebubbles_x = []
    feasiblebubbles_y = []
    midpoints_x = []
    midpoints_y = []
    midpoint_x = 0.0
    midpoint_y = 0.0
    max_radius = 5
    radius = 0
    end_of_spline_reached = False
    
    s_1 = 0.0
    s_2 = 0.1
    point_1 = interpolate.splev(s_1, tck)
    point_2 = interpolate.splev(s_2, tck)
    dist = ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5
    ref_dist = (s_2-s_1)/dist
    
    while not end_of_spline_reached:
        #while s < 1.0: # Move over the spline until the end
        if s >= 0.90:
            s = 1.0
            end_of_spline_reached = True
        while j < len(occupied_positions_x):
            point = interpolate.splev(s, tck) 
            delta_x = point[0] - occupied_positions_x[j]
            delta_y = point[1] - occupied_positions_y[j]
            distance = (((delta_x**2) + (delta_y**2)))**0.5
            
            if j == 0:
                smallest_distance = distance
                k = j
            if distance < smallest_distance:
                smallest_distance = distance
                k = j
            j = j + 1
        
        closest_points.append([occupied_positions_x[k], occupied_positions_y[k]])
    
        if smallest_distance < max_radius:
            delta_x = point[0] - occupied_positions_x[k]
            delta_y = point[1] - occupied_positions_y[k]
    
            midpoint_x = point[0] + (1/2)*delta_x
            midpoint_y = point[1] + (1/2)*delta_y
    
            radius = (2.9/2.0) * smallest_distance
            new_radius_feasible = True
    
            j = 0
            while j < len(occupied_positions_x):
                distance = (((midpoint_x - occupied_positions_x[j])**2) + (midpoint_y - occupied_positions_y[j])**2)**0.5
                
                if distance >= radius:
                    j = j + 1
                else:
                    new_radius_feasible = False
                    j = len(occupied_positions_x)
            
            if new_radius_feasible:
                radii.append(radius)
                midpoints_x.append(midpoint_x)
                midpoints_y.append(midpoint_y)
                smallest_distance = radius
    
            else:
                radii.append(smallest_distance)
                midpoints_x.append(point[0])
                midpoints_y.append(point[1])
                midpoint_x = point[0]
                midpoint_y = point[1]
                radius = smallest_distance
    
        else:
            radii.append(smallest_distance)
            midpoints_x.append(point[0])
            midpoints_y.append(point[1])
            midpoint_x = point[0]
            midpoint_y = point[1]
            radius = smallest_distance
    
        feasiblebubbles_x.append(midpoint_x + radius*np.cos(ts))
        feasiblebubbles_y.append(midpoint_y + radius*np.sin(ts))
        if radius >= 4:
            s = s + ref_dist*radius
        elif ref_dist*radius <= 0.10:
            s = s + ref_dist*radius
        else:
            s = s + 0.10
        j = 0
        
    return feasiblebubbles_x, feasiblebubbles_y, radii, midpoints_x, midpoints_y

    
    
    
    

