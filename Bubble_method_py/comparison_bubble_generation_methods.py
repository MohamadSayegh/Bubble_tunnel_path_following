# from Bubble_tunnel_generation import generate_bubbles
from Bubble_tunnel_generation_v2 import generate_bubbles_v2
from Grid_generation import create_obstacles, create_global_path
from Bubble_tunnel_generation_Mat import defineDiscreteBubbles
import time
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------#
#                         Generate Grid and Obstacles                        #
#----------------------------------------------------------------------------#

end_goal_x      =   9     # position of initial and end point
end_goal_y      =   9
initial_pos_x   =   0
initial_pos_y   =   0
xlim_min        =   -0.5  # xlim and ylim of plots
xlim_max        =   10.5
ylim_min        =   -3
ylim_max        =   14
n               =   10    # size of square grid


''' 
possible combinations of the two options below: 1/1 2/1

option 1/1 works 

option 2/1 doesnt work = its is a very hard situiation 
                       = the bubbles stop generating (side effect of no wall crossing technique)

''' 
obstacles_option = 1 
path_option = 1

occupied_positions_x, occupied_positions_y = create_obstacles(obstacles_option)
Bspline_obj, global_path = create_global_path(path_option)

#----------------------------------------------------------------------------#
#                           Creating the Bubbles                             #
#----------------------------------------------------------------------------#

start_time = time.time()

# # new method
# f_x, f_y,sx,sy,mx, my, r, smx, smy, sr \
#     = generate_bubbles(global_path,Bspline_obj,occupied_positions_x,occupied_positions_y)

end_time_1 = time.time()

# new method v2
shifted_midpoints_x, shifted_midpoints_y, shifted_radii\
= generate_bubbles_v2(global_path[0],global_path[1],occupied_positions_x,occupied_positions_y)

end_time_2 = time.time()

#old method
occupied_positions = []    
for i in range(0, len(occupied_positions_x)):    
    occupied_positions.append([occupied_positions_x[i], occupied_positions_y[i]])
    
feasiblebubbles_x_mat, feasiblebubbles_y_mat, radii_mat, midpoints_x_mat, midpoints_y_mat \
    = defineDiscreteBubbles(Bspline_obj,occupied_positions_x,occupied_positions_y)  
 
end_time_3 = time.time()



#----------------------------------------------------------------------------#
#                         Comparison - time                                  #
#----------------------------------------------------------------------------#


print("time needed for generating of bubbles / new method / in seconds: "    , end_time_1 - start_time)
print("time needed for generating of bubbles / new method v2 / in seconds: " , end_time_2 - end_time_1)
print("time needed for generating of bubbles / old method / in seconds: "    , end_time_3 - end_time_2)



#----------------------------------------------------------------------------#
#                    Create feasible bubble points                           #
#----------------------------------------------------------------------------#


npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
    
shifted_feasiblebubbles_x = []
shifted_feasiblebubbles_y = []
for i in range (0, len(shifted_midpoints_x)):
        shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
        shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))





#----------------------------------------------------------------------------#
#                         Comparison - Plots                                 #
#----------------------------------------------------------------------------#

         

plt.figure()
plt.plot(global_path[0], global_path[1], 'b-')
plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'rx')
plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
plt.plot(shifted_feasiblebubbles_x,shifted_feasiblebubbles_y,'gx',markersize= 0.5)
plt.legend(['Global Reference Trajectory','Midpoints of the bubbles', 'Occupied Positions'])
plt.title('Results of new method with shifting bubbles')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])



# plt.figure()
# plt.plot(global_path[0], global_path[1], 'b-')
# plt.plot(midpoints_x_mat, midpoints_y_mat, 'rx')
# plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
# plt.legend(['Global Reference Trajectory','Midpoints of the bubbles', 'Occupied Positions'])
# plt.title('Results of old method')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([xlim_min,xlim_max])
# plt.ylim([ylim_min,ylim_max])

# plt.figure()
# plt.plot(global_path[0], global_path[1], 'b-')
# plt.plot(midpoints_x_mat, midpoints_y_mat, 'rx')
# plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
# plt.plot(feasiblebubbles_x_mat,feasiblebubbles_y_mat,'yx',markersize= 0.5)
# plt.legend(['Global Reference Trajectory','Midpoints of the bubbles', 'Occupied Positions'])
# plt.title('Results of old method')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([xlim_min,xlim_max])
# plt.ylim([ylim_min,ylim_max])






















