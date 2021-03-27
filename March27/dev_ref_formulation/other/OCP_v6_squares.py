"""
Created on Dec 12

Ninja Robot Thesis

@author: Mohamad Sayegh


"""

import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate

from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting

from Bubble_tunnel_generation_v2 import generate_bubbles_v2, plotting_v2, create_tunnel, create_tunnel_sqaures, plotting_v2_squares
from Grid_generation import create_obstacles, create_global_path


#----------------------------------------------------------------------------#
#                         Generate Grid and Obstacles                        #
#----------------------------------------------------------------------------#

end_goal_x      =   9     # position of initial and end point
end_goal_y      =   9
initial_pos_x   =   0
initial_pos_y   =   0
xlim_min        =   -1
xlim_max        =   13
ylim_min        =   -1
ylim_max        =   11


obstacles_option = 1 
path_option = 1                  #options are 1 or 2

occupied_positions_x, occupied_positions_y = create_obstacles(obstacles_option)
Bspline_obj, global_path = create_global_path(path_option)

#----------------------------------------------------------------------------#
#                           Creating the Bubbles                             #
#----------------------------------------------------------------------------#


#using new function 

midpoints_x, midpoints_y, radii = generate_bubbles_v2(global_path[0],global_path[1],occupied_positions_x,occupied_positions_y)


plotting_v2_squares(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
                xlim_min, xlim_max, ylim_min, ylim_max, midpoints_x, midpoints_y, radii)
    

#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#


N   = 30        # number of control intervals   #20 works for path option 1
dt  = 3         # horizon

ocp = Ocp(T = N*dt)                       



# System model
x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()

#path parameters s1 and s2
s1       =  ocp.state()
sdot1    =  ocp.control()

s2       =  ocp.state()
sdot2    =  ocp.control()

#ODEs
ocp.set_der(x       ,        v*cos(theta))
ocp.set_der(y       ,        v*sin(theta))
ocp.set_der(theta   ,        w)
ocp.set_der(s1      ,        sdot1)
ocp.set_der(s2      ,        sdot2)



# Constraints at t0
ocp.subject_to(ocp.at_t0(x)      == initial_pos_x)
ocp.subject_to(ocp.at_t0(y)      == initial_pos_y)
ocp.subject_to(ocp.at_t0(theta)  == 0.0)
ocp.subject_to(ocp.at_t0(s1)     == 0.0)     
ocp.subject_to(ocp.at_t0(s2)     == 0.0)    

#---------------------  Constraints at tf ----------------------


pf = vertcat(end_goal_x,end_goal_y) # end point


#constraints on controls 
ocp.subject_to(  0      <=  ( v  <= 1   ))
ocp.subject_to( -pi     <=  ( w  <= pi  ))
ocp.subject_to(  sdot1  >=  0)                  
ocp.subject_to(  sdot2  >=  0)   


#------------ Obscatles avoidance tunnel ---------------------------

bubbles_radii     =  radii
bubbles_x         =  midpoints_x
bubbles_y         =  midpoints_y
tlength1          =  len(bubbles_x)
tunnel_s1         =  np.linspace(0,1,tlength1) 



ocp.subject_to(ocp.at_tf(s1) == 1)   


obs_spline_x = interpolant('obs_spline_x','bspline',[tunnel_s1],bubbles_x      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('obs_spline_y','bspline',[tunnel_s1],bubbles_y      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('obs_spline_r','bspline',[tunnel_s1],bubbles_radii  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#---------------------- Path Tunnel avoidance ----------------------

#re-evaluate the path for smaller number of points

u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)


path_x         =  global_path[0]
path_y         =  global_path[1]
tlength2       =  len(path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 


ocp.subject_to(ocp.at_tf(s2) == 1)

path_spline_x = interpolant('path_spline_x' , 'bspline', [tunnel_s2], path_x, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('path_spline_y' , 'bspline', [tunnel_s2], path_y, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})




# ------------------------ Initial guess ------------------------

# we want the initial guesss to be = the global path 

global_path_guess_x         = np.array(global_path[0])
global_path_guess_y         = np.array(global_path[1])

# Huge effect
global_path_guess_theta     = np.array([ 0.        ,  1.45000913,  1.44187553,  1.46705035,  1.41346249,
                                1.36090869,  1.48964826,  1.58368165,  1.53245965,  0.30439202,
                               -0.29080527, -1.35137464, -1.39060167, -1.34542412, -1.57897286,
                               -1.56806585, -1.36510406, -1.29393362, -0.14714978,  0.16139757,
                                1.1830938 ,  1.44548846,  1.72925488,  1.60977321,  1.24188699,
                                1.57465053,  1.68386922,  1.20732743,  0.16839987,  0.02339013,
                               -0.01030489])


ocp.set_initial(x,       global_path_guess_x) 
ocp.set_initial(y,       global_path_guess_y) 
ocp.set_initial(theta,   global_path_guess_theta)  

#path parameters
s_guess = np.linspace(0,1, N)
ocp.set_initial(s1, s_guess) 
ocp.set_initial(s2, s_guess)

sdot_guess = (s_guess[1]-s_guess[0])/dt

ocp.set_initial(sdot1, sdot_guess)
ocp.set_initial(sdot2, sdot_guess)

v_guess = np.ones(N)
w_guess = np.ones(N)

ocp.set_initial(v , v_guess)
ocp.set_initial(w , w_guess)


#----------------  constraints -------------------


C = cos(pi/4)
ocp.subject_to(  x  >=  ( obs_spline_x(s1) - C*obs_spline_r(s1) ))
ocp.subject_to(  x  <=  ( obs_spline_x(s1) + C*obs_spline_r(s1) ))
ocp.subject_to(  y  >=  ( obs_spline_y(s1) - C*obs_spline_r(s1) ))
ocp.subject_to(  y  <=  ( obs_spline_y(s1) + C*obs_spline_r(s1) ))



# ------------- Objective function ----------------------------------------

#path following
ocp.add_objective(ocp.integral((x - path_spline_x(s2))**2 + (y-path_spline_y(s2))**2))  


# ------------- Solution method------------------------------------------

options = {"ipopt": {"print_level": 5}}
options["expand"] = False
options["print_time"] = True
ocp.solver('ipopt', options)


# Multiple shooting
ocp.method(MultipleShooting(N=N,M=2,intg='rk'))

#-------------------------------------------------------------------------------#
#                          OCP Solution and Results                             #
#-------------------------------------------------------------------------------#

try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution




#---------------------- extract solution--------------------------


tsol, xsol          =  sol.sample(x, grid='control')
tsol, ysol          =  sol.sample(y, grid='control')
tsol, s_obs         =  sol.sample(s1, grid='control')
tsol, s_path        =  sol.sample(s2, grid='control')
tsol, xsol_refined  =  sol.sample(x, grid='integrator',refine=10)
tsol, ysol_refined  =  sol.sample(y, grid='integrator',refine=10)

tsol, vsol          =  sol.sample(v, grid='control')
tsol, wsol          =  sol.sample(w, grid='control')


#----------------------------- Plotting------------------------

npoints =  100  #numbr of points of every circle
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
        
        



#------------------ Plotting with respect to initial path/bubbles



plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
plt.plot(xsol, ysol,'bo', markersize = 5)
plt.legend(['original path','Obstacles', 'Feasible tunnel', 'OCP solution'])
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP Solution', dpi=300)


plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([1,3])
plt.ylim([8,11])
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 2)
plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
plt.plot(xsol, ysol,'bo', markersize = 5)
plt.legend(['original path','Occupied Positions', 'Feasible tunnel', 'OCP solution'], loc = "best")
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP Solution 1', dpi=300)


plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([2.9,6])
plt.ylim([-0.1,3])
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'bx',markersize = 3)
plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'g.', markersize= 0.2)
plt.plot(xsol, ysol,'bo', markersize = 5)
plt.legend(['original path','Occupied Positions', 'Feasible tunnel', 'OCP solution'], loc = (0.6,0.6))
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP Solution 2', dpi=300)


