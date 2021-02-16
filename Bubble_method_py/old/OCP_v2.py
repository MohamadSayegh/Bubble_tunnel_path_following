

"""
Created on Thu Nov 29

Ninja Robot Thesis

----------------------------------------------------------------------------
                      Results Discussion and notes                         
----------------------------------------------------------------------------

1) The curve at the beginning of the solution is because of 
   the initial angle at 0 

2) There is still a problem: how to impose that the path goes towards
   the more open space in the bubble ? (away from obstacles)

3) The bubble method here is not time optimal (excution time)
   this needs to be considered when the code is applied in c++
 
4) Should generate more case scnearios to make sure this
   works in all cases
   
   recieved feedback = here I add two path parameters instead of one
   
   #### note: the code still sometimes doesnt work =  constraints violated on end goal, or control v
                                                   =  when I change the weights = is this normal ?


@author: Mohamad Sayegh


"""

import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from Bubble_tunnel_generation import generate_bubbles, plotting
from Grid_generation import create_obstacles, create_global_path


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


#using functions 
feasiblebubbles_x, feasiblebubbles_y,shifted_feasiblebubbles_x,\
shifted_feasiblebubbles_y,midpoints_x, midpoints_y, radii,\
shifted_midpoints_x, shifted_midpoints_y, shifted_radii\
= generate_bubbles(global_path,Bspline_obj,occupied_positions_x,occupied_positions_y)

   
plotting(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
             xlim_min, xlim_max, ylim_min, ylim_max,feasiblebubbles_x, feasiblebubbles_y,\
             shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, midpoints_x, midpoints_y, radii,\
             shifted_midpoints_x, shifted_midpoints_y, shifted_radii)


#----------------------------------------------------------------------------#
#                  Creating the new shifted path                             #
#----------------------------------------------------------------------------#

#this path will be used only for the obstacle avoidance = creating by the new midpoints

#change s here to control smoothness of path = could affect the ocp
shifted_Bspline_obj, u = interpolate.splprep([shifted_midpoints_x,shifted_midpoints_y], u = None, s = 1) 


u = np.linspace(0,1,100)
shifted_global_path = interpolate.splev(u, shifted_Bspline_obj)


plt.figure()
plt.plot(shifted_global_path[0], shifted_global_path[1], 'r-')
plt.plot(global_path[0], global_path[1], 'g-')
plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 2)
plt.legend(['shifted global path','Original global path','Obstacles'])
plt.title('The original path and the shifted path')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#


ocp = Ocp(T = FreeTime(10.0))


N   = 20            # number of control intervals


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



# Initial states 
ocp.set_initial(x,      initial_pos_x)
ocp.set_initial(y,      initial_pos_y)
ocp.set_initial(theta,  0.0)


# Constraints at t0
ocp.subject_to(ocp.at_t0(x)      == initial_pos_x)
ocp.subject_to(ocp.at_t0(y)      == initial_pos_y)
ocp.subject_to(ocp.at_t0(theta)  == 0.0)
ocp.subject_to(ocp.at_t0(s1)     == 0.0)     
ocp.subject_to(ocp.at_t0(s2)     == 0.0)    

#Constraints at tf
ocp.subject_to(ocp.at_tf(x) == end_goal_x)
ocp.subject_to(ocp.at_tf(y) == end_goal_y)
ocp.subject_to(ocp.at_tf(theta) == 0.0)
ocp.subject_to(ocp.at_tf(s1) == 1)              #this constraint on s is sometimes violated. why ?
ocp.subject_to(ocp.at_tf(s2) == 1)


#constraints on controls 
ocp.subject_to(  0      <=  ( v  <= 1   ))
ocp.subject_to( -pi     <=  ( w  <= pi  ))
ocp.subject_to(  sdot1  >=  0)                      #path "increasing"
ocp.subject_to(  sdot2  >=  0)   


#-----------  use the midpoints and radius to create interpolated tunnels


#------------ Obscatles avoidance tunnel

bubbles_radii     =  shifted_radii
bubbles_x         =  shifted_midpoints_x
bubbles_y         =  shifted_midpoints_y
tlength1          =  len(bubbles_x)
tunnel_s1         =  np.linspace(0,1,tlength1) 


ocp.set_initial(s1, np.linspace(tunnel_s1[0],tunnel_s1[-9], N))  #dont take the last points in tunnel_s so the path wont go back to zero
ocp.set_initial(sdot1, (tunnel_s1[-2]-tunnel_s1[0])/tlength1)


obs_spline_x = interpolant('x','bspline',[tunnel_s1],bubbles_x,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})
obs_spline_y = interpolant('y','bspline',[tunnel_s1],bubbles_y,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})
obs_spline_r = interpolant('r','bspline',[tunnel_s1],bubbles_radii,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})


#------------ Path Tunnel avoidance 

#re-evaluate the path only in same length as bubbles data


path_x         =  global_path[0]
path_y         =  global_path[1]
tlength2       =  len(path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 


ocp.set_initial(s2, np.linspace(tunnel_s2[0],tunnel_s2[-9], N))  #dont take the last points in tunnel_s so the path wont go back to zero
ocp.set_initial(sdot1, (tunnel_s2[-2]-tunnel_s2[0])/tlength2)

path_spline_x = interpolant('x','bspline',[tunnel_s2],path_x,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})  #low fraction ?
path_spline_y = interpolant('y','bspline',[tunnel_s2],path_y,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})

#---------------- Slack variables for soft constraints 

slack_obs       = ocp.variable()
slack_next_obs  = ocp.variable()

ocp.subject_to(slack_obs      >= 0)
ocp.subject_to(slack_next_obs >= 0)

#stay in bubbles as much as possible
ocp.subject_to( (  ( x - obs_spline_x(s1) )**2 + ( y-obs_spline_y(s1))**2  - obs_spline_r(s1)**2 ) <= slack_obs )

#stay in next bubbles as much as possible
ocp.subject_to( (  ( x - obs_spline_x(ocp.next(s1)) )**2 + ( y-obs_spline_y(ocp.next(s1)))**2  - obs_spline_r(ocp.next(s1))**2 ) <= slack_next_obs )
 

# ------------- Objective function 

#path following
ocp.add_objective(2*ocp.integral((x - path_spline_x(s2))**2 + (y-path_spline_y(s2))**2))

ocp.add_objective(3*slack_obs)

ocp.add_objective(3*slack_next_obs)


# ------------- Solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)


# Make it concrete for this ocp
# N -- number of control intervals
# M -- number of integration steps per control interval
# grid -- could specify e.g. UniformGrid() or GeometricGrid(4)
ocp.method(MultipleShooting(N=N,M=1,intg='rk'))

#-------------------------------------------------------------------------------#
#                          OCP Solution and Results                             #
#-------------------------------------------------------------------------------#

try:
    sol = ocp.solve()
except:
    #failed_to_converge = True
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution


plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
ts, s_obs = sol.sample(s1, grid='integrator',refine = 200)
ts = np.linspace(0,2*np.pi,1000)
xspline_obs = np.array(obs_spline_x(s_obs))
yspline_obs = np.array(obs_spline_y(s_obs))
rspline_obs = np.array(obs_spline_r(s_obs))
for i in range(s_obs.shape[0]): plt.plot(xspline_obs[i]+rspline_obs[i]*cos(ts),yspline_obs[i]+rspline_obs[i]*sin(ts),'r-',markersize = 0.5)

ts, s_path = sol.sample(s2, grid='integrator',refine = 200)
ts = np.linspace(0,2*np.pi,1000)
xspline_path = np.array(path_spline_x(s_path))
yspline_path = np.array(path_spline_y(s_path))
plt.plot(xspline_path, yspline_path, 'g--')

tsol, xsol = sol.sample(x, grid='control')
tsol, ysol = sol.sample(y, grid='control')
plt.plot(xsol, ysol,'bo')
tsol, xsol = sol.sample(x, grid='integrator',refine=10)
tsol, ysol = sol.sample(y, grid='integrator',refine=10)
plt.plot(xsol, ysol, '--')
plt.title('OCP solution')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


#dont refine to see the original points
tsol, xsol = sol.sample(x, grid='integrator')
tsol, ysol = sol.sample(y, grid='integrator')
plt.figure()
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 1)
plt.plot(xsol, ysol, 'rx', markersize = 5)
plt.legend(['original path', 'obstacles','solution'])
plt.title('Solution compared to initial global path')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


# plt.figure()
# plt.plot(global_path[0], global_path[1], 'g--')
# plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 1)
# plt.plot(xsol, ysol, 'rx', markersize = 5)
# plt.legend(['original path', 'obstacles','solution'])
# plt.title('Solution compared to initial global path')
# plt.xlim([2.9,5])
# plt.ylim([-0.2,3])

# #plot velocities to verify that velocities are high (close to the limit)
# tsol, vsol = sol.sample(v, grid='integrator')
# tsol, wsol = sol.sample(w, grid='integrator')

# plt.figure()
# plt.plot(tsol,vsol)
# plt.title('Linear Velocity of the solution')
# plt.xlabel('Solution Time [s]')
# plt.ylabel('Linear velocity [m/s]')

# plt.figure()
# plt.plot(tsol,wsol)
# plt.title('Angular Velocity of the solution')
# plt.xlabel('Solution Time [s]')
# plt.ylabel('Angular velocity [rad/s]')
























