
"""
Created on Dec 04

Ninja Robot Thesis

@author: Mohamad Sayegh


1) bubble generation method enhanced to be faster

2) assuming no collision of global path 

goal = make solutin faster 


fixes: 
    
    1) no shifting of bubbles of first and last points
    
    2) global path has initial position as first point and last position as last point === by using s = 0 and low frac
    
    3) making the staying within bubble as hard constraint => no slack varaiables
    
    
    
questions: 1) meaning of condition on ocp.at_tf(s) ?? 
    
           2) constraints ? tolerance ?





"""

import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from Bubble_tunnel_generation_v2 import generate_bubbles_v2, plotting_v2
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
ylim_min        =   -2
ylim_max        =   16
n               =   10    # size of square grid


''' 
possible combinations of the two options below: 1/1 2/1

option 1/1 works 

option 2/1 doesnt work 

''' 
obstacles_option = 1 
path_option = 1

occupied_positions_x, occupied_positions_y = create_obstacles(obstacles_option)
Bspline_obj, global_path = create_global_path(path_option)

#----------------------------------------------------------------------------#
#                           Creating the Bubbles                             #
#----------------------------------------------------------------------------#


#using new function 

shifted_midpoints_x, shifted_midpoints_y, shifted_radii\
                = generate_bubbles_v2(global_path[0],global_path[1],occupied_positions_x,occupied_positions_y)


plotting_v2(initial_pos_x, end_goal_x, global_path, occupied_positions_x, occupied_positions_y,\
                xlim_min, xlim_max, ylim_min, ylim_max,\
                shifted_midpoints_x, shifted_midpoints_y, shifted_radii)
    


#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#


# ocp = Ocp(T = FreeTime(10.0))
ocp = Ocp(T = 60.0)                           # no freetime = reduces solution time 


N   = 20        # number of control intervals


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

#Constraints at tf
ocp.subject_to(ocp.at_tf(x) == end_goal_x)
ocp.subject_to(ocp.at_tf(y) == end_goal_y)
ocp.subject_to(ocp.at_tf(theta) == 0.0)



#constraints on controls 
ocp.subject_to(  0      <=  ( v  <= 1   ))
ocp.subject_to( -pi     <=  ( w  <= pi  ))
ocp.subject_to(  sdot1  >=  0)                      #path "increasing"
ocp.subject_to(  sdot2  >=  0)   


#-----------  use the midpoints and radius to create interpolated tunnels


#------------ Obscatles avoidance tunnel ---------------------------

bubbles_radii     =  shifted_radii
bubbles_x         =  shifted_midpoints_x
bubbles_y         =  shifted_midpoints_y
tlength1          =  len(bubbles_x)
tunnel_s1         =  np.linspace(0,1,tlength1) 



ocp.subject_to(ocp.at_tf(s1) == 1)   


obs_spline_x = interpolant('x','bspline',[tunnel_s1],bubbles_x      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('y','bspline',[tunnel_s1],bubbles_y      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('r','bspline',[tunnel_s1],bubbles_radii  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#---------------------- Path Tunnel avoidance ----------------------

#re-evaluate the path only in same length as bubbles data


path_x         =  global_path[0]
path_y         =  global_path[1]
tlength2       =  len(path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 



ocp.subject_to(ocp.at_tf(s2) < 1)

path_spline_x = interpolant('x','bspline', [tunnel_s2], path_x, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('y','bspline', [tunnel_s2], path_y, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})




# ------------------------ Initial guess ------------------------

# we want the initial guesss to be = the global path 

#create global path N points
u = np.linspace(0,1,N)
global_path_guess           = interpolate.splev(u, Bspline_obj)
global_path_guess_x         = np.array(global_path_guess[0])
# global_path_guess_x         = np.linspace(initial_pos_x, end_goal_x, N)
# global_path_guess_y         = np.array(global_path_guess[1])
global_path_guess_y         = np.linspace(initial_pos_y, end_goal_y, N)   #for some reason this is better for y
global_path_guess_theta     = np.zeros(N)

ocp.set_initial(x,       global_path_guess_x) 
ocp.set_initial(y,       global_path_guess_y) 
# ocp.set_initial(theta,   global_path_guess_theta)  #doesnt have an effect

#path parameters
s1_guess = np.linspace(tunnel_s1[0],tunnel_s1[-3], N)
sdot1_guess = (tunnel_s1[-1]-tunnel_s1[0])/tlength1 

ocp.set_initial(s1, s1_guess) 
ocp.set_initial(sdot1, sdot1_guess)

s2_guess = np.linspace(tunnel_s2[0],tunnel_s2[-3], N)
sdot2_guess = (tunnel_s2[-1]-tunnel_s2[0])/tlength2

ocp.set_initial(s2, s2_guess)
ocp.set_initial(sdot2, sdot2_guess)

#constraints on control inputs have a slight positive effect on solution time
ocp.set_initial(v , 0.0)
ocp.set_initial(w , 0.0)


#---------------- Slack variables for soft constraints -------------------

tolerance = 3  #adding this tolerance has reduced solution time and also gave better solution, but at tight areas it should not work

#stay in bubbles as much as possible

ocp.subject_to( (  ( x - obs_spline_x(s1) )**2 + ( y-obs_spline_y(s1) )**2  < (tolerance + obs_spline_r(s1)**2 ))  )

# ocp.subject_to( (  ( x - obs_spline_x(ocp.next(s1)) )**2 + ( y-obs_spline_y(ocp.next(s1)))**2  <  (tolerance + obs_spline_r(ocp.next(s1))**2 ) ) )
 


# ------------- Objective function ----------------------------------------

#path following
ocp.add_objective(ocp.integral((x - path_spline_x(s2))**2 + (y-path_spline_y(s2))**2))    #not enough by itself to make path following a priority

# ocp.add_objective(-ocp.at_tf(s2))



# ------------- Solution method------------------------------------------
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
ts, s_obs = sol.sample(s1, grid='integrator',refine = 500)
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
plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1.5)
plt.title('OCP solution')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


# dont refine to see the original points

# tsol, xsol = sol.sample(x, grid='integrator')
# tsol, ysol = sol.sample(y, grid='integrator')
# plt.figure()
# plt.plot(global_path[0], global_path[1], 'g--')
# plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 1)
# plt.plot(xsol, ysol, 'rx', markersize = 5)
# plt.legend(['original path', 'obstacles','solution'])
# plt.title('Solution compared to initial global path')
# plt.xlim([xlim_min,xlim_max])
# plt.ylim([ylim_min,ylim_max])


# plt.figure()
# plt.plot(global_path[0], global_path[1], 'g--')
# plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 1)
# plt.plot(xsol, ysol, 'rx', markersize = 5)
# plt.legend(['original path', 'obstacles','solution'])
# plt.title('Solution compared to initial global path')
# plt.xlim([2.9,5])
# plt.ylim([-0.2,3])

#plot velocities to verify that velocities are high (close to the limit)
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


# tsol, sdot1_sol = sol.sample(sdot1, grid='integrator')
# tsol, sdot2_sol = sol.sample(sdot2, grid='integrator')

# plt.figure()
# plt.plot(tsol,sdot1_sol)

# plt.figure()
# plt.plot(tsol,sdot2_sol)























