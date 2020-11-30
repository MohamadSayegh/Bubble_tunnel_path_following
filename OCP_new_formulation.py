"""

Created on Thu Nov 26

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


#----------------------------------------------------------------------------#
#                         Generate Grid and Obstacles                        #
#----------------------------------------------------------------------------#

end_goal_x      =   9;     # position of initial and end point
end_goal_y      =   9;
initial_pos_x   =   0;
initial_pos_y   =   0;
xlim_min        =   -0.5;  # xlim and ylim of plots
xlim_max        =   10.5;
ylim_min        =   -3;
ylim_max        =   14;
n               =   10;    # size of square grid


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


# Manually design the main points of the path
path_x = np.linspace(0, 9, 20)
path_y = [ 0.0, 5.0, 10.0, 11.0, 11.0, 11.0, 11.0, 6.0, 0.1, 0.1,
           0.1, 0.1, 1.0, 1.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0 ]


# Interpolate the path using a spline
# this interpolation will now be considered the 'original global path'
Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 1)
u = np.linspace(0,1,100)
global_path = interpolate.splev(u, Bspline_obj)


#----------------------------------------------------------------------------#
#                           Creating the Bubbles                             #
#----------------------------------------------------------------------------#

    
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

shifted_Bspline_obj, u = interpolate.splprep([shifted_midpoints_x,shifted_midpoints_y], u = None, s = 0)
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

#path parameter s
s       =  ocp.state()
sdot    =  ocp.control()

#ODEs
ocp.set_der(x       ,        v*cos(theta))
ocp.set_der(y       ,        v*sin(theta))
ocp.set_der(theta   ,        w)
ocp.set_der(s       ,        sdot)



# Initial states 
ocp.set_initial(x,      initial_pos_x)
ocp.set_initial(y,      initial_pos_y)
ocp.set_initial(theta,  0.0)


# Constraints at t0
ocp.subject_to(ocp.at_t0(x)      == initial_pos_x)
ocp.subject_to(ocp.at_t0(y)      == initial_pos_y)
ocp.subject_to(ocp.at_t0(theta)  == 0.0)
ocp.subject_to(ocp.at_t0(s)      == 0.0)     

#Constraints at tf
ocp.subject_to(ocp.at_tf(x) == end_goal_x)
ocp.subject_to(ocp.at_tf(y) == end_goal_y)
ocp.subject_to(ocp.at_tf(theta) == 0.0)
ocp.subject_to(ocp.at_tf(s) == 1)


#constraints on controls 
ocp.subject_to(  0    <= ( v  <= 1   ))
ocp.subject_to( -pi   <= ( w  <= pi  ))
ocp.subject_to( sdot  >=   0)              #path "increasing"



#-----------  use the midpoints and radius to create interpolated tunnels


#------------ Obscatles avoidance tunnel

bubbles_radii   =  shifted_radii
bubbles_x       =  shifted_midpoints_x
bubbles_y       =  shifted_midpoints_y
tlength         =  len(bubbles_x)
tunnel_s        =  np.linspace(0,1,tlength) 


ocp.set_initial(s, np.linspace(tunnel_s[0],tunnel_s[-7], N))  #dont take the last points in tunnel_s so the path wont go back to zero
ocp.set_initial(sdot, (tunnel_s[-2]-tunnel_s[0])/tlength)


obs_spline_x = interpolant('x','bspline',[tunnel_s],bubbles_x,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})
obs_spline_y = interpolant('y','bspline',[tunnel_s],bubbles_y,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})
obs_spline_r = interpolant('r','bspline',[tunnel_s],bubbles_radii,{"algorithm": "smooth_linear","smooth_linear_frac":0.1})


#------------ Path Tunnel avoidance 

#re-evaluate the path only in same length as bubbles data


global_path     =  interpolate.splev(tunnel_s,Bspline_obj)
path_x          =  global_path[0]
path_y          =  global_path[1]


path_spline_x = interpolant('x','bspline',[tunnel_s],path_x,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})  #low fraction ?
path_spline_y = interpolant('y','bspline',[tunnel_s],path_y,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})

#---------------- Slack variables for soft constraints 

slack_obs       = ocp.variable()
slack_next_obs  = ocp.variable()

ocp.subject_to(slack_obs      >= 0)
ocp.subject_to(slack_next_obs >= 0)

#stay in bubbles as much as possible
ocp.subject_to( (  ( x - obs_spline_x(s) )**2 + ( y-obs_spline_y(s))**2  - obs_spline_r(s)**2 ) <= slack_obs )

#stay in next bubbles as much as possible
ocp.subject_to( (  ( x - obs_spline_x(ocp.next(s)) )**2 + ( y-obs_spline_y(ocp.next(s)))**2  - obs_spline_r(ocp.next(s))**2 ) <= slack_next_obs )
 

# ------------- Objective function 

#path following
ocp.add_objective(3.5*ocp.integral((x - path_spline_x(s))**2 + (y-path_spline_y(s))**2))

ocp.add_objective(2*slack_obs)

ocp.add_objective(2*slack_next_obs)


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
ts, s_obs = sol.sample(s, grid='integrator',refine = 100)
ts = np.linspace(0,2*np.pi,1000)
xspline_obs = np.array(obs_spline_x(s_obs))
yspline_obs = np.array(obs_spline_y(s_obs))
rspline_obs = np.array(obs_spline_r(s_obs))
for i in range(s_obs.shape[0]): plt.plot(xspline_obs[i]+rspline_obs[i]*cos(ts),yspline_obs[i]+rspline_obs[i]*sin(ts),'r-',markersize = 0.5)

ts, s_path = sol.sample(s, grid='integrator',refine = 100)
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


plt.figure()
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 1)
plt.plot(xsol, ysol, 'rx', markersize = 5)
plt.legend(['original path', 'obstacles','solution'])
plt.title('Solution compared to initial global path')
plt.xlim([2.9,5])
plt.ylim([-0.2,3])

#plot velocities to verify that velocities are high (close to the limit)
tsol, vsol = sol.sample(v, grid='integrator')
tsol, wsol = sol.sample(w, grid='integrator')

plt.figure()
plt.plot(tsol,vsol)
plt.title('Linear Velocity of the solution')
plt.xlabel('Solution Time [s]')
plt.ylabel('Linear velocity [m/s]')

plt.figure()
plt.plot(tsol,wsol)
plt.title('Angular Velocity of the solution')
plt.xlabel('Solution Time [s]')
plt.ylabel('Angular velocity [rad/s]')
























