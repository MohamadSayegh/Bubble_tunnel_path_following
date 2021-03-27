

import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate

from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting

from Bubble_tunnel_generation_v2 import generate_bubbles_v2, plotting_v2, create_tunnel
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



N   = 30        # number of control intervals   #20 works for path option 1
dt  = 3         # horizon

ocp = Ocp(T = N*dt)                       


#----------------------------------------------------------------------------#
#                       Creating the Path and Bubbles                        #
#----------------------------------------------------------------------------#



path_x = np.linspace(0, 9, 6)

path_y = [0., 3., 9., 1., 5., 9.]


Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 0)
u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)
    
shifted_midpoints_x = global_path[0]
shifted_midpoints_y = global_path[1]
shifted_radii       = 1*np.ones(N)


#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#



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

slack_tf_x = ocp.variable()
slack_tf_y = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)

ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - pf[0]) <= slack_tf_x))
ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - pf[1]) <= slack_tf_y))

ocp.add_objective(100*(slack_tf_x + slack_tf_y))


#constraints on controls 
ocp.subject_to(  0      <=  ( v  <= 1   ))
ocp.subject_to( -pi     <=  ( w  <= pi  ))
ocp.subject_to(  sdot1  >=  0)                  
ocp.subject_to(  sdot2  >=  0)   


#------------ Obscatles avoidance tunnel ---------------------------


bubbles_radii     =  shifted_radii
bubbles_x         =  shifted_midpoints_x
bubbles_y         =  shifted_midpoints_y
tlength1          =  len(bubbles_x)
tunnel_s1         =  np.linspace(0,1,tlength1) 


ocp.subject_to(ocp.at_tf(s1) <= 1)   


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


ocp.subject_to(ocp.at_tf(s2) <= 1)

path_spline_x = interpolant('path_spline_x' , 'bspline', [tunnel_s2], path_x, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('path_spline_y' , 'bspline', [tunnel_s2], path_y, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


# ------------------------ Initial guess ------------------------

# we want the initial guesss to be = the global path 

# global_path_guess_x         = np.array(global_path[0])
# global_path_guess_y         = np.array(global_path[1])

# global_path_guess_theta     = np.zeros(N)

#try bubbles midpoints as guesses
# Bspline_obj, u = interpolate.splprep([shifted_midpoints_x,shifted_midpoints_y], u = None, s = 0)
# u = np.linspace(0,1,N)
# path_guess = interpolate.splev(u, Bspline_obj)

# path_guess_x         = np.array(path_guess[0])
# path_guess_y         = np.array(path_guess[1])


ocp.set_initial(x,       global_path[0]) 
ocp.set_initial(y,       global_path[1]) 



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

v_guess = 0.5*np.ones(N)
ocp.set_initial(v , v_guess)

w_guess = np.zeros(N)
ocp.set_initial(w , w_guess)


#----------------  constraints -------------------


#stay in bubbles as much as possible

ocp.subject_to( (  ( x - obs_spline_x(s1) )**2 + ( y-obs_spline_y(s1) )**2  < (obs_spline_r(s1)**2 ))  )


# ------------- Objective function ----------------------------------------

#path following
ocp.add_objective(1*ocp.integral((x - path_spline_x(s2))**2 + (y-path_spline_y(s2))**2))  


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







#---------------------- extract solution
tsol, xsol          =  sol.sample(x, grid='control')
tsol, ysol          =  sol.sample(y, grid='control')
tsol, s_obs         =  sol.sample(s1, grid='control')
tsol, s_path        = sol.sample(s2, grid='control')
tsol, xsol_refined  = sol.sample(x, grid='integrator',refine=100)
tsol, ysol_refined  = sol.sample(y, grid='integrator',refine=100)



#------------------------ Plotting with path/bubbles depending on path parameter solution

#note: here the path and bubbles are plotted based on the path parameters of the solution
# so they can be other than the real global path and bubbles

xspline_path = np.array(path_spline_x(s_path))
yspline_path = np.array(path_spline_y(s_path))

plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


ts = np.linspace(0,2*np.pi,50)
xspline_obs = np.array(obs_spline_x(s_obs))
yspline_obs = np.array(obs_spline_y(s_obs))
rspline_obs = np.array(obs_spline_r(s_obs))
for i in range(s_obs.shape[0]): plt.plot(xspline_obs[i]+rspline_obs[i]*cos(ts),yspline_obs[i]+rspline_obs[i]*sin(ts),'r-',markersize = 0.5)

plt.plot(xspline_path, yspline_path, 'g--')
plt.plot(xsol, ysol,'bo')
plt.plot(xsol_refined, ysol_refined, '--')
plt.title('OCP solution')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.show(block=True)


#------------------ Plotting with respect to initial path/bubbles


tunnel_x, tunnel_y = create_tunnel(shifted_midpoints_x,shifted_midpoints_y,shifted_radii)
       

plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.plot(global_path[0], global_path[1], 'c--')
plt.plot(tunnel_x, tunnel_y, 'r.', markersize= 1)
plt.plot(xsol, ysol,'bo', markersize = 5)
plt.plot(xsol, ysol,'b-', markersize = 3)
plt.legend(['original path', 'Feasible tunnel', 'Solution points', 'Solution trajectory'])
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP Solution 0 ', dpi=300)


plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.plot(global_path[0], global_path[1], 'g--')
plt.plot(tunnel_x, tunnel_y, 'r.', markersize= 1)
plt.plot(0,0,'bx', markersize = 10)
plt.plot(9,9,'rx', markersize = 10)
plt.legend(['original path','Obstacles', 'Feasible tunnel', 'Starting point', 'End point'])
plt.title('Global Path and feasible tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP problem', dpi=300)


# plt.figure(dpi=300)
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([1,3])
# plt.ylim([8,11])
# plt.plot(global_path[0], global_path[1], 'g--')
# plt.plot(tunnel_x, tunnel_y, 'r.', markersize= 3)
# plt.plot(xsol, ysol,'bo', markersize = 5)
# plt.plot(xsol, ysol,'b-', markersize = 3)
# plt.legend(['original path','Obstacles', 'Feasible tunnel', 'Solution points', 'Solution trajectory'])
# plt.title('OCP Solutin with given path and tunnel')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.savefig('OCP Solution 1', dpi=300)


# plt.figure(dpi = 300)
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([2.9,6])
# plt.ylim([-0.1,3])
# plt.plot(global_path[0], global_path[1], 'g--')
# plt.plot(tunnel_x, tunnel_y, 'r.', markersize= 3)
# plt.plot(xsol, ysol,'bo', markersize = 5)
# plt.plot(xsol, ysol,'b-', markersize = 3)
# plt.legend(['original path','Obstacles', 'Feasible tunnel', 'Solution points', 'Solution trajectory'], loc = (0.6,0.6))
# plt.title('OCP Solutin with given path and tunnel')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.savefig('OCP Solution 2', dpi=300)
