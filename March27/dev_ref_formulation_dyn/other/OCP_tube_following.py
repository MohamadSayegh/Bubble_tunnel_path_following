

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


obstacles_option = 1 
path_option = 1                  #options are 1 or 2

occupied_positions_x, occupied_positions_y = create_obstacles(obstacles_option)
Bspline_obj, global_path = create_global_path(path_option)

#----------------------------------------------------------------------------#
#                           Creating the Bubbles                             #
#----------------------------------------------------------------------------#


#using new function 

shifted_midpoints_x, shifted_midpoints_y, shifted_radii\
                = generate_bubbles_v2(global_path[0],global_path[1],occupied_positions_x,occupied_positions_y)



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



#ODEs
ocp.set_der(x       ,        v*cos(theta))
ocp.set_der(y       ,        v*sin(theta))
ocp.set_der(theta   ,        w)
ocp.set_der(s1      ,        sdot1)




# Constraints at t0
ocp.subject_to(ocp.at_t0(x)      == initial_pos_x)
ocp.subject_to(ocp.at_t0(y)      == initial_pos_y)
ocp.subject_to(ocp.at_t0(theta)  == 0.0)
ocp.subject_to(ocp.at_t0(s1)     == 0.0)     
  

#---------------------  Constraints at tf ----------------------


# pf = vertcat(end_goal_x,end_goal_y) # end point

# slack_tf_x = ocp.variable()
# slack_tf_y = ocp.variable()

# ocp.subject_to(slack_tf_x >= 0)
# ocp.subject_to(slack_tf_y >= 0)

# ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - pf[0]) <= slack_tf_x))
# ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - pf[1]) <= slack_tf_y))

# ocp.add_objective(1*(slack_tf_x + slack_tf_y))


#constraints on controls 
ocp.subject_to(  0      <=  ( v  <= 1   ))
ocp.subject_to( -pi     <=  ( w  <= pi  ))

ocp.subject_to(  sdot1  >=  0)                  


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



# ------------------------ Initial guess ------------------------

# we want the initial guesss to be = the global path 
u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)

global_path_guess_x         = np.array(global_path[0])
global_path_guess_y         = np.array(global_path[1])

ocp.set_initial(x,       global_path_guess_x) 
ocp.set_initial(y,       global_path_guess_y) 


#path parameters
s1_guess = np.linspace(0,1, N)
sdot1_guess = (s1_guess[1] - s1_guess[0])/N 

ocp.set_initial(s1, s1_guess) 
ocp.set_initial(sdot1, sdot1_guess)


# v_guess = 1*np.ones(N)
# ocp.set_initial(v , v_guess)

# w_guess = np.zeros(N)
# ocp.set_initial(w , w_guess)


#----------------  constraints -------------------

tolerance = 0  #no need for tolerance anymore = makes more sense now

#stay in bubbles as much as possible

ocp.subject_to( (  ( x - obs_spline_x(s1) )**2 + ( y-obs_spline_y(s1) )**2  < (tolerance + obs_spline_r(s1)**2 ))  )


# ------------- Objective function ----------------------------------------


ocp.add_objective(-100*ocp.at_tf(s1))

# ocp.add_objective(ocp.sum(v))
                  
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

tsol,       xsol          =  sol.sample(x, grid='control')
tsol,       ysol          =  sol.sample(y, grid='control')
tsol_sobs,  s_obs         =  sol.sample(s1, grid='control')
tsol,       xsol_refined  =  sol.sample(x, grid='integrator',refine=100)
tsol,       ysol_refined  =  sol.sample(y, grid='integrator',refine=100)


print("end point x: ", xsol[-1])
print("end point y: ", ysol[-1])

#------------------------ Plotting with path/bubbles depending on path parameter solution

#note: here the path and bubbles are plotted based on the path parameters of the solution
# so they can be other than the real global path and bubbles

plt.figure(dpi=300)
plt.plot(tsol_sobs, s_obs)



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
plt.plot(xsol, ysol,'bo')
plt.plot(xsol_refined, ysol_refined, '--')
plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1.5)
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
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 1)
plt.plot(tunnel_x, tunnel_y, 'r.', markersize= 1)
plt.plot(xsol, ysol,'bo', markersize = 5)
plt.plot(xsol, ysol,'b-', markersize = 3)
plt.legend(['original path','Obstacles', 'Feasible tunnel', 'Solution points', 'Solution trajectory'])
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')



