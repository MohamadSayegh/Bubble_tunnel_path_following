"""
Created on Dec 05

Ninja Robot Thesis

@author: Mohamad Sayegh


using MPC functions that send parts of grid every time 


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
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc

#-------------------------------------------------------------------------------#
#                   Initialization of Grid and bubbles                          #
#-------------------------------------------------------------------------------#

global_end_goal_x      =   9;   
global_end_goal_y      =   9;


initial_pos_x   =   0;
initial_pos_y   =   0;


#xlim and ylim of plots
xlim_min        =   -0.5;  
xlim_max        =   10.5;
ylim_min        =   -2;
ylim_max        =   12;


obstacles_option   = 1 
path_option        = 1


#------------------------- Generate grid and path ---------------------------

global_path_x, global_path_y , Bspline_obj       =  create_global_path_mpc(path_option,initial_pos_x,initial_pos_y)
occupied_positions_x, occupied_positions_y       =  create_obstacles_mpc(obstacles_option, initial_pos_x, initial_pos_y)

#---------------- Creating the Bubbles----------------------------------------

if (occupied_positions_x.size != 0):
    
    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_v2(global_path_x,global_path_y,occupied_positions_x,occupied_positions_y)

else:  
    
    shifted_midpoints_x  =   global_path_x 
    shifted_midpoints_y  =   global_path_y
    shifted_radii        =   np.linspace(1,2,len(global_path_x))

    
#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#


ocp = Ocp(T = 20.0)         #time of one ocp problem ?

Nsim    = 10            # how much samples
N       = 10            # number of shooting points of every ocp


# Logging variables
time_hist           = np.zeros((Nsim+1, N+1))
x_hist              = np.zeros((Nsim+1, N+1))
y_hist              = np.zeros((Nsim+1, N+1))
theta_hist          = np.zeros((Nsim+1, N+1))
s_path_hist         = np.zeros((Nsim+1, N+1))
s_obs_hist          = np.zeros((Nsim+1, N+1))
v_hist              = np.zeros((Nsim+1, N+1))
w_hist              = np.zeros((Nsim+1, N+1))
sdot_path_hist      = np.zeros((Nsim+1, N+1))
sdot_obs_hist       = np.zeros((Nsim+1, N+1))


#number of states
nx = 5

# System model
x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()

#path parameters 
s_path       =  ocp.state()
sdot_path    =  ocp.control()

s_obs        =  ocp.state()
sdot_obs     =  ocp.control()

#ODEs
ocp.set_der(x            ,        v*cos(theta))
ocp.set_der(y            ,        v*sin(theta))
ocp.set_der(theta        ,        w)
ocp.set_der(s_path       ,        sdot_path)
ocp.set_der(s_obs        ,        sdot_obs)



# vector of states

X    = vertcat(x, y, theta, s_path, s_obs)

# Constraints on initial point and final state

X_0  = ocp.parameter(nx)
X_F  = ocp.parameter(nx)


ocp.subject_to(ocp.at_t0(X) == X_0)
ocp.subject_to(ocp.at_tf(X) == X_F)



#-------------------------------------------------------------------------------#
#                            Solve the first iteration                          #
#-------------------------------------------------------------------------------#


#same initial pos but end pos is the last point in global path of every section (i.e. local paths)
initial_pos_x =  initial_pos_x
initial_pos_y =  initial_pos_y

end_goal_x    =  global_path_x[-1]
end_goal_y    =  global_path_y[-1]

current_X     =  vertcat(initial_pos_x,   initial_pos_y,   0.0,  0.0,  0.0) 
goal_X        =  vertcat(end_goal_x,      end_goal_y,      0.0,  1.0,  1.0) 


ocp.set_value(X_0,   current_X)
ocp.set_value(X_F,   goal_X)


#constraints on controls 
ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
ocp.subject_to( sdot_path   >=   0)        
ocp.subject_to( sdot_obs    >=   0)



#------------ Obscatles avoidance tunnel ---------------------------


param_length = len(global_path_x)

bubbles_radii_param = ocp.parameter(param_length)
# bubbles_x_param     = ocp.parameter(param_length)
# bubbles_y_param     = ocp.parameter(param_length)

bubbles_radii     =  shifted_radii
bubbles_x         =  shifted_midpoints_x
bubbles_y         =  shifted_midpoints_y
tlength1          =  len(bubbles_x)

tunnel_s1         =  np.linspace(0,1,param_length) 
    

# ocp.set_value(bubbles_x_param,      bubbles_x)
# ocp.set_value(bubbles_y_param,      bubbles_y)

p = vertcat(bubbles_radii)
ocp.set_value(bubbles_radii_param,  p)


obs_spline_x = interpolant('obs_spline_x','bspline',[tunnel_s1],bubbles_x      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('obs_spline_y','bspline',[tunnel_s1],bubbles_y      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})



#---- replace bubbles_radii_param bi bubbles_radii for the code to work
obs_spline_r = interpolant('obs_spline_r','bspline',[tunnel_s1],bubbles_radii_param , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})




#---------------------- Path Following Tunnel ----------------------


tlength2       =  len(global_path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 

path_spline_x = interpolant('path_spline_x' , 'bspline', [tunnel_s2], global_path_x, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('path_spline_y' , 'bspline', [tunnel_s2], global_path_y, {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


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
s_obs_guess = np.linspace(tunnel_s1[0],tunnel_s1[-3], N)
sdot_obs_guess = (tunnel_s1[-1]-tunnel_s1[0])/tlength1 

ocp.set_initial(s_obs, s_obs_guess) 
ocp.set_initial(sdot_obs, sdot_obs_guess)

s_path_guess = np.linspace(tunnel_s2[0],tunnel_s2[-3], N)
sdot_path_guess = (tunnel_s2[-1]-tunnel_s2[0])/tlength2

ocp.set_initial(s_path , s_path_guess )
ocp.set_initial(sdot_path_guess, sdot_path_guess)

#constraints on control inputs have a slight positive effect on solution time
ocp.set_initial(v , 0.0)
ocp.set_initial(w , 0.0)


#----------------  Obstacle avoidance constraints -------------------

tolerance = 3  #adding this tolerance has reduced solution time and also gave better solution, but at tight areas it should not work

#stay in bubbles as much as possible

ocp.subject_to( (  ( x - obs_spline_x(s_obs) )**2 + ( y-obs_spline_y(s_obs) )**2  < (tolerance + obs_spline_r(s_obs)**2 ))  )

# ocp.subject_to( (  ( x - obs_spline_x(ocp.next(s1)) )**2 + ( y-obs_spline_y(ocp.next(s1)))**2  <  (tolerance + obs_spline_r(ocp.next(s1))**2 ) ) )
 



# ------------- Objective function ----------------------------------------

#path following
ocp.add_objective(ocp.integral((x - path_spline_x(s_path))**2 + (y-path_spline_y(s_path))**2))    #not enough by itself to make path following a priority

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
ts, s1 = sol.sample(s_obs, grid='integrator',refine = 500)

ts = np.linspace(0,2*np.pi,1000)
xspline_obs = np.array(obs_spline_x(s1))
yspline_obs = np.array(obs_spline_y(s1))
rspline_obs = np.array(obs_spline_r(s1))
for i in range(s1.shape[0]): plt.plot(xspline_obs[i]+rspline_obs[i]*cos(ts),yspline_obs[i]+rspline_obs[i]*sin(ts),'r-',markersize = 0.5)

ts, s2 = sol.sample(s_path, grid='integrator',refine = 200)
ts = np.linspace(0,2*np.pi,1000)
xspline_path = np.array(path_spline_x(s2))
yspline_path = np.array(path_spline_y(s2))
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



















