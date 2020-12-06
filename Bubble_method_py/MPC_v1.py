"""
Created on Dec 04

Ninja Robot Thesis

@author: Mohamad Sayegh

first trial of MPC


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

#-------------------------------------------------------------------------------#
#                   Generate Grid and Random Obstacles                          #
#-------------------------------------------------------------------------------#

end_goal_x      =   9;     #position of initial and end point
end_goal_y      =   9;
initial_pos_x   =   0;
initial_pos_y   =   0;
xlim_min        =   -0.5;  #xlim and ylim of plots
xlim_max        =   10.5;
ylim_min        =   -2;
ylim_max        =   12;
n               =   10;    #size of square grid



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



ocp = Ocp(T = 60.0)       

Nsim    = 10            # how much samples to simulate
N       = 20            # number of control intervals


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


#-------------------------------------------------------------------------------#
#                            Solve the first iteration                          #
#-------------------------------------------------------------------------------#



# Constraints on initial point
nx  = 5 #number of states
X_0 = ocp.parameter(nx)
X   = vertcat(x, y, theta, s_path, s_obs)

ocp.subject_to(ocp.at_t0(X) == X_0)
current_X = vertcat(initial_pos_x,initial_pos_y,0.0,0.0,0.0) 
ocp.set_value(X_0, current_X)

# Constraints at the final point  = maybe we hsould not add this in the MPC = every MPC iteration should have another final point!
# ocp.subject_to(ocp.at_tf(x) == end_goal_x)
# ocp.subject_to(ocp.at_tf(y) == end_goal_y)
# ocp.subject_to(ocp.at_tf(theta) == 0.0)


#constraints on controls 
ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
ocp.subject_to( sdot_path   >=   0)        
ocp.subject_to( sdot_obs    >=   0)



#------------ Obscatles avoidance tunnel ---------------------------

bubbles_radii     =  shifted_radii
bubbles_x         =  shifted_midpoints_x
bubbles_y         =  shifted_midpoints_y
tlength1          =  len(bubbles_x)
tunnel_s1         =  np.linspace(0,1,tlength1) 



ocp.subject_to(ocp.at_tf(s_obs) == 1)   


obs_spline_x = interpolant('x','bspline',[tunnel_s1],bubbles_x      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('y','bspline',[tunnel_s1],bubbles_y      , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('r','bspline',[tunnel_s1],bubbles_radii  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#---------------------- Path Tunnel avoidance ----------------------

#re-evaluate the path only in same length as bubbles data


path_x         =  global_path[0]
path_y         =  global_path[1]
tlength2       =  len(path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 



ocp.subject_to(ocp.at_tf(s_path) < 1)

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



#-------------------------------------------------------------------------------#
#                                   MPC                                         #
#-------------------------------------------------------------------------------#



# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing  
t_sol, x_sol            = sol.sample(x,           grid='control')
t_sol, y_sol            = sol.sample(y,           grid='control')
t_sol, theta_sol        = sol.sample(theta,       grid='control')
t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
t_sol, v_sol            = sol.sample(v,           grid='control')
t_sol, w_sol            = sol.sample(w,           grid='control')
t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')

time_hist[0,:]          = t_sol
x_hist[0,:]             = x_sol
y_hist[0,:]             = y_sol
theta_hist[0,:]         = theta_sol
s_path_hist[0,:]        = s_path_sol
s_obs_hist[0,:]         = s_obs_sol
v_hist[0,:]             = v_sol
w_hist[0,:]             = w_sol
sdot_path_hist[0,:]     = sdot_path_sol
sdot_obs_hist[0,:]      = sdot_obs_sol


# Simulate the MPC solving the OCP (with the updated state) several times

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    
    # Combine first control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_path_sol[0], sdot_obs_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing  
    t_sol, x_sol            = sol.sample(x,           grid='control')
    t_sol, y_sol            = sol.sample(y,           grid='control')
    t_sol, theta_sol        = sol.sample(theta,       grid='control')
    t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
    t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
    t_sol, v_sol            = sol.sample(v,           grid='control')
    t_sol, w_sol            = sol.sample(w,           grid='control')
    t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
    t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')
 
    
    time_hist[i+1,:]          = t_sol
    x_hist[i+1,:]             = x_sol
    y_hist[i+1,:]             = y_sol
    theta_hist[i+1,:]         = theta_sol
    s_path_hist[i+1,:]        = s_path_sol
    s_obs_hist[i+1,:]         = s_obs_sol
    v_hist[i+1,:]             = v_sol
    w_hist[i+1,:]             = w_sol
    sdot_path_hist[i+1,:]     = sdot_path_sol
    sdot_obs_hist[i+1,:]      = sdot_obs_sol
    

    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(s_path, s_path_sol)
    ocp.set_initial(s_obs, s_obs_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(w, w_sol)
    ocp.set_initial(sdot_path, sdot_path_sol)
    ocp.set_initial(sdot_obs, sdot_obs_sol)




# -------------------------------------------
#          Plot the results
# -------------------------------------------

T_start = 0
T_end   = sum(time_hist[k,1] - time_hist[k,0] for k in range(Nsim+1))

fig = plt.figure()
ax2 = plt.subplot(1, 1, 1)
ax2.plot(global_path[0], global_path[1], '--')
ax2.plot(occupied_positions_x, occupied_positions_y, 'bo')
ax2.plot(shifted_midpoints_x,shifted_midpoints_y,'rx')
ax2.plot(x_hist[0,0], y_hist[0,0], 'b-')
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')

for k in range(Nsim+1):
    ax2.plot(x_hist[k,:], y_hist[k,:], 'b-')
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')  
    T_start = T_start + (time_hist[k,1] - time_hist[k,0])
    plt.pause(0.5)




















