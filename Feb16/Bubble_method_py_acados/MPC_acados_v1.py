"""
Created on Dec 04

Ninja Robot Thesis

@author: Mohamad Sayegh

MPC 

using path parameters

problems: 
    
   1) path points need to be  less  for path following 
    
      path points need to be more for bubbles
    
      solution = use bspline object wisely         ??
      
   2) set initial inside mpc loop


"""


import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, plotting
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import create_tunnel


#---------------- Need to run this first 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/mohamad/acados/lib"
# export ACADOS_SOURCE_DIR="/home/mohamad/acados"


global_end_goal_x       =    9     #position of initial and end point
global_end_goal_y       =    9
initial_pos_x           =    0
initial_pos_y           =    0
xlim_min                =   -1     #xlim and ylim of plots
xlim_max                =    11
ylim_min                =   -2
ylim_max                =    12
n                       =    10    #size of square grid


# option 2,2 works with N = 5 T = 10
obstacles_option  = 2
path_option       = 2

obs_horizon       = 100
path_horizon      = 1

ocp = Ocp(T = 10.0)       

Nsim    = 10        #max allowed iterations   
N       = 5

occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)

global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon, N)

shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)



#select N points for path spline
Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
u = np.linspace(0,1,N+1)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x = np.array(global_path[0])
global_path_y = np.array(global_path[1])



i = len(shifted_midpoints_x)
while len(shifted_midpoints_x) < (N+1):
    shifted_midpoints_x.append(global_path_x[i])
    shifted_midpoints_y.append(global_path_y[i])
    shifted_radii.append(1)
    i = i + 1
               

# global_path_x           = global_path_x[0:N]
# global_path_y           = global_path_y[0:N]
# shifted_midpoints_x     = shifted_midpoints_x[0:N]
# shifted_midpoints_y     = shifted_midpoints_y[0:N]
# shifted_radii           = shifted_radii[0:N]


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




#------------------------- System model

x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()

#--------------------------path parameters 

s_path       =  ocp.state()
sdot_path    =  ocp.control()

s_obs        =  ocp.state()
sdot_obs     =  ocp.control()

#-----------------------------ODEs

ocp.set_der(x            ,        v*cos(theta))
ocp.set_der(y            ,        v*sin(theta))
ocp.set_der(theta        ,        w)
ocp.set_der(s_path       ,        sdot_path)
ocp.set_der(s_obs        ,        sdot_obs)


#-------------------------------------------------------------------------------#
#                            Solve the first iteration                          #
#-------------------------------------------------------------------------------#


#------------------------- Constraints on initial point


X_0 = ocp.parameter(5)
X   = vertcat(x, y, theta, s_path, s_obs)

ocp.subject_to(ocp.at_t0(X) == X_0)

current_X = vertcat(initial_pos_x,initial_pos_y,0.0,0.0,0.0) 
ocp.set_value(X_0, current_X)


#------------------------- Constraints on Final point

global_goal = vertcat(global_end_goal_x,global_end_goal_y) 




end_goal_x = ocp.parameter(1)
end_goal_y = ocp.parameter(1)

ocp.set_value( end_goal_x, global_path_x[-1])
ocp.set_value( end_goal_y, global_path_y[-1])

pf = vertcat(9,9)

slack_tf_x   = ocp.variable()
slack_tf_y   = ocp.variable()
slack_tf_x_2 = ocp.variable()
slack_tf_y_2 = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)
ocp.subject_to(slack_tf_x_2 >= 0)
ocp.subject_to(slack_tf_y_2 >= 0)

ocp.subject_to(-slack_tf_x_2 <= ((ocp.at_tf(x) - end_goal_x) <= slack_tf_x))
ocp.subject_to(-slack_tf_y_2 <= ((ocp.at_tf(y) - end_goal_y) <= slack_tf_y))

ocp.add_objective( 1*(  slack_tf_x   +  slack_tf_y   ))
ocp.add_objective( 1*(  slack_tf_x_2 +  slack_tf_y_2 ))


#----------------------------- constraints on controls 

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
ocp.subject_to( sdot_path   >=   0)        
ocp.subject_to( sdot_obs    >=   0)



#---------------------- Obscatles avoidance tunnel 

bubbles_x       =  ocp.parameter(1, grid = 'control')
bubbles_y       =  ocp.parameter(1, grid = 'control')
bubbles_radii   =  ocp.parameter(1, grid = 'control')

ocp.set_value(bubbles_x, shifted_midpoints_x)
ocp.set_value(bubbles_y, shifted_midpoints_y)
ocp.set_value(bubbles_radii,   shifted_radii)

tlength1        =  len(shifted_midpoints_x)
tunnel_s1       =  np.linspace(0,1,tlength1) 

# ocp.subject_to(ocp.at_tf(s_obs) == 1)   

obs_spline_x = interpolant('x','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('y','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('r','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#---------------------------------- Path Tunnel avoidance 



path_x          =  ocp.parameter(1, grid = 'control')
path_y          =  ocp.parameter(1, grid = 'control')

ocp.set_value(path_x, global_path_x)
ocp.set_value(path_y, global_path_y)


tlength2       =  len(global_path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 

# ocp.subject_to(ocp.at_tf(s_path) == 1)


path_spline_x = interpolant('x','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('y','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


# -------------------------------- Initial guess 



#path parameters
s_obs_guess = np.linspace(0,1,N+1)
s_path_guess = np.linspace(0,1,N+1)

ocp.set_initial(s_obs, s_obs_guess) 
ocp.set_initial(s_path , s_path_guess )


#constraints on control inputs have a slight positive effect on solution time
ocp.set_initial(v , 0.5)
ocp.set_initial(w , 0.0)



#---------------------------  Obstacle avoidance constraints 

tolerance = 0 

#stay in bubbles as much as possible

ocp.subject_to( (  ( x - obs_spline_x(s_obs,bubbles_x) )**2 + ( y-obs_spline_y(s_obs,bubbles_y) )**2 < (tolerance + obs_spline_r(s_obs,bubbles_radii)**2 )) ) 

# -------------------------------------- Objective function 

#path following

ocp.add_objective( 1*ocp.integral((x - path_spline_x(s_path, path_x))**2 + (y-path_spline_y(s_path,path_y))**2))    #not enough by itself to make path following a priority

# ocp.add_objective(-1*ocp.at_tf(s_path))

# ----------------------------------- Solution method
# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')


# qp_solvers = ('PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP')
# integrator_types = ('ERK', 'IRK', 'GNSF', 'DISCRETE')
# SOLVER_TYPE_values = ['SQP', 'SQP_RTI']
# HESS_APPROX_values = ['GAUSS_NEWTON', 'EXACT']
# REGULARIZATION_values = ['NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY']


# Pick a solution method
method = external_method('acados', N=N,qp_solver= 'FULL_CONDENSING_HPIPM', nlp_solver_max_iter= 500, hessian_approx='EXACT', regularize_method = 'MIRROR' ,integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=N)
ocp.method(method)



# Multiple shooting
# ocp.method(MultipleShooting(N=N,M=3,intg='rk'))


# ----------- initialization of x,y states to global path

ocp.set_initial(x,       global_path_x) 
ocp.set_initial(y,       global_path_y) 


#-------------------------------- OCP Solution and Results                             

sol = ocp.solve()


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


clearance = 0.1


npoints =  100  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints)
    

plt.figure()
plt.title('MPC solution')

    
    
for i in range(Nsim):
    
    
    print("timestep", i+1, "of", Nsim)
    
    
    #------------------- Update initial position ------------------------------
    
    # Combine control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_path_sol[0], sdot_obs_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]

    
    
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
    # print( f' theta: {current_X[2]}' )
    
    initial_pos_x = double(current_X[0])
    initial_pos_y = double(current_X[1])
    
   
    #------------------------- Generate grid and path -------------------------

    global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon, N)
    

    #----------------- get obstacles ------------------------------------------
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    #---------------- Creating the Bubbles-------------------------------------


    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    iter = len(shifted_midpoints_x)
    while len(shifted_midpoints_x) < len(global_path_x):
        shifted_midpoints_x.append(global_path_x[iter])
        shifted_midpoints_y.append(global_path_y[iter])
        shifted_radii.append(1)
        iter = iter + 1
    
    # --------------- select N points for path spline-------------------------
    Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
    u = np.linspace(0,1,N)
    global_path = interpolate.splev(u, Bspline_obj)
    global_path_x = np.array(global_path[0])
    global_path_y = np.array(global_path[1])

    #------------------- Updating Tunnels ------------------------------------

    global_path_x           = global_path_x[0:N]
    global_path_y           = global_path_y[0:N]
    shifted_midpoints_x     = shifted_midpoints_x[0:N]
    shifted_midpoints_y     = shifted_midpoints_y[0:N]
    shifted_radii           = shifted_radii[0:N]

    ocp.set_value(path_x, global_path_x)
    ocp.set_value(path_y, global_path_y)
    
    ocp.set_value(bubbles_x, shifted_midpoints_x)
    ocp.set_value(bubbles_y, shifted_midpoints_y)
    ocp.set_value(bubbles_radii,   shifted_radii)
    
    ocp.set_value( end_goal_x, global_path_x[-1])
    ocp.set_value( end_goal_y, global_path_y[-1])
    
    
    #-------------- initial guess
    ocp.set_initial(x,       global_path_x) 
    ocp.set_initial(y,       global_path_y) 

    
    #----------------  Simulate dynamic system --------------------------------
    

    
    error = sumsqr(current_X[0:2] - global_goal)
    if error < clearance: 
        break   #solution reached the global end goal 
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    


    #------------------------ Plot result
    # shifted_feasiblebubbles_x = []
    # shifted_feasiblebubbles_y = []
    # for k in range (0, len(shifted_midpoints_x)):
    #         shifted_feasiblebubbles_x.append(shifted_midpoints_x[k] + shifted_radii[k]*np.cos(ts))
    #         shifted_feasiblebubbles_y.append(shifted_midpoints_y[k] + shifted_radii[k]*np.sin(ts))

    # plt.plot(x_sol, y_sol, 'ro')
    # plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'ro', markersize = 0.5)
    # plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1.5)
    # plt.plot(global_path_x, global_path_y, 'g--')
    # plt.plot(x_sol[0],y_sol[0], 'bx', markersize = 5)
    # plt.xlim([xlim_min,xlim_max])
    # plt.ylim([ylim_min,ylim_max])
    # plt.pause(0.001)


    #------------------------- Solve the optimization problem

    try:
        sol = ocp.solve()
    except:
        #failed_to_converge = True
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution

    #-------------------------- Log data for next iteration  
    
    t_sol, x_sol            = sol.sample(x,           grid='control')
    t_sol, y_sol            = sol.sample(y,           grid='control')
    t_sol, theta_sol        = sol.sample(theta,       grid='control')
    t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
    t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
    t_sol, v_sol            = sol.sample(v,           grid='control')
    t_sol, w_sol            = sol.sample(w,           grid='control')
    t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
    t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')
 
    
    # for post processing
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
    


    

# -------------------------------------------
#          Plot the results
# -------------------------------------------

#global path from initial to end point
global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,0,0,1000,30)

fig = plt.figure()
ax2 = plt.subplot(1, 1, 1)
ax2.plot(global_path[0], global_path[1], '--')
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
ax2.plot(x_hist[0,0], y_hist[0,0], 'b-')
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')
ax2.set_title('Interations of OCP solutions')
ax2.plot(x_hist[0:i,0], y_hist[0:i,0], 'ro')  
for k in range(i):
    # ax2.plot(x_hist[k,:], y_hist[k,:], 'b-')
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')  
plt.savefig('MPC solution with all ocp iterations', dpi=300)




shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)   
tunnel_x, tunnel_y = create_tunnel(shifted_midpoints_x,shifted_midpoints_y,shifted_radii)
       

plt.figure()
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'b-', markersize = 5)
plt.plot(8.9,9,'bo', markersize = 12)
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 1.5)
plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
plt.legend(['MPC solution points','solution trajectory','end goal',' global path ', 'Obstacles', 'Feasible Bubbles'], loc = "best")
plt.title('MPC Solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.savefig('MPC solution', dpi=300)










