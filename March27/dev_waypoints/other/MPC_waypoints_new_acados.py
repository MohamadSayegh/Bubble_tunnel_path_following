# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:20:18 2021

@author: Mohammed
"""



import time as tmp
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting


from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, get_bubbles_mpc_loop
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import create_tunnel
from MPC_Bubble_tunnel_generation_functions import generate_bubbles_mpc_v3


#------------- use sqaures or circles as bubbles ?

use_acados = False
# use_acados = True


use_squares = False

obstacles_option  = 2
path_option       = 2


global_end_goal_x       =    7   
global_end_goal_y       =    9
initial_pos_x           =    0
initial_pos_y           =    0
xlim_min                =   -1     
xlim_max                =    11
ylim_min                =   -1
ylim_max                =    11

   

obs_horizon       = 1000
path_horizon      = 2   

Nsim    = 200           
N       = 10
dt      = 0.7


NB = N
NP = N

#------------- Initialize OCP

ocp = Ocp(T = N*dt)   

#---------------- Initialize grid, occupied positions and bubbles

occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)

global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)


#-----------------------------------

# waypoints_x = [ 0,0.1, 0.5   ,  0.95     , 1.5  , 2.5    , 2.85  ,
#                 3.    , 3.4   , 3.4   , 3.75  , 4.   , 4.25 , 4.8  , 5.5,
#                 6.5  , 6.25   , 6.5   , 6.75  , 6.55   , 7   , 8.     , 9.    ]

# waypoints_y = [ 0,1, 4.0   , 7.5    , 9.8   , 9.8   , 9.8  ,
#                 9.0   , 7.2    , 3.5   , 2.0   , 1.0  , 0.2   ,  0.2   , 0.2   ,
        
#                 0.5  , 3.5    , 5.0   , 6.0   , 8.5  , 9.0   ,  9.0   , 9.0   ]


waypoints_x = np.linspace(0, 9, 10)

waypoints_y = [ 0.0, 5.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]

Bspline_obj, u = interpolate.splprep([waypoints_x,waypoints_y], u = None, s = 0)
u = np.linspace(0,1,10)
wp_path = interpolate.splev(u, Bspline_obj)
waypoints_x = np.array(wp_path[0])
waypoints_y = np.array(wp_path[1])

waypoints_x = waypoints_x[2:]
waypoints_y = waypoints_y[2:]

waypoint = vertcat(waypoints_x[0],waypoints_y[0]) 

# plt.figure(dpi=300)
# plt.title('MPC')    
# plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
# plt.plot(global_path_x, global_path_y, 'g--')
# plt.plot(waypoints_x, waypoints_y, 'go')
# plt.xlim([xlim_min,xlim_max])
# plt.ylim([ylim_min,ylim_max])



shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)



#select N points for path spline
Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
u = np.linspace(0,1,NP)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x = np.array(global_path[0])
global_path_y = np.array(global_path[1])




#---------------- Initialize Logging variables

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

midpoints_x_hist         = np.zeros((Nsim+1, NB))
midpoints_y_hist         = np.zeros((Nsim+1, NB))
radii_x_hist             = np.zeros((Nsim+1, NB))
radii_y_hist             = np.zeros((Nsim+1, NB))
global_path_hist_x       = np.zeros((Nsim+1, NP))
global_path_hist_y       = np.zeros((Nsim+1, NP))

waypoints_hist_x         = np.zeros((Nsim+1, 1))
waypoints_hist_y         = np.zeros((Nsim+1, 1))

#------------------------- System model

x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()

#--------------------------path parameters 

s_obs        =  ocp.state()
sdot_obs     =  ocp.control()

#-----------------------------ODEs

ocp.set_der(x            ,        v*cos(theta))
ocp.set_der(y            ,        v*sin(theta))
ocp.set_der(theta        ,        w)
ocp.set_der(s_obs        ,        sdot_obs)


#-------------------------------------------------------------------------------#
#                            Solve the first iteration                          #
#-------------------------------------------------------------------------------#


#------------------------- Constraints on initial point


X_0 = ocp.parameter(4)
X   = vertcat(x, y, theta, s_obs)

ocp.subject_to(ocp.at_t0(X) == X_0)

current_X = vertcat(initial_pos_x,initial_pos_y,0.0,0.0) 
ocp.set_value(X_0, current_X)


#------------------------- Constraints on Final point

global_goal = vertcat(global_end_goal_x,global_end_goal_y) 


end_goal_x = ocp.parameter(1)
end_goal_y = ocp.parameter(1)

ocp.set_value( end_goal_x, waypoints_x[0])
ocp.set_value( end_goal_y, waypoints_y[0])




#----------------------------- constraints on controls 

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))       
ocp.subject_to( sdot_obs    >=   0)



#---------------------- Obscatles avoidance tunnel 



while len(shifted_midpoints_x) < NB:
    shifted_midpoints_x.append(shifted_midpoints_x[-1])
    shifted_midpoints_y.append(shifted_midpoints_y[-1])
    shifted_radii.append(shifted_radii[-1])

             
shifted_midpoints_x     = shifted_midpoints_x[0:NB]
shifted_midpoints_y     = shifted_midpoints_y[0:NB]
shifted_radii           = shifted_radii[0:NB]


bubbles_x       =  ocp.parameter(NB)
bubbles_y       =  ocp.parameter(NB)
bubbles_radii   =  ocp.parameter(NB)

ocp.set_value(bubbles_x, shifted_midpoints_x)
ocp.set_value(bubbles_y, shifted_midpoints_y)
ocp.set_value(bubbles_radii,   shifted_radii)

tlength1        =  len(shifted_midpoints_x)
tunnel_s1       =  np.linspace(0,1,tlength1) 

ocp.subject_to(ocp.at_tf(s_obs) <= 1)   

obs_spline_x = interpolant('x','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('y','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('r','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#---------------------------------- Path Tunnel avoidance 


# path_x          =  ocp.parameter(NP)
# path_y          =  ocp.parameter(NP)

# ocp.set_value(path_x, global_path_x)
# ocp.set_value(path_y, global_path_y)


# tlength2       =  len(global_path_x)
# tunnel_s2      =  np.linspace(0,1,tlength2) 

# ocp.subject_to(ocp.at_tf(s_path) <= 1)


# path_spline_x = interpolant('x','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
# path_spline_y = interpolant('y','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


# ------------------------ Initial guess ------------------------

if use_acados == False:
    # ------------------------ Initial guess - IPOPT ------------------------
    # we want the initial guesss to be = the global path 

    ocp.set_initial(x,       global_path_x) 
    ocp.set_initial(y,       global_path_y) 

    #path parameters
    s_guess = np.linspace(tunnel_s1[0],tunnel_s1[-3], N)
    sdot_guess = (tunnel_s1[-1]-tunnel_s1[0])/tlength1 

    ocp.set_initial(s_obs, s_guess) 
    ocp.set_initial(sdot_obs, sdot_guess)
    
    v_guess = 0.5*np.ones(N)
    ocp.set_initial(v , v_guess)

    w_guess = np.zeros(N)
    ocp.set_initial(w , w_guess)

else: 
    # ------------------------ Initial guess - ACADOS ------------------------
    # we want the initial guesss to be = the global path 

    Bspline_obj, u = interpolate.splprep([global_path_y,global_path_y], u = None, s = 0)
    u = np.linspace(0,1,N+1)
    path_guess = interpolate.splev(u, Bspline_obj)

    path_guess_x         = np.array(path_guess[0])
    path_guess_y         = np.array(path_guess[1])

    ocp.set_initial(x,       path_guess_x   ) 
    ocp.set_initial(y,       path_guess_y   ) 


    #path parameters
    s_guess = np.linspace(tunnel_s1[0],tunnel_s1[-3], N+1)
    sdot_guess = (tunnel_s1[-1]-tunnel_s1[0])/(tlength1 + 1)

    ocp.set_initial(s_obs, s_guess) 
    ocp.set_initial(sdot_obs, sdot_guess)

    #constraints on control inputs have a slight positive effect on solution time

    v_guess = 0.5*np.ones(N+1)
    ocp.set_initial(v , v_guess)

    w_guess = np.zeros(N+1)
    ocp.set_initial(w , w_guess)





#---------------------------  Obstacle avoidance constraints 

# this tolerance makes the solution possible ====== has to be fixed
tol = 2
ocp.subject_to( ( ( ( x - obs_spline_x(s_obs,bubbles_x) )**2 + ( y-obs_spline_y(s_obs,bubbles_y) )**2 ) <= (tol + obs_spline_r(s_obs,bubbles_radii)**2 ) ) )


# -------------------------------------- Objective function 

#path following

# ocp.add_objective( 1*ocp.integral((x - path_spline_x(s_path, path_x))**2 + (y-path_spline_y(s_path,path_y))**2))   

ocp.add_objective( 1*ocp.integral((x - end_goal_x)**2 + (y-end_goal_y)**2))  


# ----------------- Solver

options = {"ipopt": {"print_level": 5}}
options["expand"] = False
options["print_time"] = True
ocp.solver('ipopt', options)

if use_acados == False:
    # Multiple shooting/ IPOPT
    ocp.method(MultipleShooting(N=N,M=2,intg='rk'))
else: 
    # qp_solvers = ('PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP')
    # integrator_types = ('ERK', 'IRK', 'GNSF', 'DISCRETE')
    # SOLVER_TYPE_values = ['SQP', 'SQP_RTI']
    # HESS_APPROX_values = ['GAUSS_NEWTON', 'EXACT']
    # REGULARIZATION_values = ['NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY']


    # Pick a solution method
    method = external_method('acados', N=N,qp_solver= 'FULL_CONDENSING_HPIPM', nlp_solver_max_iter= 1000, hessian_approx='EXACT', regularize_method = 'MIRROR' ,integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=N)
    ocp.method(method)



#-------------------------------- OCP Solution and Results                             


start_time = tmp.time()

try:
    sol = ocp.solve()
except:
    #failed_to_converge = True
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution


end_time = tmp.time()

delta_time =  end_time - start_time 




#-------------------------------------------------------------------------------#
#                                   MPC                                         #
#-------------------------------------------------------------------------------#


# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing  
t_sol, x_sol            = sol.sample(x,           grid='control')
t_sol, y_sol            = sol.sample(y,           grid='control')
t_sol, theta_sol        = sol.sample(theta,       grid='control')
# t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
t_sol, v_sol            = sol.sample(v,           grid='control')
t_sol, w_sol            = sol.sample(w,           grid='control')
# t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')


# for post processing
time_hist[0,:]          = t_sol
x_hist[0,:]             = x_sol
y_hist[0,:]             = y_sol
theta_hist[0,:]         = theta_sol
# s_path_hist[0,:]        = s_path_sol
s_obs_hist[0,:]         = s_obs_sol
v_hist[0,:]             = v_sol
w_hist[0,:]             = w_sol
# sdot_path_hist[0,:]     = sdot_path_sol
sdot_obs_hist[0,:]      = sdot_obs_sol

midpoints_x_hist[0,:]       = shifted_midpoints_x
midpoints_y_hist[0,:]       = shifted_midpoints_y
radii_x_hist[0,:]           = shifted_radii
radii_y_hist[0,:]           = shifted_radii
global_path_hist_x[0,:]     = global_path_x
global_path_hist_y[0,:]     = global_path_y

waypoints_hist_x[0] = waypoints_x[0]
waypoints_hist_y[0] = waypoints_y[0]        

    

clearance = 0.2

clearance_wp = 0.5

waypoint_indx = 0
    
i = 0
    
time = 0 
    
for i in range(Nsim):
    
    
    print("timestep", i+1, "of", Nsim)
    
        
    error = sumsqr(current_X[0:2] - global_goal)
    if error < clearance: 
        break   #solution reached the global end goal 
        
        
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
      
    
    #------------------- Update initial position ------------------------------
    
    # Combine control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_obs_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    
 
    initial_pos_x = double(current_X[0])
    initial_pos_y = double(current_X[1])
    
    #------------ Update time spent to reach goal 
    
    time = time + (t_sol[1]-t_sol[0])
   
    #------------------------- Generate grid and path -------------------------

    if waypoint_indx + 1 < len(waypoints_x): #if there are waypoints left
    
        new_waypoint = vertcat(waypoints_x[waypoint_indx+1],waypoints_y[waypoint_indx+1]) 
        waypoint = vertcat(waypoints_x[waypoint_indx],waypoints_y[waypoint_indx]) 
        
        dist_waypoint     = sumsqr(current_X[0:2] - waypoint)
        dist_new_waypoint = sumsqr(current_X[0:2] - new_waypoint)
        
        
        if dist_waypoint < clearance_wp or dist_new_waypoint < clearance_wp :
            
            waypoint_indx = waypoint_indx + 1
            waypoint      = new_waypoint     
     
    waypoints_hist_x[i+1] = waypoints_x[waypoint_indx]
    waypoints_hist_y[i+1] = waypoints_y[waypoint_indx]        

    

    global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)

    #----------------- get obstacles ------------------------------------------
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    

    #---------------- Creating the Bubbles-------------------------------------
        
    # global_path_x = np.append(initial_pos_x, global_path_x)
    # global_path_y = np.append(initial_pos_y, global_path_y)

    # if error > 0.1:
    #     Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
    #     u = np.linspace(0,1,50)
    #     global_path = interpolate.splev(u, Bspline_obj)
    #     global_path_x = np.array(global_path[0])
    #     global_path_y = np.array(global_path[1])


    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    # shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    shifted_midpoints_x = np.append(initial_pos_x, shifted_midpoints_x)
    shifted_midpoints_y = np.append(initial_pos_y, shifted_midpoints_y)
    shifted_radii       = np.append(shifted_radii[0], shifted_radii)

    while len(shifted_midpoints_x) < NB:
        shifted_midpoints_x = np.append(shifted_midpoints_x, shifted_midpoints_x[-1] )
        shifted_midpoints_y = np.append(shifted_midpoints_y, shifted_midpoints_y[-1])
        shifted_radii       = np.append(shifted_radii, shifted_radii[-1])
        

    #------------------- Updating Tunnels ------------------------------------

    # global_path_x           = global_path_x[0:NP]
    # global_path_y           = global_path_y[0:NP]
    
    shifted_midpoints_x     = shifted_midpoints_x[0:NB]
    shifted_midpoints_y     = shifted_midpoints_y[0:NB]
    shifted_radii           = shifted_radii[0:NB]

    # ocp.set_value(path_x, global_path_x)
    # ocp.set_value(path_y, global_path_y)
    
    ocp.set_value(bubbles_x, shifted_midpoints_x)
    ocp.set_value(bubbles_y, shifted_midpoints_y)
    ocp.set_value(bubbles_radii,   shifted_radii)
    
    ocp.set_value( end_goal_x, waypoints_x[waypoint_indx])
    ocp.set_value( end_goal_y, waypoints_y[waypoint_indx])
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    
    if use_acados == False:
            
        #------------------ set initial - IPOPT

        ocp.set_initial(s_obs,     s_obs_sol) 
        ocp.set_initial(sdot_obs,  sdot_obs_sol) 

        # ocp.set_initial(x,         global_path_x) 
        # ocp.set_initial(y,         global_path_y) 
        
        # ocp.set_initial(x, shifted_midpoints_x)
        # ocp.set_initial(y, shifted_midpoints_y)
        
        x_guess = np.linspace(initial_pos_x, waypoints_x[waypoint_indx], N)
        y_guess = np.linspace(initial_pos_y, waypoints_y[waypoint_indx], N)
        
        ocp.set_initial(x,          x_guess)
        ocp.set_initial(y,          y_guess)
        
        # ocp.set_initial(x,         x_sol) 
        # ocp.set_initial(y,         y_sol) 
        
        ocp.set_initial(v,         v_sol) 
        ocp.set_initial(w,         w_sol) 
        ocp.set_initial(theta,     theta_sol) 
    
    else:
                
        #------------------ set initial - ACADOS
        
        ocp.set_initial(x,         x_sol) 
        ocp.set_initial(y,         y_sol) 
        ocp.set_initial(theta,     theta_sol) 
        ocp.set_initial(s_obs,     s_obs_sol) 
        ocp.set_initial(sdot_obs,  sdot_obs_sol) 
        ocp.set_initial(v,         v_sol) 
        ocp.set_initial(w,         w_sol) 

    


    #------------------------ Plot results every iteration



    shifted_feasiblebubbles_x, shifted_feasiblebubbles_y = get_bubbles_mpc_loop(global_path_x,global_path_y, x_sol, y_sol,\
                      occupied_positions_x, occupied_positions_y,\
                      xlim_min, xlim_max, ylim_min, ylim_max, shifted_midpoints_x,\
                      shifted_midpoints_y, shifted_radii, use_squares,\
                      x_hist, y_hist, x_sol, y_sol,i)
    
    plt.figure(dpi = 300)
    plt.title('MPC')    
    plt.plot(x_sol, y_sol, 'b-')
    plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
    plt.plot(global_path_x, global_path_y, 'g--')
    plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 5)
    plt.plot(x_sol[0], y_sol[0], 'ro', markersize = 5)
    plt.plot(waypoints_x[waypoint_indx], waypoints_y[waypoint_indx], 'go')
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'ro', markersize = 0.5)
    # plt.legend(['solution of current OCP','obstacles','global path', 'accumulated MPC solution', 'current OCP first shooting point','feasible bubbles'],loc = (0.8,0.3))
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.01)

    
    #------------------------- Solve the optimization problem

    start_time = tmp.time()
    
    if use_acados == False:
        try:
            sol = ocp.solve()
        except:
            #failed_to_converge = True
            ocp.show_infeasibilities(1e-6)
            sol = ocp.non_converged_solution
    else:
        sol = ocp.solve()


    end_time = tmp.time()
    
    delta_time = delta_time + ( end_time - start_time ) 

    #-------------------------- Log data for next iteration  
    
    t_sol, x_sol            = sol.sample(x,           grid='control')
    t_sol, y_sol            = sol.sample(y,           grid='control')
    t_sol, theta_sol        = sol.sample(theta,       grid='control')
    # t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
    t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
    t_sol, v_sol            = sol.sample(v,           grid='control')
    t_sol, w_sol            = sol.sample(w,           grid='control')
    # t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
    t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')
 
    # for post processing
    time_hist[i+1,:]          = t_sol
    x_hist[i+1,:]             = x_sol
    y_hist[i+1,:]             = y_sol
    theta_hist[i+1,:]         = theta_sol
    # s_path_hist[i+1,:]        = s_path_sol
    s_obs_hist[i+1,:]         = s_obs_sol
    v_hist[i+1,:]             = v_sol
    w_hist[i+1,:]             = w_sol
    # sdot_path_hist[i+1,:]     = sdot_path_sol
    sdot_obs_hist[i+1,:]      = sdot_obs_sol
    
    midpoints_x_hist[i+1,:]       = shifted_midpoints_x
    midpoints_y_hist[i+1,:]       = shifted_midpoints_y
    radii_x_hist[i+1,:]           = shifted_radii
    radii_y_hist[i+1,:]           = shifted_radii
    # global_path_hist_x[i+1,:]     = global_path_x
    # global_path_hist_y[i+1,:]     = global_path_y

    
    


# -------------------------------------------
#          Plot the results
# -------------------------------------------



#global path from initial to end point
global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,0,0,1000)
occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,100)
# shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)   

shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)   

tunnel_x, tunnel_y = create_tunnel(shifted_midpoints_x,shifted_midpoints_y,shifted_radii)
       

fig = plt.figure(dpi=300)
ax2 = plt.subplot(1, 1, 1)
ax2.plot(global_path_x, global_path_y, '--')
ax2.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')
ax2.set_title('Interations of OCP solutions')
ax2.plot(x_hist[0:i+1,0], y_hist[0:i+1,0], 'ro')  
for k in range(i):
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')  
plt.legend(['global path','obstacles','MPC final solution','all solution points of every OCP'], loc = (0.8,0.3))
plt.savefig('MPC solution with all ocp iterations', dpi=300)
plt.pause(1)


plt.figure(dpi=300)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 5)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'b-', markersize = 5)
plt.plot(global_end_goal_x , global_end_goal_y ,'bo', markersize = 10)
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
plt.legend(['MPC solution','solution trajectory','end goal',' global path ', 'Obstacles', 'Feasible Bubbles'], loc = "best")
plt.title('MPC Solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-0.5,12])
plt.ylim([-0.2,10.2])
plt.savefig('MPC solution', dpi=300)
plt.pause(1)

plt.figure(dpi=300)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 5)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'b-', markersize = 5)
plt.plot(global_end_goal_x , global_end_goal_y ,'bo', markersize = 10)
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(waypoints_x, waypoints_y ,'k^', markersize = 5)
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
plt.legend(['MPC solution','solution trajectory','end goal',' global path ' ,' global path waypoints', 'Obstacles', 'Feasible Bubbles'], loc = "best")
plt.title('MPC Solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-0.5,12])
plt.ylim([-0.2,10.2])
plt.savefig('MPC solution', dpi=300)
plt.pause(1)



# plt.figure(dpi=300)
# plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
# plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'b-', markersize = 5)
# plt.plot(8.8,9,'bo', markersize = 10)
# plt.plot(global_path_x, global_path_y, 'g--')
# plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
# plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
# plt.legend(['MPC solution','solution trajectory','end goal',' global path ', 'Obstacles', 'Feasible Bubbles'], loc = "best")
# plt.title('MPC Solution')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.xlim([-0.5,12])
# plt.ylim([-0.2,10.2])
# plt.savefig('MPC solution controls', dpi=300)



print("MPC solution time: ", time)
print("Solver execution time: ", delta_time)




# #------------------------ animation

# import matplotlib.animation as animation


# length = i+1

# fig, ax  = plt.subplots(dpi = 300)

# ax = plt.xlabel('x [m]')
# ax = plt.ylabel('y [m]')
# ax = plt.title('MPC solution')

# ts = np.linspace(0,2*np.pi,1000)

# ax = plt.axis([-0.1,10.1,-0.1,10.1])

# Obs, = plt.plot([], [] ,'k.')

# Points,  = plt.plot([], [] ,'r-', markersize = 3)

# Path, = plt.plot([], [] ,'go', markersize = 4)

# Point1,  = plt.plot([], [] ,'r^', markersize = 5)

# Bubbles1, = plt.plot([], [] ,'b.', markersize = 0.5)
# Bubbles2, = plt.plot([], [] ,'b.', markersize = 0.5)
# Bubbles3, = plt.plot([], [] ,'b.', markersize = 0.5)
# Bubbles4, = plt.plot([], [] ,'b.', markersize = 0.5)
# Bubbles5, = plt.plot([], [] ,'b.', markersize = 0.5)

# def animate(i):
    
#     Obs.set_data(occupied_positions_x, occupied_positions_y)
#     Bubbles1.set_data( midpoints_x_hist[i,0] + radii_x_hist[i,0]*np.cos(ts) , midpoints_y_hist[i,0] + radii_y_hist[i,0]*np.sin(ts) )
#     Bubbles2.set_data( midpoints_x_hist[i,1] + radii_x_hist[i,1]*np.cos(ts) , midpoints_y_hist[i,1] + radii_y_hist[i,1]*np.sin(ts) )
#     Bubbles3.set_data( midpoints_x_hist[i,2] + radii_x_hist[i,2]*np.cos(ts) , midpoints_y_hist[i,2] + radii_y_hist[i,2]*np.sin(ts) )
#     Bubbles4.set_data( midpoints_x_hist[i,3] + radii_x_hist[i,3]*np.cos(ts) , midpoints_y_hist[i,3] + radii_y_hist[i,3]*np.sin(ts) )
#     Bubbles5.set_data( midpoints_x_hist[i,4] + radii_x_hist[i,4]*np.cos(ts) , midpoints_y_hist[i,4] + radii_y_hist[i,4]*np.sin(ts) )
   
#     Point1.set_data(x_hist[i,0],y_hist[i,0])
#     Points.set_data(x_hist[i,:],y_hist[i,:])
#     Path.set_data(waypoints_hist_x[i],waypoints_hist_y[i,:])
    
#     return [Obs,Points,Path,Point1, Bubbles1,Bubbles2,Bubbles3,Bubbles4, Bubbles5]  


# myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

# myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    




