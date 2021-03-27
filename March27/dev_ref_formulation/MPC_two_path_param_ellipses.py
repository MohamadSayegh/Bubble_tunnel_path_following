"""


important to set initial guess to global path not to x_sol

if you set initial to global path but the global path extends after the bubbles = shrinking 

the slcak variable is important = less solution time / not stuck in local minina 

"""

import sys

sys.path.append('D:/desktop/New Dev/Functions')



import time as tmp
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting


from MPC_Bubble_tunnel_generation_functions import generate_bubbles_mpc_v3, generate_bubbles_mpc_v2, generate_bubbles_mpc_ellipses
from MPC_helper_functions import create_obstacles_mpc, create_global_path_mpc, create_tunnel, get_bubbles_mpc_loop



#------------- use sqaures or circles as bubbles ?

use_squares = False
# use_squares = True



obstacles_option  = 2
path_option       = 2


global_end_goal_x       =    8   
global_end_goal_y       =    9
initial_pos_x           =    0.1
initial_pos_y           =    0.1
xlim_min                =   -1     
xlim_max                =    11
ylim_min                =   -1
ylim_max                =    11

   

obs_horizon       = 1000
path_horizon      = 2   

Nsim    = 100         
N       = 10
dt      = 0.5     


NB = N
NP = N

#------------- Initialize OCP

ocp = Ocp(T = N*dt)   



#---------------- Initialize grid, occupied positions and bubbles

occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)

global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)

# shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)

midpoints_x, midpoints_y, radii_x, radii_y = generate_bubbles_mpc_ellipses(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
      


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

current_X = vertcat(initial_pos_x,initial_pos_y,0.0,0.0,0.1) 
ocp.set_value(X_0, current_X)


#------------------------- Constraints on Final point

global_goal = vertcat(global_end_goal_x,global_end_goal_y) 


end_goal_x = ocp.parameter(1)
end_goal_y = ocp.parameter(1)

ocp.set_value( end_goal_x, global_path_x[-1])
ocp.set_value( end_goal_y, global_path_y[-1])



#--------- soft end point constraints

# slack_tf_x = ocp.variable()
# slack_tf_y = ocp.variable()

# ocp.subject_to(slack_tf_x >= 0)
# ocp.subject_to(slack_tf_y >= 0)

# ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - end_goal_x) <= slack_tf_x))
# ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - end_goal_y) <= slack_tf_y))

# ocp.add_objective(1*(slack_tf_x + slack_tf_y))



#----------------------------- constraints on controls 

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
ocp.subject_to( sdot_path   >=   0)        
ocp.subject_to( sdot_obs    >=   0)



#---------------------- Obscatles avoidance tunnel 



while len(midpoints_x) < NB:
    midpoints_x.append(midpoints_x[-1])
    midpoints_y.append(midpoints_y[-1])
    radii_x.append(radii_x[-1])
    radii_y.append(radii_y[-1])
        
           
midpoints_x     = midpoints_x[0:NB]
midpoints_y     = midpoints_y[0:NB]
radii_x         = radii_x[0:NB]
radii_y         = radii_y[0:NB]    


bubbles_x         =  ocp.parameter(NB)
bubbles_y         =  ocp.parameter(NB)
bubbles_radii_x   =  ocp.parameter(NB)
bubbles_radii_y   =  ocp.parameter(NB)


ocp.set_value(bubbles_x, midpoints_x)
ocp.set_value(bubbles_y, midpoints_y)
ocp.set_value(bubbles_radii_x,  radii_x)
ocp.set_value(bubbles_radii_y,  radii_y)


tlength1        =  len(midpoints_x)
tunnel_s1       =  np.linspace(0,1,tlength1) 

ocp.subject_to(ocp.at_tf(s_obs) <= 1)

spline_x  = interpolant('x' ,'bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
spline_y  = interpolant('y' ,'bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
spline_rx = interpolant('rx','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
spline_ry = interpolant('ry','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})



#---------------------------------- Path Tunnel avoidance 


path_x          =  ocp.parameter(NP)
path_y          =  ocp.parameter(NP)

ocp.set_value(path_x, global_path_x)
ocp.set_value(path_y, global_path_y)


tlength2       =  len(global_path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 

ocp.subject_to(ocp.at_tf(s_path) <= 1)


path_spline_x = interpolant('x','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('y','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})



# ------------------------ Initial guess ------------------------

# we want the initial guesss to be = the global path 

# global_path_guess_x         = np.array(global_path[0])
# global_path_guess_y         = np.array(global_path[1])


# ocp.set_initial(x,       global_path_guess_x) 
# ocp.set_initial(y,       global_path_guess_y) 

#path parameters
s_guess = np.linspace(tunnel_s1[0],tunnel_s1[-3], N)
sdot_guess = (tunnel_s1[-1]-tunnel_s1[0])/tlength1 

ocp.set_initial(s_path, s_guess) 
ocp.set_initial(sdot_path, sdot_guess)

ocp.set_initial(s_obs, s_guess)
ocp.set_initial(sdot_obs, sdot_guess)

v_guess = 0.5*np.ones(N)
ocp.set_initial(v , v_guess)

w_guess = np.zeros(N)
ocp.set_initial(w , w_guess)


#---------------------------  Obstacle avoidance constraints 

ocp.subject_to( ( (x-spline_x(s_obs, bubbles_x))**2/(spline_rx(s_obs, bubbles_radii_x)**2 ) ) +  ( (y-spline_y(s_obs, bubbles_y))**2/(spline_ry(s_obs, bubbles_radii_y)**2 ) )  <= 1 )
 


# -------------------------------------- Objective function 

#path following

ocp.add_objective( 1*ocp.integral((x - path_spline_x(s_path, path_x))**2 + (y-path_spline_y(s_path,path_y))**2))   


# ----------------- Solver

options = {"ipopt": {"print_level": 0}}
options["expand"] = False
options["print_time"] = True
ocp.solver('ipopt', options)


# Multiple shooting
ocp.method(MultipleShooting(N=N,M=1,intg='rk'))





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
t_sol, s_path_sol       = sol.sample(s_path,      grid='control')
t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
t_sol, v_sol            = sol.sample(v,           grid='control')
t_sol, w_sol            = sol.sample(w,           grid='control')
t_sol, sdot_path_sol    = sol.sample(sdot_path,   grid='control')
t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')


t_sol_ref, x_sol_ref        = sol.sample(x,           grid='integrator', refine = 10)
t_sol_ref, y_sol_ref        = sol.sample(y,           grid='integrator', refine = 10)

# for post processing
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

midpoints_x_hist[0,:]       = midpoints_x
midpoints_y_hist[0,:]       = midpoints_y
radii_x_hist[0,:]           = radii_x
radii_y_hist[0,:]           = radii_y
global_path_hist_x[0,:]     = global_path_x
global_path_hist_y[0,:]     = global_path_y


npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points



clearance = 0.2

    
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
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_path_sol[0], sdot_obs_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    

    
    initial_pos_x = double(current_X[0])
    initial_pos_y = double(current_X[1])
    
    #------------ Update time spent to reach goal 
    
    time = time + (t_sol[1]-t_sol[0])
   
    #------------------------- Generate grid and path -------------------------

    global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)
    

    #----------------- get obstacles ------------------------------------------
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    #---------------- Creating the Bubbles-------------------------------------

    midpoints_x, midpoints_y, radii_x, radii_y = generate_bubbles_mpc_ellipses(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
      


    while len(midpoints_x) < NB:
        midpoints_x.append(midpoints_x[-1])
        midpoints_y.append(midpoints_y[-1])
        radii_x.append(radii_x[-1])
        radii_y.append(radii_y[-1])
            
               
    midpoints_x     = midpoints_x[0:NB]
    midpoints_y     = midpoints_y[0:NB]
    radii_x         = radii_x[0:NB]
    radii_y         = radii_y[0:NB]    

        
    # --------------- select N points for path spline-------------------------
    if error > 0.5:
        Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
        u = np.linspace(0,1,NP)
        global_path = interpolate.splev(u, Bspline_obj)
        global_path_x = np.array(global_path[0])
        global_path_y = np.array(global_path[1])

    #------------------- Updating Tunnels ------------------------------------

    global_path_x           = global_path_x[0:NP]
    global_path_y           = global_path_y[0:NP]
    


    ocp.set_value(path_x, global_path_x)
    ocp.set_value(path_y, global_path_y)
    
    ocp.set_value(bubbles_x, midpoints_x)
    ocp.set_value(bubbles_y, midpoints_y)
    ocp.set_value(bubbles_radii_x,  radii_x)
    ocp.set_value(bubbles_radii_y,  radii_y)
    
    ocp.set_value( end_goal_x, global_path_x[-1])
    ocp.set_value( end_goal_y, global_path_y[-1])
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    
    #------------------ set initial 
    
    ocp.set_initial(s_obs,     s_obs_sol) 
    ocp.set_initial(s_path,    s_path_sol ) 
    ocp.set_initial(sdot_obs,  sdot_obs_sol) 
    ocp.set_initial(sdot_path, sdot_path_sol ) 

    ocp.set_initial(x,         global_path_x) 
    ocp.set_initial(y,         global_path_y) 
    
    # ocp.set_initial(x, shifted_midpoints_x)
    # ocp.set_initial(y, shifted_midpoints_y)
    
    # ocp.set_initial(x,         x_sol) 
    # ocp.set_initial(y,         y_sol) 
    
    ocp.set_initial(v,         v_sol) 
    ocp.set_initial(w,         w_sol) 
    ocp.set_initial(theta,     theta_sol) 


    #------------------------ Plot results every iteration

    ellipse_x = []
    ellipse_y = []
    for it in range(0, len(midpoints_x)):      
        ellipse_x.append(midpoints_x[it] + radii_x[it]*np.cos(ts) )
        ellipse_y.append(midpoints_y[it] + radii_y[it]*np.sin(ts) ) 
        
        
    plt.figure(dpi=300)
    plt.title('MPC')    
    plt.plot(x_sol_ref, y_sol_ref, 'b-')
    plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
    plt.plot(global_path_x, global_path_y, 'g--')
    plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 5)
    plt.plot(x_sol[0], y_sol[0], 'ro', markersize = 5)
    plt.plot(ellipse_x, ellipse_y, 'bo', markersize = 0.5)
    plt.legend(['solution of current OCP','obstacles','global path', 'accumulated MPC solution', 'current OCP first shooting point','feasible bubbles'],loc = (0.8,0.3))
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.01)

    
    #------------------------- Solve the optimization problem

    start_time = tmp.time()
    
    try:
        sol = ocp.solve()
    except:
        #failed_to_converge = True
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution


    end_time = tmp.time()
    
    delta_time = delta_time + ( end_time - start_time ) 

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
 
    t_sol_ref, x_sol_ref        = sol.sample(x,           grid='integrator', refine = 20)
    t_sol_ref, y_sol_ref        = sol.sample(y,           grid='integrator', refine = 20)

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
    
    midpoints_x_hist[i+1,:]       = midpoints_x
    midpoints_y_hist[i+1,:]       = midpoints_y
    radii_x_hist[i+1,:]           = radii_x
    radii_y_hist[i+1,:]           = radii_y
    global_path_hist_x[i+1,:]     = global_path_x
    global_path_hist_y[i+1,:]     = global_path_y

    
    


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



midpoints_x, midpoints_y, radii_x, radii_y = generate_bubbles_mpc_ellipses(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
      
ellipse_x = []
ellipse_y = []
for it in range(0, len(midpoints_x)):      
    ellipse_x.append(midpoints_x[it] + radii_x[it]*np.cos(ts) )
    ellipse_y.append(midpoints_y[it] + radii_y[it]*np.sin(ts) ) 
    
    
plt.figure(dpi=300)
plt.title('MPC')    
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 1.5)
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 5)
plt.plot(8,9,'bo',markersize = 10)
plt.plot(ellipse_x, ellipse_y, 'r.', markersize = 0.2)
plt.legend(['obstacles','global path', 'MPC solution','end goal','feasible bubbles'],loc = (0.4,0.2))
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.savefig('MPC solution ellipses', dpi=300)
   
    
    
    
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




#------------------------ animation

import matplotlib.animation as animation


length = i+1

fig, ax  = plt.subplots(dpi = 300)

ax = plt.xlabel('x [m]')
ax = plt.ylabel('y [m]')
ax = plt.title('MPC solution')

ts = np.linspace(0,2*np.pi,1000)

ax = plt.axis([-0.1,10.1,-0.1,10.1])

Obs, = plt.plot([], [] ,'k.')

Points,  = plt.plot([], [] ,'r-', markersize = 3)

Path, = plt.plot([], [] ,'g--', markersize = 2)

Point1,  = plt.plot([], [] ,'r^', markersize = 5)

Bubbles1, = plt.plot([], [] ,'b.', markersize = 0.5)
Bubbles2, = plt.plot([], [] ,'b.', markersize = 0.5)
Bubbles3, = plt.plot([], [] ,'b.', markersize = 0.5)
Bubbles4, = plt.plot([], [] ,'b.', markersize = 0.5)
Bubbles5, = plt.plot([], [] ,'b.', markersize = 0.5)

def animate(i):
    
    Obs.set_data(occupied_positions_x, occupied_positions_y)
    Bubbles1.set_data( midpoints_x_hist[i,0] + radii_x_hist[i,0]*np.cos(ts) , midpoints_y_hist[i,0] + radii_y_hist[i,0]*np.sin(ts) )
    Bubbles2.set_data( midpoints_x_hist[i,1] + radii_x_hist[i,1]*np.cos(ts) , midpoints_y_hist[i,1] + radii_y_hist[i,1]*np.sin(ts) )
    Bubbles3.set_data( midpoints_x_hist[i,2] + radii_x_hist[i,2]*np.cos(ts) , midpoints_y_hist[i,2] + radii_y_hist[i,2]*np.sin(ts) )
    Bubbles4.set_data( midpoints_x_hist[i,3] + radii_x_hist[i,3]*np.cos(ts) , midpoints_y_hist[i,3] + radii_y_hist[i,3]*np.sin(ts) )
    Bubbles5.set_data( midpoints_x_hist[i,4] + radii_x_hist[i,4]*np.cos(ts) , midpoints_y_hist[i,4] + radii_y_hist[i,4]*np.sin(ts) )
   
    Point1.set_data(x_hist[i,0],y_hist[i,0])
    Points.set_data(x_hist[i,:],y_hist[i,:])
    Path.set_data(global_path_hist_x[i,:],global_path_hist_y[i,:])
    
    return [Obs,Points,Path,Point1, Bubbles1,Bubbles2,Bubbles3,Bubbles4, Bubbles5]  


myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    




