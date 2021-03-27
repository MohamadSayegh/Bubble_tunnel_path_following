

"""

working for path following 

what to do with violations on X0 ?? always low 

fast enough 

could get stuck in local minima 




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
use_squares = False


obstacles_option  = 2
path_option       = 2


global_end_goal_x       =    9     #position of initial and end point
global_end_goal_y       =    9
initial_pos_x           =    0
initial_pos_y           =    0
xlim_min                =   -2     #xlim and ylim of plotsR
xlim_max                =    12
ylim_min                =   -2
ylim_max                =    12

   

obs_horizon       = 30
path_horizon      = 2     

Nsim    = 100            
N       = 5
dt      = 0.5        

#------------- Initialize OCP

ocp = Ocp(T = N*dt)   
    

#---------------- Initialize grid, occupied positions and bubbles

occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)


#------------------------------ waypoints ----------------------------


waypoints_x = [ 0,0.1, 0.5   , 0.8     , 1.5   , 3.4  ,
                 3.6   , 3.6   , 3.8  , 4.   , 4.25 , 4.8  , 5.5,
                6.0  , 6.25   , 6.5   , 6.75  , 6.55   , 7   , 8.     , 9.    ]

waypoints_y = [ 0,1, 4.0   , 7.5    , 9.5  , 9.5  ,
                7.2    , 3.5   , 2.0   , 1.0  , 0.2   ,  0.2   , 0.2   ,
                1  , 3.5    , 5.0   , 6.0   , 8.5  , 9.0   ,  9.0   , 9.0   ]


Bspline_obj, u = interpolate.splprep([waypoints_x,waypoints_y], u = None, s = 0)
u = np.linspace(0,1,20)
wp_path = interpolate.splev(u, Bspline_obj)
waypoints_x = np.array(wp_path[0])
waypoints_y = np.array(wp_path[1])

waypoints_x = waypoints_x[2:]
waypoints_y = waypoints_y[2:]

waypoint = vertcat(waypoints_x[0],waypoints_y[0]) 


#select N points for path spline
Bspline_obj, u = interpolate.splprep([waypoints_x ,waypoints_y], u = None, s = 0)
u = np.linspace(0,1,100)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x = np.array(global_path[0])
global_path_y = np.array(global_path[1])



# plt.figure(dpi=300)
# plt.title('MPC')    
# plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
# plt.plot(global_path_x, global_path_y, 'g--')
# plt.plot(waypoints_x, waypoints_y, 'go')
# plt.xlim([xlim_min,xlim_max])
# plt.ylim([ylim_min,ylim_max])


        
#---------------- Initialize Logging variables

time_hist           = np.zeros((Nsim+1, N+1))
x_hist              = np.zeros((Nsim+1, N+1))
y_hist              = np.zeros((Nsim+1, N+1))
theta_hist          = np.zeros((Nsim+1, N+1))

s_obs_hist          = np.zeros((Nsim+1, N+1))
v_hist              = np.zeros((Nsim+1, N+1))
w_hist              = np.zeros((Nsim+1, N+1))
sdot_obs_hist       = np.zeros((Nsim+1, N+1))




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


pf = ocp.parameter(2)
p_final = vertcat(waypoints_x[0],waypoints_y[0])
ocp.set_value(pf, p_final)

p = vertcat(x,y)

#----------------------------- constraints on controls 

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
  
ocp.subject_to( sdot_obs    >=   0)



#---------------------- Obscatles avoidance tunnel 


shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)


nb = 2

while len(shifted_midpoints_x) < nb:
    shifted_midpoints_x.append(shifted_midpoints_x[-1])
    shifted_midpoints_y.append(shifted_midpoints_y[-1])
    shifted_radii.append(shifted_radii[-1])
               


bubbles_x       =  ocp.parameter(nb)
bubbles_y       =  ocp.parameter(nb)
bubbles_radii   =  ocp.parameter(nb)

shifted_midpoints_x     = shifted_midpoints_x[0:nb]
shifted_midpoints_y     = shifted_midpoints_y[0:nb]
shifted_radii           = shifted_radii[0:nb]


ocp.set_value(bubbles_x, shifted_midpoints_x)
ocp.set_value(bubbles_y, shifted_midpoints_y)
ocp.set_value(bubbles_radii,   shifted_radii)

tlength1        =  len(shifted_midpoints_x)
tunnel_s1       =  np.linspace(0,1,tlength1) 

ocp.subject_to(ocp.at_tf(s_obs) <= 1)   

obs_spline_x = interpolant('x','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_y = interpolant('y','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
obs_spline_r = interpolant('r','bspline',[tunnel_s1], 1  , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})



# -------------------------------- Initial guess 


#path parameters
s_obs_guess = np.linspace(0,1,N)
ocp.set_initial(s_obs, s_obs_guess) 



#---------------------------  Obstacle avoidance constraints 




ocp.subject_to( ( ( ( x - obs_spline_x(s_obs,bubbles_x) )**2 + ( y-obs_spline_y(s_obs,bubbles_y) )**2 ) <= (obs_spline_r(s_obs,bubbles_radii)**2 ) ) )



# -------------------------------------- Objective function 


ocp.add_objective(1*ocp.integral(sumsqr(p-pf)))

# ----------------- Solver

options = {"ipopt": {"print_level": 0}}
options["expand"] = False
options["print_time"] = True
ocp.solver('ipopt', options)


# Multiple shooting
ocp.method(MultipleShooting(N=N,M=2,intg='rk'))



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
# 
t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
t_sol, v_sol            = sol.sample(v,           grid='control')
t_sol, w_sol            = sol.sample(w,           grid='control')

t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')


t_sol_ref, x_sol_ref    = sol.sample(x,           grid='integrator', refine = 10)
t_sol_ref, y_sol_ref    = sol.sample(y,           grid='integrator', refine = 10)


# for post processing
time_hist[0,:]          = t_sol
x_hist[0,:]             = x_sol
y_hist[0,:]             = y_sol
theta_hist[0,:]         = theta_sol

s_obs_hist[0,:]         = s_obs_sol
v_hist[0,:]             = v_sol
w_hist[0,:]             = w_sol

sdot_obs_hist[0,:]      = sdot_obs_sol


clearance_wp = 1
clearance_goal = 0.05
  
i = 0
    
time = 0 
    

waypoint_indx = 0

for i in range(Nsim):
    
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )
    
    
    print("timestep", i+1, "of", Nsim)
    
    
    #------------------- Update initial position ------------------------------
    
    # Combine control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_obs_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
        
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    
    
    initial_pos_x = double(current_X[0])
    initial_pos_y = double(current_X[1])
    
    #------------ Update time spent to reach goal 
    
    time = time + (t_sol[1]-t_sol[0])
   
    #------------------------- Generate grid and path -------------------------

    
    if waypoint_indx + 4 < len(waypoints_x): #if there are waypoints left
        #select N points for path spline
        Bspline_obj, u = interpolate.splprep([waypoints_x[waypoint_indx:] ,waypoints_y[waypoint_indx:]], u = None, s = 0)
        u = np.linspace(0,1,100)
        global_path = interpolate.splev(u, Bspline_obj)
        global_path_x = np.array(global_path[0])
        global_path_y = np.array(global_path[1])

    #----------------- get obstacles ------------------------------------------
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    #---------------- Creating the Bubbles-------------------------------------


    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    while len(shifted_midpoints_x) < N:
        shifted_midpoints_x.append(shifted_midpoints_x[-1])
        shifted_midpoints_y.append(shifted_midpoints_y[-1])
        shifted_radii.append(shifted_radii[-1])
    

    # ---------------- update waypoint --------------
    
    
    if waypoint_indx + 1 < len(waypoints_x): #if there are waypoints left
    
        new_waypoint = vertcat(waypoints_x[waypoint_indx+1],waypoints_y[waypoint_indx+1]) 
        waypoint = vertcat(waypoints_x[waypoint_indx],waypoints_y[waypoint_indx]) 
        
        dist_waypoint     = sumsqr(current_X[0:2] - waypoint)
        dist_new_waypoint = sumsqr(current_X[0:2] - new_waypoint)
        
        
        if dist_waypoint < clearance_wp or dist_new_waypoint < clearance_wp :
            
            waypoint_indx = waypoint_indx + 1
            waypoint      = new_waypoint     
     
            
    
    #------------------- Updating Tunnels ------------------------------------

    p_final = vertcat(waypoints_x[waypoint_indx],waypoints_y[waypoint_indx])
    ocp.set_value(pf, p_final)

    shifted_midpoints_x     = shifted_midpoints_x[0:nb]
    shifted_midpoints_y     = shifted_midpoints_y[0:nb]
    shifted_radii           = shifted_radii[0:nb]


    ocp.set_value(bubbles_x, shifted_midpoints_x)
    ocp.set_value(bubbles_y, shifted_midpoints_y)
    ocp.set_value(bubbles_radii,   shifted_radii)
    
    
    #-------------- reach goal test -----------
    
    error = sumsqr(current_X[0:2] - global_goal)
    if error < clearance_goal: 
        break   #solution reached the global end goal 
        
    

                     
    #----------------- set initial ---------------------
    
    ocp.set_initial(s_obs,      s_obs_sol) 
    ocp.set_initial(sdot_obs,   sdot_obs_sol) 
    
    ocp.set_initial(v,          v_sol)
    ocp.set_initial(w,          w_sol)
    
    ocp.set_initial(theta,      theta_sol) 
 
    # ocp.set_initial(x,          x_sol)
    # ocp.set_initial(y,          y_sol)
    
    x_guess = np.linspace(initial_pos_x, waypoints_x[waypoint_indx], N)
    y_guess = np.linspace(initial_pos_y, waypoints_y[waypoint_indx], N)
    
    ocp.set_initial(x,          x_guess)
    ocp.set_initial(y,          y_guess)
    

    
    #------------------------- Solve the optimization problem -----------

    start_time = tmp.time()
        
    try:
        sol = ocp.solve()
    except:
        #failed_to_converge = True
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution

    end_time = tmp.time()
    
    delta_time = delta_time + ( end_time - start_time ) 

    #-------------------------- Log data -----------------------------
    
    t_sol, x_sol            = sol.sample(x,           grid='control')
    t_sol, y_sol            = sol.sample(y,           grid='control')
    t_sol, theta_sol        = sol.sample(theta,       grid='control')

    t_sol, s_obs_sol        = sol.sample(s_obs,       grid='control')
    t_sol, v_sol            = sol.sample(v,           grid='control')
    t_sol, w_sol            = sol.sample(w,           grid='control')

    t_sol, sdot_obs_sol     = sol.sample(sdot_obs,    grid='control')
 
    t_sol_ref, x_sol_ref        = sol.sample(x,           grid='integrator', refine = 20)
    t_sol_ref, y_sol_ref        = sol.sample(y,           grid='integrator', refine = 20)

    # for post processing
    time_hist[i+1,:]          = t_sol
    x_hist[i+1,:]             = x_sol
    y_hist[i+1,:]             = y_sol
    theta_hist[i+1,:]         = theta_sol
    s_obs_hist[i+1,:]         = s_obs_sol
    v_hist[i+1,:]             = v_sol
    w_hist[i+1,:]             = w_sol
    sdot_obs_hist[i+1,:]      = sdot_obs_sol
    
    #------------------------ Plot results every iteration ---------------------------

    shifted_feasiblebubbles_x, shifted_feasiblebubbles_y = get_bubbles_mpc_loop(global_path_x,global_path_y, x_sol_ref, y_sol_ref,\
                      occupied_positions_x, occupied_positions_y,\
                      xlim_min, xlim_max, ylim_min, ylim_max, shifted_midpoints_x,\
                      shifted_midpoints_y, shifted_radii, use_squares,\
                      x_hist, y_hist, x_sol, y_sol,i)
    
    plt.figure(dpi=300)
    plt.title('MPC')    
    plt.plot(waypoints_x[waypoint_indx], waypoints_y[waypoint_indx], 'go')
    plt.plot(x_sol_ref, y_sol_ref, 'b-')
    plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
    plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
    plt.plot(x_sol[0], y_sol[0], 'ro', markersize = 5)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'ro', markersize = 0.5)
    plt.legend(['solution of current OCP','obstacles', 'accumulated MPC solution', 'current OCP first shooting point','feasible bubbles'],loc = (0.8,0.3))
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.01)
    


    

# -------------------------------------------
#          Plot the results
# -------------------------------------------

#global path from initial to end point
global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,0,0,1000,50)
occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,1000)
shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v3(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)   
tunnel_x, tunnel_y = create_tunnel(shifted_midpoints_x,shifted_midpoints_y,shifted_radii)
       

fig = plt.figure(dpi=300)
ax2 = plt.subplot(1, 1, 1)
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')
ax2.set_title('Interations of OCP solutions')
ax2.plot(x_hist[0:i+1,0], y_hist[0:i+1,0], 'ro')  
for k in range(i):
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')  
plt.legend(['Obstacles','MPC Sol','OCP sol'], loc = (0.75,0.3))
plt.savefig('MPC solution with all ocp iterations', dpi=300)



plt.figure(dpi=300)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'bo', markersize = 3)
plt.plot(x_hist[0:i+1,0],y_hist[0:i+1,0], 'b-', markersize = 5)
plt.plot(8.8,9,'bo', markersize = 10)
plt.plot(waypoints_x, waypoints_y, 'k^')
plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 1)
plt.legend(['MPC solution points','solution trajectory','end goal',' global path waypoints ', 'Feasible Bubbles', 'Obstacles'], loc = (1,1))
plt.title('MPC Solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
plt.savefig('MPC solution', bbox_inches="tight", dpi=300)





print("MPC solution time: ", time)
print("Solver execution time: ", delta_time)







