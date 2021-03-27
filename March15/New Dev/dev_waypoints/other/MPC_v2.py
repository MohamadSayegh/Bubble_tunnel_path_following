"""
Created on FEB 11

Ninja Robot Thesis

@author: Mohamad Sayegh

MPC 

using path parameters


Questions/notes:
    
    1) tried to lower dt. But less than 2 we have a problem with violated 
       constraints. why ?
    
       even if T = N*dt is the same, the problem happens. example : 
           N=10 dt=1 doesnt work but N=5 dt=2 works
        
    2) something doesnt make sense. why does the path keeps curving 
       (no convergence to path)
       = maybe because you have end goal on the end of path!!!!!
       
    3) still have to add warm starting
    
    4) still need to make sure of constraint for square bubbles
    
    5) end point constraints = any other option ?
    
    6) it seems that the solution works weel even without slack variables 
       and no end point constraint, why ?
       
       because of constraint that ocp.at_tf(s) = 1 !!!
        
    7) this instead of slack variables does not work. why ?
       # ocp.add_objective( 1*ocp.integral((x - end_goal_x)**2 + (y-end_goal_x)**2))
       
    8) Need to discuss effect of path segment chosen (its length) 
       and how to effieciently make bubbles based on that
        
"""


import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, plotting, get_bubbles_mpc_loop, generate_bubbles_mpc_v3
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import create_tunnel, plotting_v2



#------------- use sqaures or circles as bubbles ?
use_squares = False
# use_squares = True


# option 2,2 works with N = 5 T = 10
obstacles_option  = 1
path_option       = 1


global_end_goal_x       =    9     #position of initial and end point
global_end_goal_y       =    9
initial_pos_x           =    0
initial_pos_y           =    0
xlim_min                =   -2     #xlim and ylim of plotsR
xlim_max                =    12
ylim_min                =   -2
ylim_max                =    12

   

obs_horizon       = 10
path_horizon      = 3      # less than 3 causes problems (jumps overs the original path)

Nsim    = 100             #max allowed iterations   
N       = 5
dt      = 2               #reducing less than 2 gives constraint violations



#------------- Initialize OCP

ocp = Ocp(T = N*dt)       



#---------------- Initialize grid, occupied positions and bubbles

occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)

global_path_x, global_path_y, Bspline_obj   = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)

shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)


npoints =  500  #numbr of points of every circle
ts      =  np.linspace(0, 2*np.pi, npoints) #for creating circles points
    
shifted_feasiblebubbles_x = []
shifted_feasiblebubbles_y = []
for i in range (0, len(shifted_midpoints_x)):
        shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
        shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))


plt.figure(dpi=300)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(shifted_midpoints_x, shifted_midpoints_y,'bx')
plt.plot(shifted_feasiblebubbles_x,shifted_feasiblebubbles_y,'b.', markersize = 1)
plt.plot(occupied_positions_x,occupied_positions_y,'bo',markersize = 1)
plt.legend(['original path','Obstacles', 'Feasible tunnel', 'OCP solution'])
plt.title('OCP Solutin with given path and tunnel')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('OCP Solution', dpi=300)


#select N points for path spline
Bspline_obj, u = interpolate.splprep([global_path_x,global_path_y], u = None, s = 0)
u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x = np.array(global_path[0])
global_path_y = np.array(global_path[1])



i = len(shifted_midpoints_x)
while len(shifted_midpoints_x) < N:
    shifted_midpoints_x.append(shifted_midpoints_x[-1])
    shifted_midpoints_y.append(shifted_midpoints_y[-1])
    shifted_radii.append(1)
    i = i + 1
               

global_path_x           = global_path_x[0:N]
global_path_y           = global_path_y[0:N]
shifted_midpoints_x     = shifted_midpoints_x[0:N]
shifted_midpoints_y     = shifted_midpoints_y[0:N]
shifted_radii           = shifted_radii[0:N]


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


#if you want end goal to stay the same always == global end goal
# ocp.set_value( end_goal_x, global_end_goal_x)
# ocp.set_value( end_goal_y, global_end_goal_y)



# ------- hard end point constraints

# ocp.subject_to( ocp.at_tf(x) == end_goal_x )
# ocp.subject_to( ocp.at_tf(y) == end_goal_y )


#--------- soft end point constraints

# slack_tf_x = ocp.variable()
# slack_tf_y = ocp.variable()

# ocp.subject_to(slack_tf_x >= 0)
# ocp.subject_to(slack_tf_y >= 0)

# ocp.subject_to(-slack_tf_x <= ((ocp.at_tf(x) - end_goal_x) <= slack_tf_x))
# ocp.subject_to(-slack_tf_y <= ((ocp.at_tf(y) - end_goal_y) <= slack_tf_y))

# ocp.add_objective(10*(slack_tf_x + slack_tf_y))



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

ocp.subject_to(ocp.at_tf(s_obs) <= 1)   

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

#better that == for relaxing condition
ocp.subject_to(ocp.at_tf(s_path) <= 1)


path_spline_x = interpolant('x','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('y','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


# -------------------------------- Initial guess 


#path parameters
s_obs_guess = np.linspace(0,1,N)
s_path_guess = np.linspace(0,1,N)

ocp.set_initial(s_obs, s_obs_guess) 
ocp.set_initial(s_path , s_path_guess )


#constraints on control inputs have a slight positive effect on solution time
ocp.set_initial(v , 0.5)
ocp.set_initial(w , 0.0)


ocp.set_initial(x,       global_path_x) 
ocp.set_initial(y,       global_path_y) 


#---------------------------  Obstacle avoidance constraints 

#to avoid constraint violation at hard corner


#stay in bubbles as much as possible


if use_squares == True:
    # this is wrong
    # ocp.subject_to( (  (  x - obs_spline_x(s_obs,bubbles_x) )**2    <  (tolerance + obs_spline_r(s_obs,bubbles_x)**2)    ))
    # ocp.subject_to( (  (  y - obs_spline_y(s_obs,bubbles_y) )**2    <  (tolerance + obs_spline_r(s_obs,bubbles_y)**2)    ))
    
    tolerance = 0.5
    
    C = cos(pi/4)
    ocp.subject_to(  x  >=  (-tolerance + obs_spline_x(s_obs,bubbles_x) - C*obs_spline_r(s_obs,bubbles_x)))
    ocp.subject_to(  x  <=  (tolerance + obs_spline_x(s_obs,bubbles_x) + C*obs_spline_r(s_obs,bubbles_x)))
    ocp.subject_to(  y  >=  (-tolerance + obs_spline_y(s_obs,bubbles_y) - C*obs_spline_r(s_obs,bubbles_y)))
    ocp.subject_to(  y  <=  (tolerance + obs_spline_y(s_obs,bubbles_y) + C*obs_spline_r(s_obs,bubbles_y)))
    
else:
    tolerance = 0
    ocp.subject_to( ( ( ( x - obs_spline_x(s_obs,bubbles_x) )**2 + ( y-obs_spline_y(s_obs,bubbles_y) )**2 ) <= (tolerance + obs_spline_r(s_obs,bubbles_radii)**2 ) ) )





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


try:
    sol = ocp.solve()
except:
    #failed_to_converge = True
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution





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


clearance = 0.2

    
i = 0
    
time = 0 
    
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
    
    #------------ Update time spent to reach goal 
    
    time = time + (t_sol[1]-t_sol[0])
   
    #------------------------- Generate grid and path -------------------------

    global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,initial_pos_x,initial_pos_y,path_horizon)
    

    #----------------- get obstacles ------------------------------------------
    
    occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,obs_horizon)
    
    #---------------- Creating the Bubbles-------------------------------------


    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)
    
    iter = len(shifted_midpoints_x)
    while len(shifted_midpoints_x) < N:
        shifted_midpoints_x.append(shifted_midpoints_x[-1])
        shifted_midpoints_y.append(shifted_midpoints_y[-1])
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
    
    ocp.set_initial(s_obs, s_obs_guess) 
    ocp.set_initial(s_path , s_path_guess ) 

    #initial guess
    ocp.set_initial(x,       global_path_x) 
    ocp.set_initial(y,       global_path_y) 

    
    #----------------  Simulate dynamic system --------------------------------
    

    
    error = sumsqr(current_X[0:2] - global_goal)
    if error < clearance: 
        break   #solution reached the global end goal 
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    


    #------------------------ Plot results every iteration



    shifted_feasiblebubbles_x, shifted_feasiblebubbles_y = get_bubbles_mpc_loop(global_path_x,global_path_y, x_sol_ref, y_sol_ref,\
                      occupied_positions_x, occupied_positions_y,\
                      xlim_min, xlim_max, ylim_min, ylim_max, shifted_midpoints_x,\
                      shifted_midpoints_y, shifted_radii, use_squares,\
                      x_hist, y_hist, x_sol, y_sol,i)
    
    plt.figure(dpi=300)
    plt.title('MPC')    
    plt.plot(x_sol_ref, y_sol_ref, 'b-')
    for count in range(0, len(shifted_feasiblebubbles_x)):
        plt.plot(shifted_feasiblebubbles_x[count], shifted_feasiblebubbles_y[count], 'ro', markersize = 0.5)
    plt.plot(occupied_positions_x,occupied_positions_y,'co',markersize = 1.5)
    # for count2 in range(0,len(s_path_sol)-1):
    #     plt.plot(path_spline_x(s_path_sol[count2], global_path_x[count2]),path_spline_y(s_path_sol[count2], global_path_y[count2]),'ko')
    plt.plot(global_path_x, global_path_y, 'g--')
    plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
    plt.plot(x_sol[0], y_sol[0], 'bo', markersize = 5)
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([ylim_min,ylim_max])
    plt.pause(0.001)

    
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
    


    

# -------------------------------------------
#          Plot the results
# -------------------------------------------

#global path from initial to end point
global_path_x, global_path_y, Bspline_obj = create_global_path_mpc(path_option,0,0,1000)
occupied_positions_x , occupied_positions_y = create_obstacles_mpc(obstacles_option,initial_pos_x,initial_pos_y,100)
shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x, global_path_y,occupied_positions_x,occupied_positions_y)   
tunnel_x, tunnel_y = create_tunnel(shifted_midpoints_x,shifted_midpoints_y,shifted_radii)
       

fig = plt.figure(dpi=300)
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



plt.figure(dpi=300)
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'b-', markersize = 5)
plt.plot(8.8,9,'bo', markersize = 10)
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


plt.figure(dpi=300)
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'bo', markersize = 5)
plt.plot(x_hist[0:i,0],y_hist[0:i,0], 'b-', markersize = 5)
plt.plot(8.8,9,'bo', markersize = 10)
plt.plot(global_path_x, global_path_y, 'g--')
plt.plot(occupied_positions_x,occupied_positions_y,'ko',markersize = 2)
plt.plot(tunnel_x, tunnel_y, 'ro', markersize = 1)
plt.legend(['MPC solution','solution trajectory','end goal',' global path ', 'Obstacles', 'Feasible Bubbles'], loc = "best")
plt.title('MPC Solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-0.5,12])
plt.ylim([-0.2,10.2])
plt.savefig('MPC solution controls', dpi=300)



print("MPC solution time: ", time)






