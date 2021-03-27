"""

@author: Mohamad Sayegh 

"""


from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan, square
from function_create_trajectory import create_trajectory, get_occupied_positions
from scipy import interpolate
from update_obstacles_position import update_obs
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from casadi import *
from pylab import *
from rockit import Ocp , FreeTime, MultipleShooting
from MPC_Bubble_tunnel_generation_v2 import generate_bubbles_mpc_v2, get_bubbles_mpc_loop
from MPC_Grid_generation import create_obstacles_mpc, create_global_path_mpc
from Bubble_tunnel_generation_v2 import create_tunnel
from MPC_Bubble_tunnel_generation_functions import generate_bubbles_mpc_v3

#--------------- Problem parameters-------------------------------------

Nsim    = 150           
N       = 10           
dt      = 0.05             


xf = 0.5
yf = 0.6
zf = 0.5

NB = N 
NP = N

#-------------------- Logging variables---------------------------------

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


#------------ initialize OCP -------------

ocp = Ocp(T = N*dt)

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

#------------------------------- Control constraints ----------------------

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))
ocp.subject_to( sdot_path   >=   0)      
ocp.subject_to( sdot_obs    >=   0)

# ------------------------ Add obstacles -----------------------------------



#round obstacles
p0 = ocp.parameter(2)
x0 = 0.2
y0 = 0.3
p0_coord = vertcat(x0,y0)
ocp.set_value(p0, p0_coord)
r0 = 0.1       

p1 = ocp.parameter(2)
x1 = 0.2
y1 = 0.8
p1_coord = vertcat(x1,y1)
ocp.set_value(p1, p1_coord)
r1 = 0.1

p2 = ocp.parameter(2)
x2 = 0.5
y2 = 0.3
p2_coord = vertcat(x2,y2)
ocp.set_value(p2, p2_coord)
r2 = 0.05

p3 = ocp.parameter(2)
x3 = 0.8
y3 = 0.2
p3_coord = vertcat(x3,y3)       
ocp.set_value(p3, p3_coord)
r3 = 0.05

p4 = ocp.parameter(2)
x4 = 0.8
y4 = 0.8
p4_coord = vertcat(x4,y4)        
ocp.set_value(p4, p4_coord)
r4 = 0.05



p = vertcat(x,y)



#-------------------------- Constraints -----------------------------------



X_0 = ocp.parameter(5)
X   = vertcat(x, y, theta, s_path, s_obs)

ocp.subject_to(ocp.at_t0(X) == X_0)

current_X = vertcat(0.0,0.0,0.0,0.0,0.0) 
ocp.set_value(X_0, current_X)


#----------------- reach end point  ------------------------------------


pf = ocp.parameter(2)
p_final = vertcat(xf,yf)
ocp.set_value(pf, p_final)


slack_tf_x = ocp.variable()
slack_tf_y = ocp.variable()


ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)


ocp.subject_to((ocp.at_tf(x) - pf[0]) <= slack_tf_x)
ocp.subject_to((ocp.at_tf(y) - pf[1]) <= slack_tf_y)



#------------------- Path following ---------------

psx = [p0_coord[0], p1_coord[0], p2_coord[0], p3_coord[0], p4_coord[0]]
psy = [p0_coord[1], p1_coord[1], p2_coord[1], p3_coord[1], p4_coord[1]]
rs  = [r0, r1, r2, r3, r4]
pathx, pathy = create_trajectory(0, 0, xf, yf, psx, psy, rs)
pathx = np.flip(pathx)
pathy = np.flip(pathy)


#select N points for path spline
Bspline_obj, u = interpolate.splprep([pathx,pathy], u = None, s = 0)
u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x = np.array(global_path[0])
global_path_y = np.array(global_path[1])


path_x          =  ocp.parameter(N)
path_y          =  ocp.parameter(N)

ocp.set_value(path_x, global_path_x)
ocp.set_value(path_y, global_path_y)

tlength2       =  len(global_path_x)
tunnel_s2      =  np.linspace(0,1,tlength2) 

ocp.subject_to(ocp.at_tf(s_path) <= 1)


path_spline_x = interpolant('x','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('y','bspline', [tunnel_s2], 1   , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


#-------------------- feasible tunnel -----------------------------------------



occupied_positions_x, occupied_positions_y =  get_occupied_positions(psx, psy, rs)

Bspline_obj, u = interpolate.splprep([pathx,pathy], u = None, s = 0)
u = np.linspace(0,1,100)
global_path = interpolate.splev(u, Bspline_obj)
global_path_x_b = np.array(global_path[0])
global_path_y_b = np.array(global_path[1])


shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x_b, global_path_y_b,occupied_positions_x,occupied_positions_y)


# plt.figure(dpi = 300)
# plt.plot(occupied_positions_x, occupied_positions_y,'ro', markersize = 1)
# plt.plot(global_path_x, global_path_y)
# plt.plot(shifted_midpoints_x, shifted_midpoints_y, 'bo')


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


ocp.subject_to( ( ( ( x - obs_spline_x(s_obs,bubbles_x) )**2 + ( y-obs_spline_y(s_obs,bubbles_y) )**2 ) <= (obs_spline_r(s_obs,bubbles_radii)**2 ) ) )



#---------------------- objectives ------------------------------

# weights
w0 = 10
w1 = 25
w2 = 1e-6
w3 = 0.1



ocp.add_objective(w0*ocp.integral((x - path_spline_x(s_path, path_x))**2 + (y-path_spline_y(s_path,path_y))**2))   

# ocp.add_objective(w1*ocp.integral(sumsqr(p-pf)))

# ocp.add_objective(w2*ocp.integral(sumsqr(v + w)))

ocp.add_objective(w3*(slack_tf_x**2 + slack_tf_y**2))    




#-------------------------  Pick a solution method: ipopt --------------------

options = {"ipopt": {"print_level": 0}}
# options = {'ipopt': {"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}}
options["expand"] = False   # causes erros on this branch
options["print_time"] = True
ocp.solver('ipopt', options)


#-------------------------- try other solvers here -------------------


# Multiple Shooting
ocp.method(MultipleShooting(N=N, M=1, intg='rk') )


#-------------------- Set initial-----------------

s_guess = np.linspace(tunnel_s2[0],tunnel_s2[-3], N)
sdot_guess = (tunnel_s2[-1]-tunnel_s2[0])/tlength2

ocp.set_initial(s_path, s_guess) 
ocp.set_initial(sdot_path, sdot_guess)


v_guess = 0.5*np.ones(N)
ocp.set_initial(v , v_guess)

w_guess = np.zeros(N)
ocp.set_initial(w , w_guess)



#---------------- Solve the OCP for the first time step--------------------


    
# Solve the optimization problem
try:
    sol = ocp.solve()
except:
    ocp.show_infeasibilities(1e-6)
    sol = ocp.non_converged_solution


# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)  



#----------------------- Log data for post-processing---------------------

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

# t_sol, sx_sol           = sol.sample(slack_tf_x,        grid='control')
# t_sol, sy_sol           = sol.sample(slack_tf_y,        grid='control')


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

midpoints_x_hist[0,:]       = shifted_midpoints_x
midpoints_y_hist[0,:]       = shifted_midpoints_y
radii_x_hist[0,:]           = shifted_radii
radii_y_hist[0,:]           = shifted_radii
global_path_hist_x[0,:]     = global_path_x
global_path_hist_y[0,:]     = global_path_y





#------------------ plot function------------------- 


def plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist_1, y_hist_1,\
           shifted_midpoints_x, shifted_midpoints_y, shifted_radii, opt, x_sol, y_sol, global_path_x, global_path_y):
    
    #x-y plot
    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111)

    plt.xlabel('x pos [m]')
    plt.ylabel('y pos [m]')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.title('solution in x,y')
    ax.set_aspect('equal', adjustable='box')
    
    ts = np.linspace(0,2*pi,1000)
    
            
    shifted_feasiblebubbles_x = []
    shifted_feasiblebubbles_y = []
    for i in range (0, len(shifted_midpoints_x)):
            shifted_feasiblebubbles_x.append(shifted_midpoints_x[i] + shifted_radii[i]*np.cos(ts))
            shifted_feasiblebubbles_y.append(shifted_midpoints_y[i] + shifted_radii[i]*np.sin(ts))


    
    plt.plot(p0_coord[0]+r0*cos(ts),p0_coord[1]+r0*sin(ts),'r-')
    plt.plot(p1_coord[0]+r1*cos(ts),p1_coord[1]+r1*sin(ts),'b-')
    plt.plot(p2_coord[0]+r2*cos(ts),p2_coord[1]+r2*sin(ts),'g-')
    plt.plot(p3_coord[0]+r3*cos(ts),p3_coord[1]+r3*sin(ts),'c-')
    plt.plot(p4_coord[0]+r4*cos(ts),p4_coord[1]+r4*sin(ts),'y-')
    plt.plot(xf,yf,'ro', markersize = 10)
    plt.plot(global_path_x, global_path_y, 'g--', markersize = 3)
    plt.plot(shifted_feasiblebubbles_x, shifted_feasiblebubbles_y, 'k.', markersize = 0.5)    
    
    if opt == 1:
        plt.plot(x_sol, y_sol, 'go' )
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
        
    else:
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
  
    plt.show(block=True)
    




#----------------- Simulate the MPC solving the OCP ----------------------

clearance = 1e-3

i = 0

obs_hist_0  = np.zeros((Nsim+1, 3))
obs_hist_1  = np.zeros((Nsim+1, 3))
obs_hist_2  = np.zeros((Nsim+1, 3))
obs_hist_3  = np.zeros((Nsim+1, 3))
obs_hist_4  = np.zeros((Nsim+1, 3))


intermediate_points = []
intermediate_points_required = False
new_path_not_needed = False
intermediate_points_index = 0
is_stuck = False


trajectory_x = np.zeros((Nsim+1, 10))
trajectory_y = np.zeros((Nsim+1, 10))


t_tot = 0

X =  [x0,x1,x2,x3,x4]
Y =  [y0,y1,y2,y3,y4]
R =  [r0,r1,r2,r3,r4]
Dx = [1 ,1 ,1 ,1 ,1 ]
Dy = [1 ,1 ,1 ,1 ,1 ]


while True:
    

    #-------------------- Print and plot -------------------------------
    
    print("timestep", i+1, "of", Nsim)
    
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )

    
    plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist[0:i,0], y_hist[0:i,0],\
           shifted_midpoints_x, shifted_midpoints_y, shifted_radii, 1, x_sol, y_sol, global_path_x, global_path_y)

    # ------------------ SImulate system -------------------------
    # Combine first control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_path_sol[0], sdot_obs_sol[0])
    
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T = dt)["xf"]
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)
    
    t_tot = t_tot + dt

    #---------------- dynamic obstacles --------------------

    X,Y,Dx,Dy =  update_obs(X,Y,R,Dx,Dy)
    
    x0 = X[0]
    x1 = X[1]
    x2 = X[2]
    x3 = X[3]
    x4 = X[4]
    
    
    y0 = Y[0]
    y1 = Y[1]
    y2 = Y[2]
    y3 = Y[3]
    y4 = Y[4]

    
    p0_coord = vertcat(x0,y0)
    ocp.set_value(p0, p0_coord)
   
    p1_coord = vertcat(x1,y1)
    ocp.set_value(p1, p1_coord)
   
    p2_coord = vertcat(x2,y2)
    ocp.set_value(p2, p2_coord)
    
    p3_coord = vertcat(x3,y3)
    ocp.set_value(p3, p3_coord)
    
    p4_coord = vertcat(x4,y4)
    ocp.set_value(p4, p4_coord)


    # ---------------------- update local path -------------------------------
    
    
    
    pos_x = double(current_X[0])
    pos_y = double(current_X[1])
    
    psx = [p0_coord[0], p1_coord[0], p2_coord[0], p3_coord[0], p4_coord[0]]
    psy = [p0_coord[1], p1_coord[1], p2_coord[1], p3_coord[1], p4_coord[1]]
    rs  = [r0, r1, r2, r3, r4]
    
    print(" Running Astar local planner ")
    pathx_new, pathy_new = create_trajectory(pos_x, pos_y, xf, yf, psx, psy, rs)
    
    if len(pathx_new) <= 1:
        print(" ------ No Local Path Found, using previous Path --------- ")
    else:
        pathx = pathx_new
        pathy = pathy_new 
       
    
    pathx = np.flip(pathx)
    pathy = np.flip(pathy)
    
    Bspline_obj, u = interpolate.splprep([pathx,pathy], u = None, s = 0)
    u = np.linspace(0,1,N)
    global_path = interpolate.splev(u, Bspline_obj)
    global_path_x = np.array(global_path[0])
    global_path_y = np.array(global_path[1])
    
    
    ocp.set_value(path_x, global_path_x)
    ocp.set_value(path_y, global_path_y)
    
    global_path_hist_x[i+1,:]     = global_path_x
    global_path_hist_y[i+1,:]     = global_path_y
    
    
    #------------------------ get bubbles -----------------------------------
    
    occupied_positions_x, occupied_positions_y =  get_occupied_positions(psx, psy, rs)
    
    Bspline_obj, u = interpolate.splprep([pathx,pathy], u = None, s = 0)
    u = np.linspace(0,1,100)
    global_path = interpolate.splev(u, Bspline_obj)
    global_path_x_b = np.array(global_path[0])
    global_path_y_b = np.array(global_path[1])
    
    
    shifted_midpoints_x, shifted_midpoints_y, shifted_radii = generate_bubbles_mpc_v2(global_path_x_b, global_path_y_b,occupied_positions_x,occupied_positions_y)

    
    while len(shifted_midpoints_x) < NB:
        shifted_midpoints_x.append(shifted_midpoints_x[-1])
        shifted_midpoints_y.append(shifted_midpoints_y[-1])
        shifted_radii.append(shifted_radii[-1])
    
                 
    shifted_midpoints_x     = shifted_midpoints_x[0:NB]
    shifted_midpoints_y     = shifted_midpoints_y[0:NB]
    shifted_radii           = shifted_radii[0:NB]
    
    ocp.set_value(bubbles_x, shifted_midpoints_x)
    ocp.set_value(bubbles_y, shifted_midpoints_y)
    ocp.set_value(bubbles_radii,   shifted_radii)
    
    #------------------- Obstacle avoidance check ---------------------------
    
    if ( sumsqr(current_X[0:2] - p0_coord)   -  r0**2  ) >= 0: print('outside obs 1') 
    else: print('-------------------------------- Problem! inside obs  1')
    if ( sumsqr(current_X[0:2] - p1_coord)   -  r1**2  ) >= 0: print('outside obs 2') 
    else: print('-------------------------------- Problem! inside obs  2')
    if ( sumsqr(current_X[0:2] - p2_coord)   -  r2**2  ) >= 0: print('outside obs 3') 
    else: print('-------------------------------- Problem! inside obs  3')
    if ( sumsqr(current_X[0:2] - p3_coord)   -  r3**2  ) >= 0: print('outside obs 4') 
    else: print('-------------------------------- Problem! inside obs  4')
    if ( sumsqr(current_X[0:2] - p4_coord)   -  r4**2  ) >= 0: print('outside obs 5') 
    else: print('-------------------------------- Problem! inside obs  5')


    #------------------ reach goal check ---------------------------------


    if is_stuck or i == Nsim:
        break
        
    error   = sumsqr(current_X[0:2] - p_final)
    if error < clearance:
        print('Location reached')
        break
                 


    #--------------------- Initial guess -------------------
    
    # ocp.set_initial(x, x_sol)
    # ocp.set_initial(y, y_sol)
    
    ocp.set_initial(x, global_path_x)
    ocp.set_initial(y, global_path_y)
    
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(s_path, s_path_sol)
    ocp.set_initial(sdot_path, sdot_path_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(w, w_sol)
    
    
    #----------------- Solve the optimization problem -----------------
    try:
        sol = ocp.solve()
    except:
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution


    #-------------------- Log data --------------------------- 
    
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


    # t_sol, sx_sol      = sol.sample(slack_tf_x,        grid='control')
    # t_sol, sy_sol      = sol.sample(slack_tf_y,        grid='control')

    
    obs_hist_0[i,0] = x0
    obs_hist_0[i,1] = y0
    obs_hist_0[i,2] = r0
    
    obs_hist_1[i,0] = x1
    obs_hist_1[i,1] = y1
    obs_hist_1[i,2] = r1
    
    obs_hist_2[i,0] = x2
    obs_hist_2[i,1] = y2
    obs_hist_2[i,2] = r2
    
    obs_hist_3[i,0] = x3
    obs_hist_3[i,1] = y3
    obs_hist_3[i,2] = r3
    
    obs_hist_4[i,0] = x4
    obs_hist_4[i,1] = y4
    obs_hist_4[i,2] = r4
    
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
    
    midpoints_x_hist[i+1,:]       = shifted_midpoints_x
    midpoints_y_hist[i+1,:]       = shifted_midpoints_y
    radii_x_hist[i+1,:]           = shifted_radii
    radii_y_hist[i+1,:]           = shifted_radii

    
    i = i+1
        




    
        
    
global_path_hist_x[i,:]     = global_path_hist_x[i+1,:]
global_path_hist_y[i,:]     = global_path_hist_y[i+1,:]
   

    
    
    


# # ------------------- Results

# print(f'Total execution time is: {t_tot}')

# plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist[0:i,0], y_hist[0:i,0], 0, x_sol, y_sol)


# timestep = np.linspace(0,t_tot, len(ux_hist[0:i,0]))

# fig2 = plt.figure(dpi = 300, figsize=(4,2))
# plt.plot(timestep, ux_hist[0:i,0], "-b", label="ux")
# plt.plot(timestep, uy_hist[0:i,0], "-r", label="uy")
# plt.plot(timestep, uz_hist[0:i,0], "-g", label="uz")
# plt.plot(timestep, uphi_hist[0:i,0], "-k", label="uphi")
# plt.title("Control Inputs")
# plt.ylim(-1.02, 1.02)
# plt.xlabel("Time (s)")
# plt.ylabel("Control Inputs (m/s^2)")
# # plt.legend(loc="upper right")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=True)

# fig3 = plt.figure(dpi = 300, figsize=(4,2))
# plt.plot(timestep, vx_hist[0:i,0], "-b", label="vx")
# plt.plot(timestep, vy_hist[0:i,0], "-r", label="vy")
# plt.plot(timestep, vz_hist[0:i,0], "-g", label="vz")
# plt.plot(timestep, vphi_hist[0:i,0], "-k", label="vphi")
# plt.title("Velocity")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=True)

# fig4 = plt.figure(dpi = 300, figsize=(4,2))
# plt.plot(timestep, x_hist[0:i,0], "b.", label="x")
# plt.plot(timestep, y_hist[0:i,0], "r.", label="y")
# plt.plot(timestep, z_hist[0:i,0], "g.", label="z")
# plt.plot(timestep, phi_hist[0:i,0], "k.", label="phi")
# plt.plot(timestep, yf*np.ones(i),'r--', linewidth = 0.5, label='y goal')
# plt.plot(timestep, zf*np.ones(i),'g--', linewidth = 0.5, label='x and z goal')
# plt.title("Position")
# plt.xlabel("Time (s)")
# plt.ylabel("Positon (m)")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=True)




#------------------------ animation ----------------------------------

import matplotlib.animation as animation


length = i+2

fig, ax  = plt.subplots(dpi = 300)

ax = plt.xlabel('x [m]')
ax = plt.ylabel('y [m]')
ax = plt.title('MPC solution')

ts = np.linspace(0,2*pi,1000)

ax = plt.axis([0,1,0,1])

P0  = plt.plot(xf, yf ,'o', markersize = 10)

O1, = plt.plot([], [] ,'g-', markersize = 5)
O2, = plt.plot([], [] ,'y-', markersize = 5)
O3, = plt.plot([], [] ,'c-', markersize = 5)
O4, = plt.plot([], [] ,'-' , markersize = 5)
O5, = plt.plot([], [] ,'-' , markersize = 5)

P,  = plt.plot([], [] ,'bo', markersize = 5)
Px, = plt.plot([], [] ,'r--', markersize = 2)

Bubbles1,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles2,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles3,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles4,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles5,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles6,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles7,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles8,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles9,  = plt.plot([], [] ,'k.', markersize = 0.2)
Bubbles10, = plt.plot([], [] ,'k.', markersize = 0.2)

GP,        = plt.plot([], [], 'g-', markersize = 1)

def animate(i):
    
    O1.set_data(obs_hist_0[i,0]+obs_hist_0[i,2]*cos(ts), obs_hist_0[i,1]+obs_hist_0[i,2]*sin(ts))
    O2.set_data(obs_hist_1[i,0]+obs_hist_1[i,2]*cos(ts), obs_hist_1[i,1]+obs_hist_1[i,2]*sin(ts))
    O3.set_data(obs_hist_2[i,0]+obs_hist_2[i,2]*cos(ts), obs_hist_2[i,1]+obs_hist_2[i,2]*sin(ts))
    O4.set_data(obs_hist_3[i,0]+obs_hist_3[i,2]*cos(ts), obs_hist_3[i,1]+obs_hist_3[i,2]*sin(ts))
    O5.set_data(obs_hist_4[i,0]+obs_hist_4[i,2]*cos(ts), obs_hist_4[i,1]+obs_hist_4[i,2]*sin(ts))
    
    P.set_data(x_hist[i,0],y_hist[i,0])
    Px.set_data(x_hist[i,:],y_hist[i,:])
    
    Bubbles1.set_data(  midpoints_x_hist[i,0] + radii_x_hist[i,0]*np.cos(ts) , midpoints_y_hist[i,0] + radii_y_hist[i,0]*np.sin(ts) )
    Bubbles2.set_data(  midpoints_x_hist[i,1] + radii_x_hist[i,1]*np.cos(ts) , midpoints_y_hist[i,1] + radii_y_hist[i,1]*np.sin(ts) )
    Bubbles3.set_data(  midpoints_x_hist[i,2] + radii_x_hist[i,2]*np.cos(ts) , midpoints_y_hist[i,2] + radii_y_hist[i,2]*np.sin(ts) )
    Bubbles4.set_data(  midpoints_x_hist[i,3] + radii_x_hist[i,3]*np.cos(ts) , midpoints_y_hist[i,3] + radii_y_hist[i,3]*np.sin(ts) )
    Bubbles5.set_data(  midpoints_x_hist[i,4] + radii_x_hist[i,4]*np.cos(ts) , midpoints_y_hist[i,4] + radii_y_hist[i,4]*np.sin(ts) )
    Bubbles6.set_data(  midpoints_x_hist[i,5] + radii_x_hist[i,5]*np.cos(ts) , midpoints_y_hist[i,5] + radii_y_hist[i,5]*np.sin(ts) )
    Bubbles7.set_data(  midpoints_x_hist[i,6] + radii_x_hist[i,6]*np.cos(ts) , midpoints_y_hist[i,6] + radii_y_hist[i,6]*np.sin(ts) )
    Bubbles8.set_data(  midpoints_x_hist[i,7] + radii_x_hist[i,7]*np.cos(ts) , midpoints_y_hist[i,7] + radii_y_hist[i,7]*np.sin(ts) )
    Bubbles9.set_data(  midpoints_x_hist[i,8] + radii_x_hist[i,8]*np.cos(ts) , midpoints_y_hist[i,8] + radii_y_hist[i,8]*np.sin(ts) )
    Bubbles10.set_data( midpoints_x_hist[i,9] + radii_x_hist[i,9]*np.cos(ts) , midpoints_y_hist[i,9] + radii_y_hist[i,9]*np.sin(ts) )
   
    GP.set_data(global_path_hist_x[i,:],global_path_hist_y[i,:])
    
    return [O1,O2,O3,O4,O5,P,Px, Bubbles1,Bubbles2,Bubbles3,Bubbles4, Bubbles5, Bubbles6,Bubbles7,Bubbles8,Bubbles9, Bubbles10, GP ]  


myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    
    
    

