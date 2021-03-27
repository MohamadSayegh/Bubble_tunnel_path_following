"""
Created on Thu Jan 28 23:35:27 2021

@author: Mohamad Sayegh & Elias Rached

"""


from rockit import *
from rockit import FreeTime, MultipleShooting, Ocp
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr, Function, exp, vcat, sum1
from function_create_trajectory import create_trajectory
from scipy import interpolate
from update_obstacles_position import update_obs



#--------------- Problem parameters-------------------------------------

Nsim    = 150             # how much samples to simulate in total (affect sampling time?)
nx      = 8               # the system is composed of 8 states
nu      = 4               # the system has 4 control inputs
N       = 20              # number of control intervals = the horizon for multipls shooting
dt      = 0.05            # time interval


xf = 0.5
yf = 0.6
zf = 0.5

#-------------------- Logging variables---------------------------------

time_hist           = np.zeros((Nsim+1, N+1))
x_hist              = np.zeros((Nsim+1, N+1))
y_hist              = np.zeros((Nsim+1, N+1))
theta_hist          = np.zeros((Nsim+1, N+1))
v_hist              = np.zeros((Nsim+1, N+1))
w_hist              = np.zeros((Nsim+1, N+1))



#------------ initialize OCP -------------

ocp = Ocp(T = N*dt)

#------------------------- System model

x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()



#Specify ODEs
ocp.set_der(x            ,        v*cos(theta))
ocp.set_der(y            ,        v*sin(theta))
ocp.set_der(theta        ,        w)



#------------------------------- Control constraints ----------------------

ocp.subject_to(  0          <= ( v  <= 1   ))
ocp.subject_to( -pi         <= ( w  <= pi  ))


# ------------------------ Add obstacles -----------------------------------



#round obstacles
p0 = ocp.parameter(2)
x0 = 0.2
y0 = 0.3
p0_coord = vertcat(x0,y0)
ocp.set_value(p0, p0_coord)
r0 = 0.15          

p1 = ocp.parameter(2)
x1 = 0.2
y1 = 0.8
p1_coord = vertcat(x1,y1)
ocp.set_value(p1, p1_coord)
r1 = 0.15

p2 = ocp.parameter(2)
x2 = 0.5
y2 = 0.3
p2_coord = vertcat(x2,y2)
ocp.set_value(p2, p2_coord)
r2 = 0.1

p3 = ocp.parameter(2)
x3 = 0.8
y3 = 0.2
p3_coord = vertcat(x3,y3)       
ocp.set_value(p3, p3_coord)
r3 = 0.1

p4 = ocp.parameter(2)
x4 = 0.8
y4 = 0.8
p4_coord = vertcat(x4,y4)        
ocp.set_value(p4, p4_coord)
r4 = 0.1



p = vertcat(x,y)



#-------------------------- Constraints -----------------------------------



X_0 = ocp.parameter(3)
X   = vertcat(x, y, theta)

ocp.subject_to(ocp.at_t0(X) == X_0)

current_X = vertcat(0.0,0.0,0.0) 
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


#---------------- constraints on velocity ---------------------------------

v_final = vertcat(0,0)

ocp.subject_to(ocp.at_tf(v) == 0)
ocp.subject_to(ocp.at_tf(w) == 0)

#------------------- Potential field -----------------


X = [x0,x1,x2,x3,x4]
Y = [y0,y1,y2,y3,y4]
R = [r0,r1,r2,r3,r4]


ox  =  ocp.parameter(len(X), 1)
oy  =  ocp.parameter(len(Y), 1)
ro  =  ocp.parameter(len(R), 1)

so =  + 0.01   #affects the spread of function #important for small obstacles

e = 1

we = ocp.parameter(1)
ocp.set_value(we,  e)

g = we*sum1(exp(- ((x - ox )**2/((ro+so)**2)) - ((y - oy)**2/((ro+so)**2))))

costf = Function('costf',[ox,oy,ro,x,y,we],[g])

ocp.set_value(ox,  X)
ocp.set_value(oy,  Y)
ocp.set_value(ro , R)


potential_field = np.zeros((100,100))

    
for i in range(0,100):
    for j in range(0,100):     
        
        potential_field[i,j] = costf( X , Y , R, 0.01*i , 0.01*j, 1)
      
plt.figure(dpi = 300)        
plt.imshow(potential_field.T, cmap='hot', interpolation = 'none', origin='lower')
ts = np.linspace(0,2*pi,1000)

plt.plot(100*p0_coord[0]+100*r0*cos(ts),100*p0_coord[1]+100*r0*sin(ts),'r-')
plt.plot(100*p1_coord[0]+100*r1*cos(ts),100*p1_coord[1]+100*r1*sin(ts),'b-')
plt.plot(100*p2_coord[0]+100*r2*cos(ts),100*p2_coord[1]+100*r2*sin(ts),'g-')
plt.plot(100*p3_coord[0]+100*r3*cos(ts),100*p3_coord[1]+100*r3*sin(ts),'c-')
plt.plot(100*p4_coord[0]+100*r4*cos(ts),100*p4_coord[1]+100*r4*sin(ts),'k-')
plt.plot(100*xf,100*yf,'bo', markersize = 10)
plt.xlim([0,100])
plt.ylim([0,100])
plt.xlabel('x [cm]')
plt.ylabel('y [cm]')
plt.title('Potential Field')    

print(potential_field[int(100*xf)  , int(100*yf) ]) #value at xf,yf

#---------------------- objectives ------------------------------



# weights
w0 = 50
w1 = 25
w2 = 1e-6
w3 = 1


ocp.add_objective(w0*ocp.integral(g))

ocp.add_objective(w1*ocp.integral(sumsqr(p-pf)))

ocp.add_objective(w2*ocp.integral(sumsqr(v + w)))

ocp.add_objective(w3*(slack_tf_x**2 + slack_tf_y**2))    




# to evaluate objective function 
obj =   w0*sum1( 10*exp(- ((x - ox )**2/((ro+so)**2)) - ((y - oy)**2/((ro+so)**2))))+ \
        w1*ocp.integral(sumsqr(p-pf)) + \
        w2*ocp.integral(sumsqr(v + w)) + \
        w3*(slack_tf_x + slack_tf_y)




#-------------------------  Pick a solution method: ipopt --------------------

options = {"ipopt": {"print_level": 0}}
# options = {'ipopt': {"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}}
options["expand"] = True
options["print_time"] = True
ocp.solver('ipopt', options)



#-------------------------- try other solvers here -------------------


# Multiple Shooting
ocp.method(MultipleShooting(N=N, M=2, intg='rk') )


#-------------------- Set initial-----------------


v_guess = 0.5*np.ones(N)
ocp.set_initial(v , v_guess)

w_guess = np.zeros(N)
ocp.set_initial(w , w_guess)



#---------------- Solve the OCP for the first time step--------------------

# First waypoint is current position
index_closest_point = 0


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
t_sol, v_sol            = sol.sample(v,           grid='control')
t_sol, w_sol            = sol.sample(w,           grid='control')


t_sol, sx_sol      = sol.sample(slack_tf_x,        grid='control')
t_sol, sy_sol      = sol.sample(slack_tf_y,        grid='control')


time_hist[0,:]          = t_sol
x_hist[0,:]             = x_sol
y_hist[0,:]             = y_sol
theta_hist[0,:]         = theta_sol
v_hist[0,:]             = v_sol
w_hist[0,:]             = w_sol





#------------------ plot function------------------- 


def plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist_1, y_hist_1, opt, x_sol, y_sol):
    
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
    plt.plot(p0_coord[0]+r0*cos(ts),p0_coord[1]+r0*sin(ts),'r-')
    plt.plot(p1_coord[0]+r1*cos(ts),p1_coord[1]+r1*sin(ts),'b-')
    plt.plot(p2_coord[0]+r2*cos(ts),p2_coord[1]+r2*sin(ts),'g-')
    plt.plot(p3_coord[0]+r3*cos(ts),p3_coord[1]+r3*sin(ts),'c-')
    plt.plot(p4_coord[0]+r4*cos(ts),p4_coord[1]+r4*sin(ts),'k-')
    plt.plot(xf,yf,'ro', markersize = 10)
    
    if opt == 1:
        plt.plot(x_sol, y_sol, 'go' )
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
        
    else:
        plt.plot(x_hist[:,0], y_hist[:,0], 'bo', markersize = 3)
  
    plt.show(block=True)
    




#----------------- Simulate the MPC solving the OCP ----------------------

clearance_obj = 1e-5
clearance_v = 1  
clearance = 1e-2
local_min_clearance = 1e-1
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
Dx = [1 , 1 ,-1 ,1 ,1 ]
Dy = [1 , 1 ,-1 ,1 ,1 ]



t_sol, new_obj = sol.sample(obj , grid = 'control')

new_obj[0] = 10

while True:
    

    
    print("timestep", i+1, "of", Nsim)
    
    print( f' x: {current_X[0]}' )
    print( f' y: {current_X[1]}' )

    
    plotxy(p0_coord, p1_coord, p2_coord, p3_coord, p4_coord, x_hist[0:i,0], y_hist[0:i,0], 1, x_sol, y_sol)

   
    # Combine first control inputs
    current_U = vertcat(v_sol[0], w_sol[0])
    
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T = dt)["xf"]
    
    t_tot = t_tot + dt


    
    #---------------- dynamic obstacles

    X,Y,Dx,Dy =  update_obs(X,Y,R,Dx,Dy)
    
    ocp.set_value(ox,  X)
    ocp.set_value(oy,  Y)
    ocp.set_value(ro , R)


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



    error_v = sumsqr(current_X[2:4] - v_final)
 
    if intermediate_points_required:
        error = sumsqr(current_X[0:2] - intermediate_points[intermediate_points_index-1])
    else:
        error   = sumsqr(current_X[0:2] - p_final)

    if is_stuck or i == Nsim:
        break

    if intermediate_points_index == len(intermediate_points):  #going to end goal
        clearance = 1e-4
    else:
        clearance = 1e-2
        
        
    if abs( y_sol[0] - yf ) < 1e-1 and abs( x_sol[0] - xf ) < 1e-1:  #turn off potential field function
        ocp.set_value(we,  1e-5)
    
    if error < clearance:
        
        if intermediate_points_index == len(intermediate_points):
            print('Location reached')
            break
            
        else:
            
            print('Intermediate point reached! Diverting to next point.')
            
            intermediate_points_index = intermediate_points_index + 1
            ocp.set_value(pf, vcat(intermediate_points[intermediate_points_index-1]))
            
   
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Solve the optimization problem
    try:
        sol = ocp.solve()
    except:
        ocp.show_infeasibilities(1e-6)
        sol = ocp.non_converged_solution
        break

    # Log data for post-processing  
    t_sol, x_sol            = sol.sample(x,           grid='control')
    t_sol, y_sol            = sol.sample(y,           grid='control')
    t_sol, theta_sol        = sol.sample(theta,       grid='control')
    t_sol, v_sol            = sol.sample(v,           grid='control')
    t_sol, w_sol            = sol.sample(w,           grid='control')

    t_sol_ref, x_sol_ref        = sol.sample(x,           grid='integrator', refine = 20)
    t_sol_ref, y_sol_ref        = sol.sample(y,           grid='integrator', refine = 20)
    
    t_sol, sx_sol      = sol.sample(slack_tf_x,        grid='control')
    t_sol, sy_sol      = sol.sample(slack_tf_y,        grid='control')

    
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
    v_hist[i+1,:]             = v_sol
    w_hist[i+1,:]             = w_sol

    
    #--------------------- Initial guess
    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(w, w_sol)

    
    i = i+1
    
    obj_old = new_obj[0]

    t_sol, new_obj = sol.sample(obj , grid = 'control')
    
    
    trajectory_x[i,:] = np.zeros(10)
    trajectory_y[i,:] = np.zeros(10)
    

    if (abs(new_obj[0] - obj_old) < clearance_obj) and (error >= local_min_clearance):
        
        psx = [p0_coord[0], p1_coord[0], p2_coord[0], p3_coord[0], p4_coord[0]]
        psy = [p0_coord[1], p1_coord[1], p2_coord[1], p3_coord[1], p4_coord[1]]
        rs = [r0, r1, r2, r3, r4]

        pathx, pathy = create_trajectory(x_hist[i,0], y_hist[i,0], xf, yf, psx, psy, rs)

        pathx = np.flip(pathx)
        pathy = np.flip(pathy)
    
        
        for index in range(1,len(pathx) - 1):
            if ((pathx[index] < pathx[index - 1]) and (pathx[index] < pathx[index + 1])) or ((pathx[index] > pathx[index - 1]) and (pathx[index] > pathx[index + 1])) or ((pathy[index] < pathy[index - 1]) and (pathy[index] < pathy[index + 1])) or ((pathy[index] > pathy[index - 1]) and (pathy[index] > pathy[index + 1])):
                intermediate_points.append([pathx[index], pathy[index], zf])

        if not intermediate_points:
            print('No intermediate points found. Drone stuck in local minimum.')
            is_stuck = True
        else:
            print('Intermediate points found. Drone leaving local minimum.')
            intermediate_points.append([xf,yf,zf])
            intermediate_points_required = True
            intermediate_points_index = 1
            ocp.set_value(pf, vcat(intermediate_points[intermediate_points_index-1]))
            obj_old = 10
            
            #to fill trajectory
            Bspline_obj, u = interpolate.splprep([pathx,pathy], u = None, s = 0)
            u = np.linspace(0,1,10)
            path = interpolate.splev(u, Bspline_obj)
            trajectory_x[i,:] = path[0]
            trajectory_y[i,:] = path[1]
            



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




#------------------------ animation

import matplotlib.animation as animation


length = i

fig, ax  = plt.subplots(dpi = 300)

ax = plt.xlabel('x [m]')
ax = plt.ylabel('y [m]')
ax = plt.title('MPC solution')

ts = np.linspace(0,2*pi,1000)

ax = plt.axis([0,1,0,1])

P0  = plt.plot(xf, yf ,'o', markersize = 10)

O1, = plt.plot([], [] ,'g-')
O2, = plt.plot([], [] ,'k-')
O3, = plt.plot([], [] ,'c-')
O4, = plt.plot([], [] ,'-')
O5, = plt.plot([], [] ,'-')
P,  = plt.plot([], [] ,'bo', markersize = 5)
Px, = plt.plot([], [] ,'r--', markersize = 2)

def animate(i):
    
    O1.set_data(obs_hist_0[i,0]+obs_hist_0[i,2]*cos(ts), obs_hist_0[i,1]+obs_hist_0[i,2]*sin(ts))
    O2.set_data(obs_hist_1[i,0]+obs_hist_1[i,2]*cos(ts), obs_hist_1[i,1]+obs_hist_1[i,2]*sin(ts))
    O3.set_data(obs_hist_2[i,0]+obs_hist_2[i,2]*cos(ts), obs_hist_2[i,1]+obs_hist_2[i,2]*sin(ts))
    O4.set_data(obs_hist_3[i,0]+obs_hist_3[i,2]*cos(ts), obs_hist_3[i,1]+obs_hist_3[i,2]*sin(ts))
    O5.set_data(obs_hist_4[i,0]+obs_hist_4[i,2]*cos(ts), obs_hist_4[i,1]+obs_hist_4[i,2]*sin(ts))
    P.set_data(x_hist[i,0],y_hist[i,0])
    Px.set_data(x_hist[i,:],y_hist[i,:])
    
    return [O1,O2,O3,O4,O5,P,Px]  


myAnimation = animation.FuncAnimation(fig, animate, frames=length, interval=700, blit=True)

myAnimation.save('MPC_simulation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    
    
    
    
    
    

