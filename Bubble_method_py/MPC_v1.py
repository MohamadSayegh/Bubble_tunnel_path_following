import numpy as np; pi = np.pi
from random import randint
import matplotlib.pyplot as plt
from scipy import interpolate
from pylab import *
from casadi import Function, linspace, vertcat, horzcat, DM, interpolant, sum1, MX, hcat, sumsqr
from rockit import *



# notes 1) = here the interpolationpoints are not used since their radii have a problem
#          = using midpoints for now
#       2) = made radius_limit, max radius, and s step smaller

#-------------------------------------------------------------------------------#
#                   Generate Grid and Random Obstacles                          #
#-------------------------------------------------------------------------------#

end_goal_x      =   9;     #position of initial and end point
end_goal_y      =   8;
initial_pos_x   =   0;
initial_pos_y   =   0;
xlim_min        =   -0.5;  #xlim and ylim of plots
xlim_max        =   10.5;
ylim_min        =   -2;
ylim_max        =   12;
n               =   10;    #size of square grid


grid = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],]


grid = np.array(grid)
occupied_positions_x = np.where(grid == 1)[0];
occupied_positions_y = np.where(grid == 1)[1];


path_x = [0,1,2,3,4,5,6,7,8,9]
path_y = [0,2,8,8,2,2,2,5,8,8]


# Interpolate the path using a spline
tck, u = interpolate.splprep([path_x,path_y], s = 0)
global_path = interpolate.splev(u, tck)

plt.figure()
plt.plot(occupied_positions_x, occupied_positions_y, 'bo', markersize = 5)
plt.plot(global_path[0],global_path[1],'r-')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


#-------------------------------------------------------------------------------#
#                                Bubble Tunnel Method                           #
#-------------------------------------------------------------------------------#

## Determine the closest occupied point and distance with reference to all the discrete points of the reference trajectory

s = 0.0
j = 0
k = 0

distance = 0
previous_distance = 0
radii = []
closest_points = []
npoints = 500                #numbr of points of every circle
ts = np.linspace(0, 2*np.pi, npoints)
feasiblebubbles_x = []
feasiblebubbles_y = []
midpoints_x = []
midpoints_y = []
midpoint_x = 0.0
midpoint_y = 0.0
max_radius = 2                 #max radius 
radius_limit = 2
radius = 0
end_of_spline_reached = False

s_1 = 0.0
s_2 = 0.1
point_1 = interpolate.splev(s_1, tck)
point_2 = interpolate.splev(s_2, tck)
dist = ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5
ref_dist = (s_2-s_1)/dist

while not end_of_spline_reached:
    if s >= 0.90:
        s = 1.0
        end_of_spline_reached = True
    while j < len(occupied_positions_x):
        point = interpolate.splev(s, tck) 
        delta_x = point[0] - occupied_positions_x[j]
        delta_y = point[1] - occupied_positions_y[j]
        distance = (((delta_x**2) + (delta_y**2)))**0.5
        
        if j == 0:
            smallest_distance = distance
            k = j
        if distance < smallest_distance:
            smallest_distance = distance
            k = j
        j = j + 1
    
    closest_points.append([occupied_positions_x[k], occupied_positions_y[k]])

    if smallest_distance < max_radius:
        delta_x = point[0] - occupied_positions_x[k]
        delta_y = point[1] - occupied_positions_y[k]

        midpoint_x = point[0] + (1/2)*delta_x
        midpoint_y = point[1] + (1/2)*delta_y

        radius = (2.9/2.0) * smallest_distance
        new_radius_feasible = True

        j = 0
        while j < len(occupied_positions_x):
            distance = (((midpoint_x - occupied_positions_x[j])**2) + (midpoint_y - occupied_positions_y[j])**2)**0.5
            
            if distance >= radius:
                j = j + 1
            else:
                new_radius_feasible = False
                j = len(occupied_positions_x)
        
        if new_radius_feasible:
            radii.append(radius)
            midpoints_x.append(midpoint_x)
            midpoints_y.append(midpoint_y)
            smallest_distance = radius

        else:
            radii.append(smallest_distance)
            midpoints_x.append(double(point[0]))
            midpoints_y.append(double(point[1]))
            midpoint_x = point[0]
            midpoint_y = point[1]
            radius = smallest_distance

    else:
        radii.append(smallest_distance)
        midpoints_x.append(double(point[0]))
        midpoints_y.append(double(point[1]))
        midpoint_x = point[0]
        midpoint_y = point[1]
        radius = smallest_distance


    fx = midpoint_x + radius*np.cos(ts)
    fy = midpoint_y + radius*np.sin(ts)

    feasiblebubbles_x.append(fx)
    feasiblebubbles_y.append(fy)

    
    if radius >= radius_limit:
        s = s + ref_dist*radius
    elif ref_dist*radius <= 0.10:
        s = s + ref_dist*radius
    else:
        s = s + 0.1
    j = 0



plt.figure()
plt.plot(global_path[0], global_path[1], 'b-')
plt.plot(midpoints_x, midpoints_y, 'rx')
plt.plot(occupied_positions_x, occupied_positions_y, 'o')
plt.legend(['Global Reference Trajectory','Midpoints of the bubbles', 'Occupied Positions'])
plt.title('The feasible bubbles midpoints and the global trajectory')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])

plt.figure()
plt.plot(midpoints_x, midpoints_y, 'rx',markersize= 5)
plt.plot(occupied_positions_x, occupied_positions_y, 'o', markersize= 3)
plt.plot(feasiblebubbles_x, feasiblebubbles_y, 'yx', markersize=1)
plt.legend(['Midpoints', 'Occupied Positions', 'Feasible Bubbles'])
plt.title('The feasible Bubbles')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])

#-------------------------------------------------------------------------------#
#                   Define the optimal control problem                          #
#-------------------------------------------------------------------------------#




ocp = Ocp(T = FreeTime(10.0))

Nsim    = 10            # how much samples to simulate
N       = 20            # number of control intervals


# Logging variables
time_hist      = np.zeros((Nsim+1, N+1))
x_hist         = np.zeros((Nsim+1, N+1))
y_hist         = np.zeros((Nsim+1, N+1))
theta_hist     = np.zeros((Nsim+1, N+1))
s_hist         = np.zeros((Nsim+1, N+1))
v_hist         = np.zeros((Nsim+1, N+1))
w_hist         = np.zeros((Nsim+1, N+1))
sdot_hist      = np.zeros((Nsim+1, N+1))


# System model
x       =  ocp.state()
y       =  ocp.state()
theta   =  ocp.state()
v       =  ocp.control()
w       =  ocp.control()

#path parameter s
s       =  ocp.state()
sdot    =  ocp.control()

#ODEs
ocp.set_der(x       ,        v*cos(theta))
ocp.set_der(y       ,        v*sin(theta))
ocp.set_der(theta   ,        w)
ocp.set_der(s       ,        sdot)


#-------------------------------------------------------------------------------#
#                            Solve the first iteration                          #
#-------------------------------------------------------------------------------#


# Initial states 
ocp.set_initial(x,      initial_pos_x)
ocp.set_initial(y,      initial_pos_y)
ocp.set_initial(theta,  0)
ocp.set_initial(s,      0)


# Constraints on initial point
nx  = 4 #number of states
X_0 = ocp.parameter(nx)
X   = vertcat(x, y, theta, s)

ocp.subject_to(ocp.at_t0(X) == X_0)
current_X = vertcat(initial_pos_x,initial_pos_y,0.0,0.0) #this initializaiton of s is causing problems
ocp.set_value(X_0, current_X)

# Constraints at the final point  = maybe we hsould not add this in the MPC = every MPC iteration should have another final point!
# X_f = ocp.parameter(nx-1)
# X2   = vertcat(x, y, theta)
# ocp.subject_to(ocp.at_tf(X2) == X_f)
# X = vertcat(end_goal_x,end_goal_y,0.0)      
# ocp.set_value(X_f, X)


#constraints on controls 
ocp.subject_to(  0    <= ( v  <= 1   ))
ocp.subject_to( -pi   <= ( w  <= pi  ))
ocp.subject_to( sdot  >=   0)             #path "increasing"


ocp.subject_to(ocp.at_tf(s) < 1)

# ocp.subject_to(ocp.at_t0(s) > 0)          #when s(0) = 0 the interpolation starts at 0 
# ocp.subject_to(ocp.at_t0(s)==0)



# use the bubbles midpoints and radius to create interpolated tunnel

bubbles_radii = radii
tckb, ub = interpolate.splprep([midpoints_x,midpoints_y], s = 0)
bubbles = interpolate.splev(ub, tckb)
bubbles_x = bubbles[0]
bubbles_y = bubbles[1]

tlength = len(bubbles_x)
tunnel_s = np.linspace(0,1,tlength)
ocp.set_initial(s, np.linspace(0,1, N))
ocp.set_initial(sdot, (tunnel_s[-1]-tunnel_s[0])/tlength)


spline_x = interpolant('x','bspline',[tunnel_s],bubbles_x,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})
spline_y = interpolant('y','bspline',[tunnel_s],bubbles_y,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})
spline_r = interpolant('r','bspline',[tunnel_s],bubbles_radii,{"algorithm": "smooth_linear","smooth_linear_frac":0.49})

#stay inside this and the next bubble
safety_margin = 0.01
ocp.subject_to(((((x-spline_x(s))**2) + ((y-spline_y(s))**2))) 
    <= (spline_r(s) - safety_margin)**2)

ocp.subject_to(((((x-spline_x(ocp.next(s)))**2) + ((y-spline_y(ocp.next(s)))**2))) 
    <= (spline_r(ocp.next(s)) - safety_margin)**2)


# ------------- Objective function 

ocp.add_objective(ocp.integral((x - spline_x(s))**2 + (y-spline_y(s))**2))

# ocp.add_objective(ocp.integral(s**2))

# ocp.add_objective(ocp.at_tf(s**2))

# ocp.add_objective(ocp.integral(((x-end_goal_x)**2) + ((y-end_goal_y)**2)))

# ocp.add_objective(-ocp.at_tf(s))




# ------------- Solution method
options = {"ipopt": {"print_level": 0}}
options["expand"] = True
options["print_time"] = False
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
ts, ss = sol.sample(s, grid='integrator',refine = 100)
xs = np.array(spline_x(ss))
ys = np.array(spline_y(ss))
plt.plot(xs,ys)
plt.plot(midpoints_x,midpoints_y,'rx')


plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])
ts, ss = sol.sample(s, grid='integrator',refine = 100)
ts = np.linspace(0,2*np.pi,1000)
xs = np.array(spline_x(ss))
ys = np.array(spline_y(ss))
rs = np.array(spline_r(ss))
for i in range(ss.shape[0]): plt.plot(xs[i]+rs[i]*cos(ts),ys[i]+rs[i]*sin(ts),'r-')
ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')
plt.plot(xs, ys,'bo')
ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)
plt.plot(xs, ys, '--')
plt.title('OCP solution')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])


plt.figure()
plt.plot(global_path[0], global_path[1], '--')
plt.plot(occupied_positions_x, occupied_positions_y, 'bo')
plt.plot(midpoints_x,midpoints_y,'rx')
plt.plot(xs, ys, 'b-')
plt.legend(['original path','obstacles','Bubbles midpoints','solution'])
plt.title('Solution compared to initial global path')
plt.xlim([xlim_min,xlim_max])
plt.ylim([ylim_min,ylim_max])

plt.show(block=True)




#-------------------------------------------------------------------------------#
#                                   MPC                                         #
#-------------------------------------------------------------------------------#



# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing  
t_sol, x_sol      = sol.sample(x,     grid='control')
t_sol, y_sol      = sol.sample(y,     grid='control')
t_sol, theta_sol  = sol.sample(theta, grid='control')
t_sol, s_sol      = sol.sample(s,     grid='control')
t_sol, v_sol      = sol.sample(v,     grid='control')
t_sol, w_sol      = sol.sample(w,     grid='control')
t_sol, sdot_sol   = sol.sample(sdot,  grid='control')

time_hist[0,:]    = t_sol
x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
theta_hist[0,:]   = theta_sol
s_hist[0,:]       = s_sol
v_hist[0,:]       = v_sol
w_hist[0,:]       = w_sol
sdot_hist[0,:]    = sdot_sol



# Simulate the MPC solving the OCP (with the updated state) several times

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    
    # Combine first control inputs
    current_U = vertcat(v_sol[0], w_sol[0] , sdot_sol[0])

    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]
    
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X)

    # Solve the optimization problem
    sol = ocp.solve()

    # Log data for post-processing  
    t_sol, x_sol      = sol.sample(x,     grid='control')
    t_sol, y_sol      = sol.sample(y,     grid='control')
    t_sol, theta_sol  = sol.sample(theta, grid='control')
    t_sol, s_sol      = sol.sample(s,     grid='control')
    t_sol, v_sol      = sol.sample(v,     grid='control')
    t_sol, w_sol      = sol.sample(w,     grid='control')
    t_sol, sdot_sol   = sol.sample(sdot,  grid='control')
    
    time_hist[i+1,:]    = t_sol
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    theta_hist[i+1,:]   = theta_sol
    s_hist[i+1,:]       = s_sol
    v_hist[i+1,:]       = v_sol
    w_hist[i+1,:]       = w_sol
    sdot_hist[i+1,:]    = sdot_sol
    

    ocp.set_initial(x, x_sol)
    ocp.set_initial(y, y_sol)
    ocp.set_initial(theta, theta_sol)
    ocp.set_initial(s, s_sol)
    ocp.set_initial(v, v_sol)
    ocp.set_initial(w, w_sol)
    ocp.set_initial(sdot, sdot_sol)




# -------------------------------------------
#          Plot the results
# -------------------------------------------

T_start = 0
T_end   = sum(time_hist[k,1] - time_hist[k,0] for k in range(Nsim+1))

fig = plt.figure()
ax2 = plt.subplot(1, 1, 1)
ax2.plot(global_path[0], global_path[1], '--')
ax2.plot(occupied_positions_x, occupied_positions_y, 'bo')
ax2.plot(midpoints_x,midpoints_y,'rx')
ax2.plot(x_hist[0,0], y_hist[0,0], 'b-')
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')

for k in range(Nsim+1):
    ax2.plot(x_hist[k,:], y_hist[k,:], 'b-')
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')  
    T_start = T_start + (time_hist[k,1] - time_hist[k,0])
    plt.pause(0.5)




















