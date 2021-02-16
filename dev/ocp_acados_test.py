
import numpy as np
from numpy import pi, cos, sin
from rockit import *
from casadi import interpolant, vertcat
from pylab import *
from Grid_generation import create_global_path
from scipy import interpolate


#------------------- Problem specification

#shooting points 
N = 150
dt = 0.1 # has to be low for good path following # erros when dt is high


# both freetime and fixed time work here 
ocp = Ocp(t0=0, T=N*dt)

# Define two scalar states (vectors and matrices also supported)
x = ocp.state()
y = ocp.state()
theta = ocp.state()

# Define one piecewise constant control input
#  (use `order=1` for piecewise linear)
v = ocp.control()
w = ocp.control()

# Specify differential equations for states
#  (DAEs also supported with `ocp.algebraic` and `add_alg`)
ocp.set_der(x       ,        v*cos(theta))
ocp.set_der(y       ,        v*sin(theta))
ocp.set_der(theta   ,        w)


# Path constraints

ocp.subject_to(x >= 0)
ocp.subject_to(y >= 0)
ocp.subject_to(-1 <= (v <= 1 ))

# Boundary constraints
ocp.subject_to(ocp.at_t0(x) == 0)
ocp.subject_to(ocp.at_t0(y) == 0)


pf = vertcat(9,9)

# ------------ soft constraints

slack_tf_x   = ocp.variable()
slack_tf_y   = ocp.variable()
slack_tf_x_2 = ocp.variable()
slack_tf_y_2 = ocp.variable()

ocp.subject_to(slack_tf_x >= 0)
ocp.subject_to(slack_tf_y >= 0)
ocp.subject_to(slack_tf_x_2 >= 0)
ocp.subject_to(slack_tf_y_2 >= 0)

ocp.subject_to( (ocp.at_tf(x) - pf[0]) <= slack_tf_x)
ocp.subject_to( (ocp.at_tf(x) - pf[0]) >= -slack_tf_x_2)

ocp.subject_to( (ocp.at_tf(y) - pf[1]) <= slack_tf_y)
ocp.subject_to( (ocp.at_tf(y) - pf[1]) >= -slack_tf_y_2)

ocp.add_objective(20*( slack_tf_x_2  + slack_tf_y_2   ))
ocp.add_objective(20*( slack_tf_x    + slack_tf_y     ))



#-------------- Add path 

path_x = np.linspace(0, 9, 5)

path_y = [  0.    ,  1.0   , 5.0   , 7.0   , 9.0]

# Interpolate the path using a spline
# this interpolation will now be considered the 'original global path'
Bspline_obj, u = interpolate.splprep([path_x,path_y], u = None, s = 0)
u = np.linspace(0,1,N)
global_path = interpolate.splev(u, Bspline_obj)

pathx = global_path[0]
pathy = global_path[1]
    

s1       =  ocp.state()
sdot1    =  ocp.control()

ocp.set_der(s1  , sdot1)

ocp.subject_to(ocp.at_t0(s1)  == 0.0)
ocp.subject_to(ocp.at_tf(s1)  == 1)

ocp.subject_to(  sdot1  >=  0)  

tlength1       =  len(pathx)
tunnel_s1      =  np.linspace(0,1,tlength1) 

path_spline_x = interpolant('path_spline_x' , 'bspline', [tunnel_s1], pathx , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
path_spline_y = interpolant('path_spline_y' , 'bspline', [tunnel_s1], pathy , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


ocp.add_objective(10*ocp.integral((x - path_spline_x(s1))**2 + (y-path_spline_y(s1))**2)) 


#-------------- Add path constraint using tunnel

constx = pathx
consty = pathy
constr = 1*np.ones(N)


s2       =  ocp.state()
sdot2    =  ocp.control()

ocp.set_der(s2  , sdot2)

ocp.subject_to(ocp.at_t0(s2)  == 0.0)

ocp.subject_to(  sdot2  >=  0)  

tlength2       =  len(constx)
tunnel_s2      =  np.linspace(0,1,tlength2) 

const_spline_x = interpolant('const_spline_x' , 'bspline', [tunnel_s2], constx , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
const_spline_y = interpolant('const_spline_y' , 'bspline', [tunnel_s2], consty , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})
const_spline_r = interpolant('const_spline_r' , 'bspline', [tunnel_s2], constr , {"algorithm": "smooth_linear","smooth_linear_frac":0.49})


ocp.subject_to( (  ( x - const_spline_x(s2) )**2 + ( y - const_spline_y(s2) )**2  <  const_spline_r(s2)**2  )  )


#--------------- Solving the problem

# Pick an NLP solver backend
#  (CasADi `nlpsol` plugin):
ocp.solver('ipopt')


# qp_solvers = ('PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP')
# integrator_types = ('ERK', 'IRK', 'GNSF', 'DISCRETE')
# SOLVER_TYPE_values = ['SQP', 'SQP_RTI']
# HESS_APPROX_values = ['GAUSS_NEWTON', 'EXACT']
# REGULARIZATION_values = ['NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY']


# Pick a solution method
method = external_method('acados', N=N,qp_solver= 'FULL_CONDENSING_HPIPM', nlp_solver_max_iter= 700, hessian_approx='EXACT', regularize_method = 'MIRROR' ,integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=N)
ocp.method(method)



#---------------------- Set initial guesses

# ocp.set_initial(x, pathx)
# ocp.set_initial(y, pathy)              



#------------------- Solve


sol = ocp.solve()



#------------------ Post-processing


t_sol, x_sol  = sol.sample(x,  grid='control')
t_sol, y_sol  = sol.sample(y,  grid='control')

t_sol, s2_sol = sol.sample(s2, grid='control')

bx = const_spline_x(s2_sol)
by = const_spline_y(s2_sol)
br = const_spline_r(s2_sol)

t_sol, s1_sol = sol.sample( s1, grid  = 'control')

xspline_path = np.array(path_spline_x(s1_sol))
yspline_path = np.array(path_spline_y(s1_sol))

# print( "x0", x_sol[0])
# print( "y0", y_sol[0])

# print( "xf" , x_sol[-1])
# print( "yf" , y_sol[-1])

t = np.linspace(0, 2*np.pi, 100)

print(x_sol)
print(y_sol)


figure()
plot(x_sol, y_sol, 'ro')
plot(x_sol, y_sol, 'r--')
plot(pathx, pathy, 'b-')
for i in range(0, N):
    plot(bx[i] + br[i]*np.cos(t),by[i] + br[i]*np.sin(t),'g.')
title("solution in x-y plane")
xlabel("Times [s]")
grid(True)

figure()
plot(x_sol, y_sol, 'ro')
plot(x_sol, y_sol, 'r--')
plot(pathx, pathy, 'b-')
plot(xspline_path,yspline_path,'go')
title("solution in x-y plane")
xlabel("Times [s]")
grid(True)

show(block=True)