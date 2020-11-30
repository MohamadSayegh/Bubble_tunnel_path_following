# Bubble tunnel method for path following OCP

## Current code
 
### Bubble tunnel method that: 

1) Does not cross walls if the original path does so

2) Reduces the number of generated midpoints in empty spaces and does the opposite in tight spaces

3) Creates two sets of Bubbles: one centered on the original path and one shifted away from obstacles to a “good” extent
 
### OCP formulation 

1) Takes the interpolated original path for path following

2) Takes the shifted bubbles (tunnel) as a soft constraint for obstacle avoidance (using slack variables)

3) Takes initial and end point constraints
 

### Drawbacks 
 
1.	I did the inverse of what Ben said (which is the intuitive thing) by setting obstacle avoidance as a soft constraint. This is problematic, because obstacle avoidance should not be allowed. However, I worry that in certain cases there will be some discontinuities in the bubbles (if the path crosses a wall for example) which would make the problem infeasible. I think this could be maybe solved later in MPC where the original path itself is modified if found to cross an obstacle (so local path planning) ?
2.	I also worry that the execution time of both the Bubble method and the solver might be more than allowed for a real application. This is natural since I tailored the design to account for all our goals except this. However, I think this should be considered later in the simulation in ROS when the code is in C or C++.
 
Please tell me what you think. I appreciate your feedback. Also I will hopefully be done with the presentation slides by tomorrow and will send them to you. 
