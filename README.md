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
 

