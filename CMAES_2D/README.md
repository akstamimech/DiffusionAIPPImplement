A full rundown of the Model architecture:
https://drive.google.com/file/d/1HzfLKIaWbUjhTdrdjHyJhebhZrX9s_IV/view?usp=sharing


AIPP simulation running scripts + Diffusion model scripts for training and sampling. 
Diffusion model can be found in ./Diffusion/SparseDiffusion.py. Training params are all in the head of the file.

SparseDiffusion.py has inputs: Mean conditions (1, 51, 51), 
Variance conditions (1, 51, 51), 
Initial position (1, 2)

And outputs the expected noise in a control waypoint vector (1, 2, 8), which is used to find a CubicSpline (1, 2, 41). 

Loss is computed by MSE loss between expected noise in the waypoint sequence vs. the true amount of noise, as well as the true amount of noise in the spline trajectory. The principle behind the spline loss is that any resultant spline is the sum of a noisy spline + the true spline that we're trying to converge to. 
