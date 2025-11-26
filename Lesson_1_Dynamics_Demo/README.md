# Overview

This lesson introduces the planar drone dynamics example used throughout the course, and demonstrates how we will use model predictive control to drive the dynamics along a designed trajectory. 

![animated drone trajectory](drone_animation.gif)

## Notebook A

Topics covered:
  * Introduction of continuous dynamics functions of the form $\mathbf{\dot{x}}=\mathbf{f}(\mathbf{x},\mathbf{u})$
  * Integrating dynamics with `odeint`
  * Designing a trajectory and using model predictive control to determine the control inputs needed to follow it
    * We use a `pybounds` wrapper for functionality provided by `casadi` and `do_mpc`
  * Formatting data into a dataframe using `pandas` and saving to .hdf files

## Notebook B

Topics covered:
  * Loading a `pandas` dataframe
  * Demo of building a `matplotlib` animation using LLM coding 

# Pre-requisites

This lesson assumes working knowledge in the following topics:
1. Basic background in continuous time dynamics
2. Writing python functions and using python packages
3. High level understanding of feedback control -- we will use model predictive control throughout the course, but background in how it works is not necessary

# Reusable Utility Functions 

This notebook introduces the dynamics functions `f(x,u)` and `h(x,u)` introduced in this notebook and the model predictive control framework form the basis of utility functions in `../Utility/planar_drone.py` that used throughout the course. 
