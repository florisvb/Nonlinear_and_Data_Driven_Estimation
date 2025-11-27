# Overview

This lesson covers methods for calculating the empirical observability of a dynamical system given the dynamics and measurement functions. Doing this with small time windows that are slid across the time series of a trajectory can reveal movement motifs that maximize observability. 

## Key Concepts

* Use perturbation approach to numerically determine the observability matrix as the Jacobian of the measurements with respect to the states: $\mathcal{O} = \frac{\delta Y}{\delta X}$
* Find the Fisher information matrix assuming $Q=0$ as $\mathcal{F}=\mathcal{O}^T \mathcal{R} \mathcal{O}$
* Find a regularized inverse of $\mathcal{F}$: $\tilde{\mathcal{F}}^{-1}=[\mathcal{F}+\lambda I]^{-1}$
* The diagonal of $\tilde{\mathcal{F}}^{-1}$ describes the minimum error variance attainable by any unbiased estimator via the Cramer Rao Bound
* All computations can be done using the python package `pybounds`

## Notebook A

Simple implementation demonstrating the perturbation approach to finding the observability matrix for a trajectory given a complete time window.

## Notebook B

Demonstration of how to use `pybounds` to simplify the computation, and perform observability calculations on sliding windows.

## Notebook C

Demonstration of how to use `pybounds` when the dynamics and/or measurements are not given by functions that follow the form $f(x,u)$ and $h(x,y)$. 

# References

Cellini, B., Boyacioglu, B., Lopez, A., & van Breugel, F. (2025). Discovering and exploiting active sensing motifs for estimation (arXiv:2511.08766). arXiv. https://arxiv.org/abs/2511.08766

# Tips and FAQ

1. Start by including as many "parameters" as "static states". That is, include these variables as a part of the state vector. Then you can explore whether they need to be known or not by including them in the measurement function, or not. 
2. If the observability of a state variable changes, determine what state values make it change. You can often increase the level of observability by amplifying the magnitude of these active sensing motifs.
3. Do not include any stochasticity in your measurement (or dynamics) functions. Stochastic dynamics will artificially increase the level of observability. The simulation paradigm you use needs to be deterministic, smooth, and sensitive to small perturbations.
4. Beware: using a large time window can lead to long computation times. Debug your systems observability by setting the time window to None, which will provide a single observability matrix. 
