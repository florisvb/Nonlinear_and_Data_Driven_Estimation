# Overview

The UKF implementation we use (found in `Utility`) is a square root implementation for improved numerical stability and performance.  

## Key Concepts

## Notebook A

The Unscented Kalman Filter applied to our planar drone example with a measurement set that includes camera and IMU information. 

# References

Wan, E. A., & van der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation. In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (pp. 153-158). IEEE. https://doi.org/10.1109/ASSPCC.2000.882463

# Tips and FAQ

1. Before trying to tune a UKF complete a nonlinear observability analysis. The insight from the observability analysis will indicate whether your UKF is not working because of observability issues, or the UKF.
2. To tune your UKF use a trajectory that has high levels of observability for the states you are most interested in and then progressively make the UKF work harder.
   * Start by setting the following parameters:
     * Set $x_0$ to be very close to the correct initial condition
     * Set $P_0$ to be quite small, e.g. $10^{-4}$
     * Set $Q$ to be quite small, e.g. $10^{-8}$
   * Next gradually make the initial condition less correct, this is a more realistic and fair test
     * Increase $Q$ to speed up the filter
     * Increase $P_0$ to speed up the filter initially
3. If your UKF only works well when you give it an accurate initial guess, try the AI-UKF (Lesson 13).
4. If the $3\sigma$ bounds is very small, even for states that have moderate levels of observability, then your Q is probably too small, try making it bigger.  
