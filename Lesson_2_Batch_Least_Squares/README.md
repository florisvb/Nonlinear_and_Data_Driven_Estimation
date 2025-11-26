# Overview

This lesson introduces batch linear least squares estimation for static parameter estimation. The notebooks demonstrate how to use batch least squares to estimate unknown static parameters from noisy measurement data, and explore the statistical properties of the estimators.

## Notes

Derivation of the batch linear least squares solution for estimating static parameters. 

Topics covered:
  * Formulating the measurement model $\tilde{\mathbf{y}} = H\mathbf{x} + \mathbf{v}$
  * Deriving the least squares solution $\hat{\mathbf{x}} = (H^TH)^{-1}H^T\tilde{\mathbf{y}}$
  * Extensions: weighted least squares, matrix decompositions (QR and SVD)
  * Statistical properties of residuals

## Notebook A

Topics covered:
  * Introduction of the motivating example: $\tilde{y}(t) = t + \sin(t) + 2\cos(2t) - 0.4e^t/10^4 + v(t)$
  * Building the measurement matrix $H$ with basis functions
  * Generating synthetic data with Gaussian noise
  * Visualizing measurements and setting plotting standards

## Notebook B

Topics covered:
  * Implementing the linear least squares solution $\hat{\mathbf{x}} = (H^TH)^{-1}H^T\tilde{\mathbf{y}}$
  * Building and comparing candidate models with different basis functions
  * Comparing model predictions against measurements
  * Extrapolating beyond the measurement time window

## Notebook C

Topics covered:
  * Statistical interpretation of least squares estimates
  * Analyzing residuals: mean, variance, and standard deviation
  * Actual vs predicted plots for model validation
  * QQ plots for assessing normality of residuals

# Pre-requisites

This lesson assumes working knowledge in the following topics:
1. Basic linear algebra (matrix operations, matrix-vector multiplication)
2. Python programming with NumPy for numerical computing
3. Basic statistics (mean, variance, Gaussian distributions)
4. Understanding of the general form of basis functions

# References

  * Crassidis, J. L., & Junkins, J. L. (2011). *Optimal Estimation of Dynamic Systems* (2nd ed.). Chapman and Hall/CRC. Chapter 1: Least Squares Approximation.

# Foundations for Future Lessons 

The parameter estimation examples and least squares methods introduced in these notebooks form the foundation for estimation techniques used throughout the course. 

**Key concepts that will be extended in later lessons:**
  * The measurement model $\tilde{\mathbf{y}} = H\mathbf{x} + \mathbf{v}$ extends to time-varying state estimation
  * Batch least squares is the foundation for understanding recursive (sequential) estimation methods
  * Statistical analysis of residuals provides the basis for filter tuning and performance evaluation
  * The matrix decomposition methods (QR, SVD) appear in more advanced estimation algorithms

**Limitations of batch methods addressed in future lessons:**
  * Batch methods require all data before estimation (recursive methods process data sequentially)
  * Static parameter estimation extends to dynamic state estimation with process models
  * Linear least squares extends to nonlinear estimation problems (Extended Kalman Filter, Unscented Kalman Filter)

# Reusable Utility Functions 

This lesson sets a plotting standard function `plot_tme` in `../Utility/plot_utility.py`. 