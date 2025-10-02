import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize

import sympy as sp

# Discretize continuous dynamics with the Runge-Kutta 4th order method (RK4)
def rk4_discretize(f, x, u, dt):
    """
    Discretizes the continuous-time dynamics using the Runge-Kutta 4th order method (RK4).

    :param f: Function that defines the system dynamics (dx/dt = f(x, u))
              f should accept the current state `x` and input `u` and return the state derivatives.
    :param x: Current state (numpy array), representing the state at time t
    :param u: Control input (numpy array), control applied at time t
    :param dt: Time step (float), the discretization time step

    :return: Discretized state at time t+dt (numpy array)
    """

    # Step 1: Compute k1, the first estimate of the state change (function evaluation at time t)
    k1 = f(x, u)  # k1 is the rate of change at the current state

    # Step 2: Compute k2, estimate of state change at time t + dt/2, based on k1
    # Perturb x by half the step size (dt/2) in the direction of k1
    k2 = f(x + 0.5 * dt * k1, u)  # k2 is the rate of change at t + dt/2

    # Step 3: Compute k3, another estimate of state change at time t + dt/2, based on k2
    # Perturb x by half the step size (dt/2) in the direction of k2
    k3 = f(x + 0.5 * dt * k2, u)  # k3 is the rate of change at t + dt/2 (but using k2)

    # Step 4: Compute k4, estimate of state change at time t + dt, based on k3
    # Perturb x by the full time step (dt) in the direction of k3
    k4 = f(x + dt * k3, u)  # k4 is the rate of change at t + dt

    # Step 5: Compute the weighted sum of the estimates (k1, k2, k3, k4) to update x
    # The final estimate is a weighted average of all k's
    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next

# Linearize with a numerical jacobian
def jacobian_numerical(f, x0, u0, epsilon=0.001):
    
    # Get A
    Aj = []
    for i in range(len(f(x0,u0))):
        
        def f_scalar(x,u,i):
            x_new = f(x, u)
            return np.ravel(x_new)[i]
        
        j = scipy.optimize.approx_fprime(x0, f_scalar, epsilon, u0, i)
        Aj.append(j)
        
    # Get B
    Bj = []
    for i in range(len(f(x0,u0))):
        
        def f_scalar(u,x,i):
            x_new = f(x, u)
            return np.ravel(x_new)[i]
        
        j = scipy.optimize.approx_fprime(u0, f_scalar, epsilon, x0, i)
        Bj.append(j)
    
    return np.array(np.vstack(Aj)), np.array(np.vstack(Bj))

def linear_discrete_kalman_forward_update(xhat_fp, P_fp, y, u, A, B, C, D, R, Q):
    """
    :param xhat_fm: prior state estimate
    :param P_fm: prior error covariance estimate
    :param y: measurements at this time step
    :param u: controls at this time step
    :param A,B,C,D: linear discrete time model 
    :param R: measurement covariance matrix 
    :param Q: process covariance matrix
    :return: current state estimate and error covariance matrix
    """
    I = np.array(np.eye(A.shape[0]))

    # predict the state and covariance for this time step given the previous estimate and the model
    xhat_fm = A@xhat_fp + B@u
    P_fm = A@P_fp@A.T + Q

    # Calculate optimal Kalman gain
    K_f = P_fm@C.T@np.linalg.pinv(C@P_fm@C.T + R)

    # If there are measurements: perform the update and predict steps
    if y is not None and not np.isnan(y).any():
        # estimate the measurements from the current state and the model
        yhat = C@xhat_fm + D@u

        # calculate the innovation
        innovation = (y - yhat)

        # Update state and covariance with optimal Kalman gain
        xhat_fp = xhat_fm + K_f@innovation
        P_fp = (I - K_f@C)@P_fm

    # If there are no measurements: cannot perform the update with innovation, so only predict forwards with the model
    else:
        xhat_fp = xhat_fm
        P_fp = (I - K_f@C)@P_fm
        
    return xhat_fp, P_fp

def linear_discrete_kalman_filter(x0, P0, Y, U, A, B, C, D, R, Q):
    """
    :param x0: initial state guess
    :param P0: initial state error covariance guess
    :param Y: array of all measurements
    :param U: array of all controls
    :param A,B,C,D: linear discrete time model 
    :param R: measurement covariance matrix 
    :param Q: process covariance matrix
    :return: state estimates and associated error covariance matrix
    """

    xhat_fp = x0
    P_fp = [P0]

    for i in range(Y.shape[1]):
        xhat_fp_i, P_fp_i = linear_discrete_kalman_forward_update(xhat_fp[:, [-1]], P_fp[-1], Y[:, [i]], U[:, [i]],
                                                                     A, B, C, D, R, Q)
        xhat_fp = np.hstack((xhat_fp, xhat_fp_i))
        P_fp.append(P_fp_i)

    # don't return that last element to keep size same as t
    return xhat_fp[:,1:], P_fp[1:]