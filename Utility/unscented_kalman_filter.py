import numpy as np
import scipy
import pandas as pd
import copy
import warnings

from scipy import linalg
import scipy.linalg

class UKF:
    def __init__(self, f, h, x0, u0, P0, Q, R,
                 dynamics_type='discrete', discretization_timestep=None,
                 alpha=0.1, beta=2):

        self.alpha = alpha
        self.beta = beta

        self.dynamics_type = dynamics_type
        self.discretization_timestep = discretization_timestep
        
        self._f = f
        self._h = h

        # Store initial state & input vectors
        self.x0 = np.asarray(x0).flatten()
        self.u0 = np.asarray(u0).flatten()

        # Run f & h, make sure they work & get sizes
        test_x0 = self._f(self.x0, self.u0)
        test_y0 = self._h(self.x0, self.u0)

        self.n = np.atleast_1d(test_x0).shape[0]  # number of states
        self.p = np.atleast_1d(test_y0).shape[0]  # number of measurements
        self.c = np.atleast_1d(u0).shape[0]       # number of controls

        self.P0 = P0
        self.Q = Q
        self.R = R
        

    def f_ukf(self, x, u, w):
        """
        discretize if self.dynamics_type is continuous
        """

        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()
        w = np.asarray(w).flatten()
        
        if self.dynamics_type == 'continuous':
            xnew = rk4_discretize(self._f, x, u, self.discretization_timestep)
        elif self.dynamics_type == 'discrete':
            xnew = self._f(x, u)

        xnew = np.asarray(xnew).flatten()

        return np.matrix(xnew + w).T


    def h_ukf(self, x, u, w):
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()
        w = np.asarray(w).flatten()

        ypred = self._h(x, u)

        ypred = np.asarray(ypred).flatten()
        
        return np.matrix(ypred + w).T

    def estimate(self, Y, U, Q=None, R=None, return_sigma_points=True):
        """
        Unscented Kalman Filter, square root implementation. 
    
        Inputs
        ======
        y  --  measurements, np.matrix [m, N]
        x0 --  initial state, np.matrix [k, 1]
        f  --  function describing process dynamics, with inputs of (x, u, w), returns state estimate xhat
        h  --  function describing observation dynamics, with inputs of (x, u, w), returns measurements
        Q  --  Process covariance as a function of time, np.matrix [k, k, N] or [k,k]
        R  --  Measurement covariance as a function of time, np.matrix [m, m, N] or [m,m]
        u  --  Control inputs, np.matrix [k, N]
        P0 --  Initial covariance, provided as a diagonal [k], defaults to 1 for all vals
    
        m: number of measurements
        k: number of states
    
        Returns
        =======
        x  --  Full state estimate, np.matrix [k, N]
        P  --  Covariance matrices, np.matrix [k, k, N]
        s  --  Standard deviations, np.matrix [k, N]
        """

        if type(Y) is pd.core.frame.DataFrame:
            Y = Y.iloc[:, :].values.T

        if type(U) is pd.core.frame.DataFrame:
            U = U.iloc[:, :].values.T

        # check orientation of measurements and do our best to fix it
        if Y.shape[0]==Y.shape[1]:
            warnings.warn("Y is square. Assuming that rows are measurements and time is columns.", UserWarning)
        else:
            if Y.shape[0] == self.p: # that means that rows are measurements and time is columns
                pass
            else:
                Y = Y.T
        Y = np.matrix(Y)

        # check orientation of control inputs and do our best to fix it
        if U.shape[0]==U.shape[1]:
            warnings.warn("U is square. Assuming that rows are controls and time is columns.", UserWarning)
        else:
            if U.shape[0] == self.c: # that means that rows are controls and time is columns
                pass
            else:
                U = U.T
        U = np.matrix(U)

        x0 = self.x0
        P0 = self.P0

        # turn x0 into a column matrix regardless of what is
        x0 = np.matrix(x0)
        if x0.shape[1] > x0.shape[0]:
            x0 = x0.T

        # P0 is expected to be just the diagonal
        if len(P0.shape) > 1:
            P0 = np.diag(P0)

        if R == None:
            R = self.R
        if Q == None:
            Q = self.Q

        y = Y
        u = U

        f = self.f_ukf
        h = self.h_ukf

        alpha = self.alpha
        beta = self.beta
    
        N = y.shape[1]
    
        nx = x0.shape[0]
        ny = y.shape[0]
        nq = Q.shape[0]
        nr = R.shape[0]
    
        a = alpha
        b = beta
        L = nx + nq + nr
        l = a**2*L - L
        g = np.sqrt(L + l)
    
        # Weights for means and covariances
        Wm = np.hstack(([[l/(L + l)]],  1/(2*(L + l))*np.ones([1, 2*L])))
        Wm = np.matrix(Wm)
        Wc = np.hstack(([[(l/(L + l) + (1 - a**2 + b))]], 1/(2*(L + l))*np.ones([1, 2*L])))
        Wc = np.matrix(Wc)
    
        # Sign of the first weight for Cholesky updates
        if Wc[0,0] >= 0:
            sgnW0 = 1
        else:
            sgnW0 = -1
    
        # Index ranges for augmented state
        ix = np.arange(0, nx)
        iy = np.arange(0, ny)
        iq = np.arange(nx, (nx+nq))
        ir = np.arange((nx+nq), (nx+nq+nr))
    
        # Initialize augmented covariance matrix square root
        Sa = np.zeros([L,L])
    
        # Process noise square root
        if len(Q.shape) > 2:
            Sa[np.ix_(iq, iq)] = linalg.cholesky(Q[:,:,0]).T
        else:
            cholQ = linalg.cholesky(Q[:,:]).T
            Sa[np.ix_(iq, iq)] = cholQ
    
        # Measurement noise square root
        if len(R.shape) > 2:
            Sa[np.ix_(ir, ir)] = linalg.cholesky(R[:,:,0]).T
        else:
            cholR = linalg.cholesky(R[:,:]).T
            Sa[np.ix_(ir, ir)] = cholR
    
        # Initialize storage arrays
        Y = np.zeros([ny, 2*L+1])  # Measurements from propagated sigma points
        x = np.zeros([nx,N])       # State estimates
        P = np.zeros([nx,nx,N])    # Covariance matrices
        
        # Initial conditions
        x[:,0:1] = x0
        if P0 is not None:
            P[:,:,0] = np.diag(P0)
        else:
            P[:,:,0] = 1*np.eye(nx)
        S = linalg.cholesky(P[:,:,0]).T  # Initial square root
    
        sigma_points = []
    
        for i in range(1, N):
            # Update augmented covariance square root with current state covariance
            Sa[np.ix_(ix, ix)] = S
    
            # Update process noise (if time-varying)
            if len(Q.shape) > 2: 
                Sa[np.ix_(iq, iq)] = linalg.cholesky(Q[:,:,i]).T
            else:
                Sa[np.ix_(iq, iq)] = cholQ 
    
            # Update measurement noise (if time-varying)
            if len(R.shape) > 2:
                Sa[np.ix_(ir, ir)] = linalg.cholesky(R[:,:,i]).T
            else:
                Sa[np.ix_(ir, ir)] = cholR
    
            # Generate sigma points
            xa = np.vstack([x[:,i-1:i], np.zeros([nq,1]), np.zeros([nr,1])])
            gsa = np.hstack((g*Sa.T, -g*Sa.T)) + xa*np.ones([1, 2*L])
            X = np.hstack([xa, gsa])
    
            # Propagate sigma points through process model
            for j in range(0, 2*L+1):
                try:
                    X[np.ix_(ix, [j])] = f(X[np.ix_(ix, [j])], 
                                           u[:,i-1:i], 
                                           X[np.ix_(iq, [j])],
                                           y[:,i-1:i])
                except:
                    X[np.ix_(ix, [j])] = f(X[np.ix_(ix, [j])], 
                                           u[:,i-1:i], 
                                           X[np.ix_(iq, [j])])
    
                # Propagate through measurement model
                Y[:, j:j+1] = h(X[np.ix_(ix, [j])], 
                                u[:,i-1:i], 
                                X[np.ix_(ir, [j])])
            
            if return_sigma_points:
                sigma_points.append(X[np.ix_(ix, np.arange(0, X.shape[1]))])
    
            # Compute predicted state (weighted mean)
            x_pred = X[np.ix_(ix, np.arange(0, X.shape[1]))] * Wm.T
            x[:,i:i+1] = x_pred
            
            # Compute predicted measurement (weighted mean)
            y_pred = Y * Wm.T
            
            # Create matrices for QR decomposition
            # State deviation matrix (weighted)
            Xdev = np.zeros([nx, 2*L+1])
            for j in range(2*L+1):
                Xdev[:,j:j+1] = np.sqrt(np.abs(Wc[0,j])) * (X[np.ix_(ix, [j])] - x_pred)
            
            # Measurement deviation matrix (weighted)  
            Ydev = np.zeros([ny, 2*L+1])
            for j in range(2*L+1):
                innovation = (Y[:,j:j+1] - y_pred)
                Ydev[:,j:j+1] = np.sqrt(np.abs(Wc[0,j])) * innovation
            
            # QR decomposition for state covariance square root
            # First handle sigma points 1 to 2L (positive weights)
            if 2*L > 0:
                _, S = scipy.linalg.qr(Xdev[:, 1:].T, mode='economic')
            else:
                S = np.zeros((nx, nx))
                
            # Update with first sigma point (potentially negative weight)
            if not np.allclose(Xdev[:, 0], 0):
                S = cholupdate(S, Xdev[:, 0], sgnW0)
            
            # QR decomposition for innovation covariance square root
            if 2*L > 0:
                _, Syy = scipy.linalg.qr(Ydev[:, 1:].T, mode='economic')
            else:
                Syy = np.zeros((ny, ny))
                
            # Update with first sigma point (potentially negative weight)  
            if not np.allclose(Ydev[:, 0], 0):
                Syy = cholupdate(Syy, Ydev[:, 0], sgnW0)
    
            # Compute cross-covariance Pxy using weights properly
            Pxy = np.zeros([nx,ny])
            for j in range(0, (2*L)+1):
                Pxy += Wc[0,j] * (X[np.ix_(ix, [j])] - x_pred) * (Y[:,j:j+1] - y_pred).T
    
            # Skip update if measurements are missing
            if np.any(np.isnan(y[:,i])):
                P[:,:,i] = S.T @ S
                continue
    
            # === MEASUREMENT UPDATE ===
            
            # Compute Kalman gain: K = Pxy * inv(Syy' * Syy)
            try:
                # Use more numerically stable approach
                Syy_full = Syy.T @ Syy
                # Add regularization for numerical stability
                Syy_full += 1e-10 * np.eye(ny)
                
                # Try Cholesky solve first (most stable)
                try:
                    L_syy = linalg.cholesky(Syy_full, lower=True)
                    temp = linalg.solve_triangular(L_syy, Pxy.T, lower=True)
                    K = linalg.solve_triangular(L_syy.T, temp, lower=False).T
                except np.linalg.LinAlgError:
                    # Fallback to LU decomposition
                    try:
                        K = Pxy @ linalg.inv(Syy_full)
                    except np.linalg.LinAlgError:
                        # Final fallback to pseudoinverse
                        K = Pxy @ np.linalg.pinv(Syy_full, rcond=1e-12)
                        
            except (np.linalg.LinAlgError, ValueError):
                # If all else fails, use a very small gain
                K = 1e-6 * np.ones((nx, ny))
            
            # Clean up K matrix
            K[~np.isfinite(K)] = 0
            
            # State update
            try:
                innovation = y[:,i:i+1] - h(x[:,i:i+1], u[:,i:i+1], np.zeros([nr,1]))
                innovation[~np.isfinite(innovation)] = 0  # Clean up innovation
                x[:,i:i+1] = x[:,i:i+1] + K @ innovation
            except:
                # If measurement update fails, just use prediction
                pass
            
            # Clean up state estimate
            x[:,i:i+1][~np.isfinite(x[:,i:i+1])] = 0
            
            # Covariance update using Joseph form in square root
            U = K @ Syy.T
            U[~np.isfinite(U)] = 0  # Clean up U matrix
            
            for j in range(ny):
                if np.linalg.norm(U[:,j]) > 1e-14:
                    S = cholupdate(S, U[:,j], -1)  # Negative update
            
            # Ensure S remains well-conditioned
            S[~np.isfinite(S)] = 0
            # Add small regularization to diagonal to maintain positive definiteness
            S += 1e-10 * np.eye(nx)
            
            # Store full covariance matrix
            P[:,:,i] = S.T @ S
            
        # Compute standard deviations
        s = np.zeros([nx, y.shape[1]])
        for i in range(nx):
            s[i,:] = np.sqrt(P[i,i,:].squeeze())


        # save history
        X = [x[:, i] for i in range(x.shape[1])]
        P = [P[:, :, i] for i in range(P.shape[2])]
        P_diags = np.vstack([np.diag(P[i]) for i in range(len(P))])
        
        self.history = {'X': X,
                        'P': P,
                        'P_diags': P_diags,
                        'sigma_points': None}
        
        if return_sigma_points:
            sigma_points = np.dstack(sigma_points)
            sigma_points_sorted = np.zeros_like(sigma_points)
            for n in range(sigma_points.shape[0]):
                for i in range(sigma_points.shape[2]):
                    sigma_points_sorted[n,:,i] = np.sort(sigma_points[n,:,i])
            self.history['sigma_points'] = sigma_points_sorted

def wrapToPi(rad):
    original_shape = rad.shape
    original_type = type(rad)
    
    rad_wrap = np.asarray(copy.copy(rad)).flatten()
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi

    rad_wrap = rad_wrap.astype(original_type)
    rad_wrap = np.reshape(rad_wrap, original_shape)
    return rad_wrap

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

def cholupdate(R, x, sign=1):
    """
    Cholesky update: update R such that R'*R = R0'*R0 + sign*x*x'
    where R0 is the input R matrix
    """
    n = R.shape[0]
    R = R.copy()
    x = x.copy()
    
    for i in range(n):
        r = np.sqrt(R[i,i]**2 + sign * x[i]**2)
        c = r / R[i,i]
        s = x[i] / R[i,i]
        R[i,i] = r
        if i < n-1:
            R[i, i+1:] = (R[i, i+1:] + sign * s * x[i+1:]) / c
            x[i+1:] = c * x[i+1:] - s * R[i, i+1:]
    
    return R

def qr_update(Q, R, u, v):
    """QR update for rank-1 modification"""
    m, n = Q.shape
    Q_new = np.hstack([Q, u.reshape(-1,1)])
    R_new = np.vstack([R, v.reshape(1,-1)])
    Q_updated, R_updated = scipy.linalg.qr(Q_new @ R_new, mode='economic')
    return Q_updated, R_updated