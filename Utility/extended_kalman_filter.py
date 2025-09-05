import numpy as np
import copy
import pandas as pd
import warnings

class EKF:
    def __init__(self, f, h, x0, u0, P0, Q, R,
                 circular_measurements=None,
                 dynamics_type='discrete', discretization_timestep=None):
        """
        Initialize Extended Kalman Filter (EKF).

        :param callable f: state transition function, f(x, u)
        :param callable h: measurement function, h(x, u)
        :param np.ndarray x0: initial guess of x
        :param np.ndarray u0: initial inputs
        :param np.ndarray P0: initial error covariance
        :param np.ndarray Q: process noise covariance matrix
        :param np.ndarray R: measurement noise covariance matrix
        :param tuple | list | np.ndarray | None circular_measurements: optional iterable of bools to indicate which measurements are circular variables
        :param string dynamics: 'discrete' or 'continuous': are the dynamics given by f discrete or continuous? If continuous, f will be discretized with rk4 method. 
        :param float discretization_timestep: default time step for EKF update, required if dynamics_type is continuous. If continuous dynamics, you can change the timestep for each update step
        """

        # Store state transition & measurement functions
        self.f = f
        self.h = h

        # Save dynamics type
        self.dynamics_type = dynamics_type
        if dynamics_type == 'continuous':
            if discretization_timestep is None:
                raise ValueError('For continuous dynamics discretization_timestep cannot be 1')

            self.discretization_timestep = discretization_timestep
            
        # Store matrices
        self.F = None
        self.H = None
        self.S = None
        self.K = None
        self.E = None

        # Store initial state & input vectors
        self.x0 = x0
        self.u0 = u0

        # Run f & h, make sure they work & get sizes
        self.x0 = self.f_discrete(self.x0, self.u0)
        self.z0 = self.h(self.x0, self.u0)

        self.n = np.atleast_1d(self.x0).shape[0]  # number of states
        self.p = np.atleast_1d(self.z0).shape[0]  # number of measurements
        self.c = np.atleast_1d(self.u0).shape[0]  # number of control inputs

        # Set what variables are circular
        if circular_measurements is None:  # default is to assume no variables are circular
            self.circular_measurements = tuple(np.zeros(self.n))
        else:
            self.circular_measurements = tuple(circular_measurements)

        # Set noise covariances
        self.P = P0
        self.Q = Q
        self.R = R

        # Store state & covariance history
        self.history = {'X': [self.x0],
                        'U': [self.u0],
                        'Z': [self.z0],
                        'P': [self.P],
                        'P_diags': [np.diag(self.P)],
                        'R': [self.R],
                        'Q': [self.Q],
                        'F': [],
                        'H': [],
                        'S': [],
                        'K': [],
                        'E': [],
                        'rho': [],
                        'Jk': [self.P],
                        'inv_Jk': [self.P]
                        }

        # Current state, inputs, & measurements
        self.x = self.x0.copy()
        self.u = self.u0.copy()
        self.z = self.z0.copy()

        # Current timestep
        self.k = 0

    def f_discrete(self, x, u):
        """
        used when self.dynamics_type is continuous, does nothing if self.dynamics_type is discrete
        """
        if self.dynamics_type == 'continuous':
            return rk4_discretize(self.f, x, u, self.discretization_timestep)
        elif self.dynamics_type == 'discrete':
            return self.f(x, u)

    def _wrap_1D(self, f):
        def wrapped_f(*args, **kwargs):
            return np.atleast_1d(f(*args, **kwargs))
        return wrapped_f

    def _predict(self, u, Q=None, discretization_timestep=None):
        """
        EKF prediction step.

        :param u: input vector
        :param Q: optional process noise covariance matrix for this step
        :param discretization_timestep: optional time step, only applicable if EKF was defined with continuous dynamics, will override the default discretization_timestep. 
        """

        if self.dynamics_type != 'continuous':
            if discretization_timestep is not None:
                raise ValueError('For discrtete dynamics discretization_timestep is not used')
        if discretization_timestep is not None and self.dynamics_type == 'continuous':
            self.discretization_timestep = discretization_timestep

        # Update controls
        self.u = u.copy()

        # Set process noise covariance
        if Q is not None:
            self.Q = Q

        # Predict next state
        self.x = self.f_discrete(self.x, self.u)

        # Use jacobian of state transition function to predict state estimate covariance
        def F_jacobian(x, u):
            Jf = jacobian_numerical(self.f_discrete, x, u)
            return Jf
        self.F = F_jacobian(self.x, self.u)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z, R=None):
        """
        EKF update step with measurement z.

        :param z: measurement vector
        :param R: optional measurement noise covariance matrix for this step
        """

        # Current measurement
        self.z = np.atleast_1d(np.array(z).copy())

        # Set measurement noise covariance
        if R is not None:
            self.R = R

        # Predicted measurements from state estimate
        z_pred = self.h(self.x, self.u)
        z_pred = np.atleast_1d(z_pred)

        # Compute innovation (measurement residual)
        # y = self.z - z_pred
        y = np.zeros(self.p)
        for j in range(self.p):
            if self.circular_measurements[j]:  # circular measurement
                # y[j] = angle_difference(z[j], z_pred[j])
                # y[j] = angle_difference(self.z[j], z_pred[j])
                y[j] = wrapToPi(np.array(self.z[j] - z_pred[j]))
            else:  # non-circular measurement
                y[j] = self.z[j] - z_pred[j]

        # Use jacobian of measurement function to compute the innovation/residual covariance
        def H_jacobian(x, u):
            Jh = jacobian_numerical(self.h, x, u)
            return Jh
        self.H = H_jacobian(self.x, self.u)
        self.S = self.H @ self.P @ self.H.T + self.R

        # Near-optimal Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # Update state estimate
        self.x = self.x + self.K @ y

        # Update state covariance estimate
        I = np.eye(self.P.shape[0])
        # self.P = (I - K @ H) @ self.P  # standard EKF covariance update
        self.P = ((I - self.K @ self.H) @ self.P   # Joseph form of the covariance update
                  @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T)

        # Spectral radius
        self.E = (I - (self.K @ self.H)) @ self.F
        eigenvalues, eigenvectors = np.linalg.eig(self.E)
        rho = np.abs(eigenvalues)

        # Posterior Cramer-Rao Bound
        inv = np.linalg.inv
        Jk = self.history['Jk'][-1]
        Jk = inv(self.Q) - inv(self.Q) @ self.F @ inv(Jk + self.F.T @ inv(self.Q) @ self.F) @ self.F.T @ inv(self.Q) + self.H.T @ inv(self.R) @ self.H
        lamb = 1e-5
        Jk = Jk + lamb * np.eye(np.shape(Jk)[0])
        inv_Jk = inv(Jk)

        # Update history
        self.history['X'].append(self.x.copy())
        self.history['U'].append(self.u.copy())
        self.history['Z'].append(self.z.copy())
        self.history['P'].append(self.P.copy())
        self.history['P_diags'].append(np.diag(self.P.copy()))
        self.history['Q'].append(self.Q.copy())
        self.history['R'].append(self.R.copy())
        self.history['F'].append(self.F.copy())
        self.history['H'].append(self.H.copy())
        self.history['S'].append(self.S.copy())
        self.history['K'].append(self.K.copy())
        self.history['E'].append(self.E.copy())
        self.history['rho'].append(rho)
        self.history['Jk'].append(Jk.copy())
        self.history['inv_Jk'].append(inv_Jk.copy())

        # If it's the 1st time-step, set the initial values
        if self.k == 0:
            self.history['Z'][0] = self.z.copy()
            self.history['P'][0] = self.P.copy()
            self.history['Q'][0] = self.Q.copy()
            self.history['R'][0] = self.R.copy()

        # Update time-step
        self.k += 1

    def forward_update(self, y, u, Q=None, R=None, discretization_timestep=None):
        """
        EKF prediction and update steps.

        :param u: input vector
        :param y: measurement vector

        :param Q: optional process noise covariance matrix for this step
        :param discretization_timestep: optional time step, only applicable if EKF was defined with continuous dynamics, will override the default discretization_timestep. 
        :param R: optional measurement noise covariance matrix for this step
        """

        self._predict(u, Q=Q, discretization_timestep=discretization_timestep)
        self._update(y, R=R)

    def estimate(self, Y, U, Q=None, R=None, discretization_timestep=None):
        """
        param Y: collection of all measurements, either 2d array with each column corresponding to measurements from one timestep, or pandas dataframe
        param U: collection of all control inputs, either 2d array with each column corresponding to controls from one timestep, or pandas dataframe

        Q, R: list of Q and R matrices to use for each corresponding measurement, control, or None, in which case the Q, R matrices from the first initilization of the EKF will be used
        discretization_timstep: list of 1d array of time steps between measurements. If None, use default set in initilization. 
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

        # check orientation of control inputs and do our best to fix it
        if U.shape[0]==U.shape[1]:
            warnings.warn("U is square. Assuming that rows are controls and time is columns.", UserWarning)
        else:
            if U.shape[0] == self.c: # that means that rows are controls and time is columns
                pass
            else:
                U = U.T
        
        for k in range(1, Y.shape[1]):  # each time-step
            # Current inputs
            u = np.squeeze(U[:,k])
        
            # Current measurements
            y = np.squeeze(Y[:,k])
        
            # Update
            self.forward_update(y, u)

        

def jacobian_numerical(f, x, u, epsilon=1e-6):
    """
    Approximate the Jacobian of a function f at the point x using finite differences.

    :param callable f: function for which to compute the Jacobian
    :param np.ndarray x: point at which to evaluate the Jacobian
    :param float epsilon: perturbation value for finite differences
    :return np.ndarray: Jacobian matrix
    """

    # Ensure floats
    x = np.array(x).astype(np.float64)
    u = np.array(u).astype(np.float64)

    n = len(x)  # number of function inputs
    m = len(f(x, u))  # number of function outputs

    # Jacobian
    jacobian = np.zeros((m, n))
    for i in range(n):
        # Perturb x[i] with epsilon in the positive direction
        x_plus = x.copy()
        x_plus[i] = x_plus[i] + epsilon

        # Perturb x[i] with epsilon in the negative direction
        x_minus = x.copy()
        x_minus[i] = x_minus[i] - epsilon

        # Evaluate the function at the perturbed point
        jacobian[:, i] = (f(x_plus, u) - f(x_minus, u)) / (2.0 * epsilon)

    return jacobian


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


def angle_difference(target, current):
    """
    Computes the signed minimal angular difference (in radians)
    from current to target, in the range [-π, π).
    """
    diff = (target - current + np.pi) % (2 * np.pi) - np.pi
    return diff

def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def wrapTo2Pi(rad):
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad