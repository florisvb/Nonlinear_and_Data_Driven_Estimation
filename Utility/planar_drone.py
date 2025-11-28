import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize

from scipy import interpolate

import pandas as pd

import pybounds

############################################################################################
# Set some global parameters
############################################################################################
m = 1 # mass (kg)
g = 9.81 # acceleration due to gravity (m/s^2)
L = 0.5 # length of the drone arm (m)
Iyy = 0.02 # moment of inertia (e.g. 1/12*m*L**2 for a solid rod, as an approximation)

############################################################################################
# continuos time dynamics function
############################################################################################
class F(object):
    def __init__(self, k=None):
        self.k = k

    def f(self, x_vec, u_vec, m=m, g=g, L=L, I=Iyy, return_state_names=False):
        """
        Continuous time dynamics function for the system shown in the equation.
        
        Parameters:
        x_vec : array-like, shape (7,)
            State vector [θ, θ̇, x, ẋ, z, ż, k]
        u_vec : array-like, shape (2,)
            Control vector [j1, j2]
        L : float, default 0.5
            drone arm length
        m : float, default 1.0
            drone mass
        g : float, default 9.81
            Gravitational acceleration
        k : float, default None
            motor control coefficient, None means that it is part of the state, if a value is specified, then it is not part of the state
        
        Returns:
        x_dot : numpy array, shape (7,)
            Time derivative of state vector
        """
        k = self.k

        if x_vec is not None:
            if k is None:
                assert len(x_vec) == 7
            elif k is not None:
                assert len(x_vec) == 6

        if return_state_names:
            if k is None:
                return ['theta', 'theta_dot', 'x', 'x_dot', 'z', 'z_dot', 'k']
            else:
                return ['theta', 'theta_dot', 'x', 'x_dot', 'z', 'z_dot']
        
        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if k is None:
            k = x_vec[6]

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]
        
        # f0 component: drift dynamics (no controls)
        f0_contribution = np.array([ theta_dot, 
                                     0, 
                                     x_dot, 
                                     0, 
                                     z_dot, 
                                     -g / m, 
                                     0])
        
        # f1 component: multiplied by control j1
        f1_contribution = j1 * np.array([0, 
                                         L*k/Iyy, 
                                         0, 
                                         0, 
                                         0, 
                                         0, 
                                         0])
        
        # f2 component: multiplied by control j2
        f2_contribution = j2 * np.array([0,
                                         0,
                                         0,
                                         -k * np.sin(theta) / m,
                                         0,
                                         k * np.cos(theta) / m,
                                         0])

        # combined dynamics
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution
        
        if self.k is None:
            return x_dot_vec
        else:
            return x_dot_vec[0:6]


############################################################################################
# continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, k=None):
        self.k = k 
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_gps(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['theta', 'x', 'z', 'k']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Measurements
        y_vec = np.array([theta, x, z, k])

        # Return measurement
        return y_vec

    def h_camera_theta_k(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['optic_flow', 'theta', 'k']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Measurements
        y_vec = np.array([x_dot/z, theta, k])

        # Return measurement
        return y_vec

    def h_camera_thetadot_k(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['optic_flow', 'theta_dot', 'k']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Measurements
        y_vec = np.array([x_dot/z, theta_dot, k])

        # Return measurement
        return y_vec

    def h_camera_imu_notheta(self, x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
        if return_measurement_names:
            return ['optic_flow', 'theta_dot', 'accel_x', 'accel_z']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Model for acceleration -- these come from the model
        accel_x = -k*np.sin(theta)*j2 / m
        accel_z = -g + k*np.cos(theta)*j2 / m

        # Measurements
        y_vec = np.array([x_dot/z, theta_dot, accel_x, accel_z])

        # Return measurement
        return y_vec

    def h_camera_imu(self, x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
        if return_measurement_names:
            return ['optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Model for acceleration -- these come from the model
        accel_x = -k*np.sin(theta)*j2 / m
        accel_z = -g + k*np.cos(theta)*j2 / m

        # Measurements
        y_vec = np.array([x_dot/z, theta, theta_dot, accel_x, accel_z])

        # Return measurement
        return y_vec

    def h_camera_imu_k(self, x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
        if return_measurement_names:
            return ['optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z', 'k']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Model for acceleration -- these come from the model
        accel_x = -k*np.sin(theta)*j2 / m
        accel_z = -g + k*np.cos(theta)*j2 / m

        # Measurements
        y_vec = np.array([x_dot/z, theta, theta_dot, accel_x, accel_z, k])

        # Return measurement
        return y_vec

    def h_all(self, x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
        if return_measurement_names:
            return ['x', 'z', 'optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z', 'k']

        # Extract state variables
        theta = x_vec[0]
        theta_dot = x_vec[1]
        x = x_vec[2]
        x_dot = x_vec[3]
        z = x_vec[4]
        z_dot = x_vec[5]
        if self.k is None:
            k = x_vec[6]
        else:
            k = self.k
            

        # Extract control inputs
        j1 = u_vec[0]
        j2 = u_vec[1]

        # Model for acceleration -- these come from the model
        accel_x = -k*np.sin(theta)*j2 / m
        accel_z = -g + k*np.cos(theta)*j2 / m

        # Measurements
        y_vec = np.array([x, z, x_dot/z, theta, theta_dot, accel_x, accel_z, k])

        # Return measurement
        return y_vec

############################################################################################
# drone simulation
############################################################################################
def simulate_drone(f, h, tsim_length=20, dt=0.1, measurement_names=None,
                    trajectory_shape='squiggle', setpoint=None, rterm=1e-4):
    """
    trajectory_shape: 'squiggle', 'alternating' 
    """
    # set state and input names
    state_names = f(None, None, return_state_names=True) #['theta', 'theta_dot', 'x', 'x_dot', 'z', 'z_dot', 'k']
    input_names = ['j1', 'j2']
    
    # choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True) 
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names, 
                                   input_names=input_names, measurement_names=measurement_names, mpc_horizon=int(1/dt))

    # First define the set-point(s) to follow
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    if setpoint is None:
        assert trajectory_shape in ['squiggle', 'bigsquiggle', 'alternating', 'random', 'constant_thetadot']

        if trajectory_shape == 'squiggle':
            setpoint = {'theta': NA,
                        'theta_dot': NA,
                        'x': 2.0*np.cos(2*np.pi*tsim*0.3),  # ground speed changes as a sinusoid
                        'x_dot': NA,
                        'z': 0.3*np.sin(2*np.pi*tsim*0.2)+0.5, # altitude also oscillates
                        'z_dot': NA,
                        'k': np.ones_like(tsim),
                       }
        elif trajectory_shape == 'bigsquiggle':
            setpoint = {'theta': NA,
                        'theta_dot': NA,
                        'x': 5.0*np.cos(2*np.pi*tsim*0.3),  # ground speed changes as a sinusoid
                        'x_dot': NA,
                        'z': 5*np.sin(2*np.pi*tsim*0.2)+6, # altitude also oscillates
                        'z_dot': NA,
                        'k': np.ones_like(tsim),
                       }
        elif trajectory_shape == 'alternating':

            a = 0
            b = int(len(tsim)/4.)
            c = int(len(tsim)*2/4.)
            d = int(len(tsim)*3/4.)
            e = -1
            
            accel_x = np.hstack((2.0*np.cos(2*np.pi*tsim*0.3)[a:b],
                              0*tsim[b:c],
                              2.0*np.cos(2*np.pi*tsim*0.3)[c:d],
                              0*tsim[d:e]))
            xvel = np.cumsum(accel_x)*dt
            xpos = 5*np.cumsum(xvel)*dt
            if len(xpos) > len(tsim):
                xpos = xpos[0:len(tsim)]
            if len(xpos) < len(tsim):
                xpos = np.hstack((xpos, [xpos[-1]]*(len(tsim)-len(xpos))))
            
            accel_z = np.hstack((0.1*np.sin(2*np.pi*tsim*0.2)[a:b],
                              0*tsim[b:c],
                              -0.1*np.sin(2*np.pi*tsim*0.2)[c:d],
                              0*tsim[d:e]))
            zvel = np.cumsum(accel_z)*dt
            zpos = 5*np.cumsum(zvel)*dt + 1
            if len(zpos) > len(tsim):
                zpos = zpos[0:len(tsim)]
            if len(zpos) < len(tsim):
                zpos = np.hstack((zpos, [zpos[-1]]*(len(tsim)-len(zpos))))

            setpoint = {'theta': NA,
                        'theta_dot': NA,
                        'x': xpos,  
                        'x_dot': NA,
                        'z': zpos, 
                        'z_dot': NA,
                        'k': np.ones_like(tsim),
                       }
        elif trajectory_shape == 'constant_thetadot':
            freq = 0.15
            amp = 0.15
            theta_dot = amp*np.sign(np.cos(tsim*2*np.pi*freq))

            zpos = np.ones_like(theta_dot)*3

            setpoint = {'theta': NA,
                        'theta_dot': theta_dot,
                        'x': NA,  
                        'x_dot': NA,
                        'z': zpos, 
                        'z_dot': NA,
                        'k': np.ones_like(tsim),
                       }

        elif trajectory_shape == 'random':
            tsim_length_part = tsim_length/3.
            tsim = np.arange(0, tsim_length_part, step=dt)

            x_curve_1 = generate_smooth_curve(tsim, method='spline', smoothness=0.15, amplitude=3.0, seed=42)
            z_curve_1 = generate_smooth_curve(tsim, method='spline', smoothness=0.15, amplitude=3.0, seed=24)

            x_curve_2 = generate_smooth_curve(tsim, method='spline', smoothness=0.04, amplitude=5.0, seed=3)
            z_curve_2 = generate_smooth_curve(tsim, method='spline', smoothness=0.04, amplitude=5.0, seed=3)

            accel_x = 0.1*np.ones_like(x_curve_1)
            xdot = np.cumsum(accel_x)*dt
            x_curve_3 = np.cumsum(xdot)*dt
            z_curve_3 = 4*np.ones_like(x_curve_1)

            tsim = np.arange(0, len(tsim)*3*dt, step=dt)
            tsim_length = len(tsim)*3*dt
            NA = np.zeros_like(tsim)
            setpoint = {'theta': NA,
                        'theta_dot': NA,
                        'x': np.hstack((x_curve_1,x_curve_2,x_curve_3)),  
                        'x_dot': NA,
                        'z': np.hstack((z_curve_1,z_curve_2,z_curve_3)) + 5, 
                        'z_dot': NA,
                        'k': np.ones_like(tsim),
                       }

    if 'k' not in state_names:
        del setpoint['k']

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function: penalize the squared error between the setpoint for g and the true g
    if trajectory_shape in ['squiggle', 'bigsquiggle', 'alternating', 'random']:
        cost_x = (simulator.model.x['x'] - simulator.model.tvp['x_set']) ** 2
        cost_z = (simulator.model.x['z'] - simulator.model.tvp['z_set']) ** 2
        cost = cost_x + cost_z 
    elif trajectory_shape in ['constant_thetadot',]:
        cost_theta_dot = (simulator.model.x['theta_dot'] - simulator.model.tvp['theta_dot_set']) ** 2
        cost_z = (simulator.model.x['z'] - simulator.model.tvp['z_set']) ** 2
        cost = cost_theta_dot + cost_z 


    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)  # objective function

    # Set input penalty: make this small for accurate state tracking
    simulator.mpc.set_rterm(j1=rterm, j2=rterm)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'theta'] = -np.pi/4
    simulator.mpc.bounds['upper', '_x', 'theta'] = np.pi/4
    simulator.mpc.bounds['lower', '_x', 'z'] = 0.0
    simulator.mpc.bounds['lower', '_u', 'j2'] = 0.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, u=None, mpc=True, return_full_output=True)

    # Return
    return t_sim, x_sim, u_sim, y_sim, simulator

def package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim):
    # turn all the sim outputs into pandas dataframes
    df_x = pd.DataFrame(x_sim) # x_sim is a dict
    df_u = pd.DataFrame(u_sim) # u_sim is a dict
    df_y = pd.DataFrame(y_sim) # y_sim is a dict
    df_t = pd.DataFrame({'time': t_sim}) # t_sim is a 1d array, make it a dict
    
    # rename the columns for y so that they do not conflict with state names
    new_names = {key: 'sensor_' + key for key in df_y}
    df_y = df_y.rename(columns=new_names)
    
    # merge into a single data frame for the entire trajectory
    df_trajec = pd.concat([df_t, df_x, df_u, df_y], axis=1)

    return df_trajec
    
###############################################################################################
# Misc helper functions
###############################################################################################

# From Claude
def generate_smooth_curve(t_points, method='spline', smoothness=0.1, amplitude=1.0, seed=None):
    """
    Generate a smooth random curve as a function of time points.
    
    Parameters:
    -----------
    t_points : array-like
        Time points where the curve should be evaluated
    method : str, default='spline'
        Method to use: 'spline', 'sine_sum', or 'noise_filter'
    smoothness : float, default=0.1
        Controls curve smoothness (interpretation varies by method)
    amplitude : float, default=1.0
        Maximum amplitude of the curve
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    numpy.ndarray
        Smooth curve values at the given time points
    """
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    
    t_points = np.array(t_points)
    
    if method == 'spline':
        # Generate random control points and interpolate with splines
        n_control = max(5, int(len(t_points) * smoothness))
        control_t = np.linspace(t_points[0], t_points[-1], n_control)
        control_y = np.random.normal(0, amplitude/3, n_control) # < I modified, used to  be uniform(-amp, amp)
        
        # Use cubic spline interpolation
        spline = interpolate.CubicSpline(control_t, control_y)
        return spline(t_points)
    
    elif method == 'sine_sum':
        # Sum of random sine waves with different frequencies
        n_harmonics = max(3, int(20 * smoothness))
        result = np.zeros_like(t_points, dtype=float)
        
        for i in range(n_harmonics):
            freq = np.random.exponential(1.0 / smoothness)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0, amplitude) / (i + 1)  # Decay higher frequencies
            result += amp * np.sin(2 * np.pi * freq * t_points + phase)
        
        return result
    
    elif method == 'noise_filter':
        # Generate noise and apply low-pass filtering
        from scipy.signal import butter, filtfilt
        
        # Generate random noise
        noise = np.random.normal(0, amplitude, len(t_points))
        
        # Apply low-pass filter
        nyquist = 0.5 * len(t_points) / (t_points[-1] - t_points[0])
        cutoff = nyquist * smoothness
        b, a = butter(3, cutoff / nyquist, btype='low')
        
        return filtfilt(b, a, noise)
    
    else:
        raise ValueError("Method must be 'spline', 'sine_sum', or 'noise_filter'")
