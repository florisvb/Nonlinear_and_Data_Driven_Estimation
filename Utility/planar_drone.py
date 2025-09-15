import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize

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
def f(x_vec, u_vec, m=m, g=g, L=L, I=Iyy, return_state_names=False):
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
    
    Returns:
    x_dot : numpy array, shape (7,)
        Time derivative of state vector
    """

    if return_state_names:
        return ['theta', 'theta_dot', 'x', 'x_dot', 'z', 'z_dot', 'k']
    
    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
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
    
    return x_dot_vec


############################################################################################
# continuous time measurement functions
############################################################################################
def h_gps(x_vec, u_vec, return_measurement_names=False):
    if return_measurement_names:
        return ['theta', 'x', 'z', 'k']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Measurements
    y_vec = np.array([theta, x, z, k])

    # Return measurement
    return y_vec

def h_camera_theta_k(x_vec, u_vec, return_measurement_names=False):
    if return_measurement_names:
        return ['optic_flow', 'theta', 'k']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Measurements
    y_vec = np.array([x_dot/z, theta, k])

    # Return measurement
    return y_vec

def h_camera_thetadot_k(x_vec, u_vec, return_measurement_names=False):
    if return_measurement_names:
        return ['optic_flow', 'theta_dot', 'k']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Measurements
    y_vec = np.array([x_dot/z, theta_dot, k])

    # Return measurement
    return y_vec

def h_camera_imu_notheta(x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
    if return_measurement_names:
        return ['optic_flow', 'theta_dot', 'accel_x', 'accel_z']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Model for acceleration -- these come from the model
    accel_x = -k * np.sin(theta) / m
    accel_z = -g + k * np.cos(theta) / m

    # Measurements
    y_vec = np.array([x_dot/z, theta_dot, accel_x, accel_z])

    # Return measurement
    return y_vec

def h_camera_imu(x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
    if return_measurement_names:
        return ['optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Model for acceleration -- these come from the model
    accel_x = -k * np.sin(theta) / m
    accel_z = -g + k * np.cos(theta) / m

    # Measurements
    y_vec = np.array([x_dot/z, theta, theta_dot, accel_x, accel_z])

    # Return measurement
    return y_vec

def h_camera_imu_k(x_vec, u_vec, g=g, m=m, L=L, return_measurement_names=False):
    if return_measurement_names:
        return ['optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z', 'k']

    # Extract state variables
    theta = x_vec[0]
    theta_dot = x_vec[1]
    x = x_vec[2]
    x_dot = x_vec[3]
    z = x_vec[4]
    z_dot = x_vec[5]
    k = x_vec[6]

    # Extract control inputs
    j1 = u_vec[0]
    j2 = u_vec[1]

    # Model for acceleration -- these come from the model
    accel_x = -k * np.sin(theta) / m
    accel_z = -g + k * np.cos(theta) / m

    # Measurements
    y_vec = np.array([x_dot/z, theta, theta_dot, accel_x, accel_z, k])

    # Return measurement
    return y_vec

############################################################################################
# drone simulation
############################################################################################
def simulate_drone(h=h_gps, tsim_length=20, dt=0.1, measurement_names=None, trajectory_shape='squiggle', setpoint=None):
    """
    trajectory_shape: 'squiggle', 'alternating' 
    """
    # set state and input names
    state_names = ['theta', 'theta_dot', 'x', 'x_dot', 'z', 'z_dot', 'k']
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
        assert trajectory_shape in ['squiggle', 'alternating']

        if trajectory_shape == 'squiggle':
            setpoint = {'theta': NA,
                        'theta_dot': NA,
                        'x': 2.0*np.cos(2*np.pi*tsim*0.3),  # ground speed changes as a sinusoid
                        'x_dot': NA,
                        'z': 0.3*np.sin(2*np.pi*tsim*0.2)+0.5, # altitude also oscillates
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

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function: penalize the squared error between the setpoint for g and the true g
    cost_x = (simulator.model.x['x'] - simulator.model.tvp['x_set']) ** 2
    cost_z = (simulator.model.x['z'] - simulator.model.tvp['z_set']) ** 2
    cost = cost_x + cost_z 

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)  # objective function

    # Set input penalty: make this small for accurate state tracking
    simulator.mpc.set_rterm(j1=1e-4, j2=1e-4)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'theta'] = -np.pi/4
    simulator.mpc.bounds['upper', '_x', 'theta'] = np.pi/4
    simulator.mpc.bounds['lower', '_x', 'z'] = 0.0
    simulator.mpc.bounds['lower', '_u', 'j2'] = 0.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, u=None, mpc=True, return_full_output=True)

    # Return
    return t_sim, x_sim, u_sim, y_sim, simulator
