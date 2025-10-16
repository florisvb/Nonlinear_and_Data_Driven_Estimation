import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import scipy.optimize
import pandas as pd
import copy
import sys
import requests
import importlib
import os

import pandas

import casadi
import do_mpc
import pybounds

def import_local_or_github(package_name, function_name=None, directory=None, giturl=None):
    # Import functions directly from github
    # Important: note that we use raw.githubusercontent.com, not github.com

    try: # to find the file locally
        if directory is not None:
            if directory not in sys.path:
                sys.path.append(directory)

        package = importlib.import_module(package_name)
        if function_name is not None:
            function = getattr(package, function_name)
            return function
        else:
            return package

    except: # get the file from github
        if giturl is None:
            giturl = 'https://raw.githubusercontent.com/florisvb/Nonlinear_and_Data_Driven_Estimation/main/Utility/' + str(package_name) + '.py'

        r = requests.get(giturl)
        print('Fetching from: ')
        print(r)

        # Store the file to the colab working directory
        with open(package_name+'.py', 'w') as f:
            f.write(r.text)
        f.close()

        # import the function we want from that file
        package = importlib.import_module(package_name)
        if function_name is not None:
            function = getattr(package , function_name)
            return function
        else:
            return package

def simulate_snippet(dt, tsim_length, trim_indices, 
                     smoothness_x=0.15, smoothness_z=0.15, smoothness_theta=0.15,
                     amplitude_x=3., amplitude_z=3., amplitude_theta=3.,
                     nominal_z=5,
                     seed_x=42, seed_z=24, seed_theta=24,
                     trajectory_shape_x='spline', trajectory_shape_z='spline', trajectory_shape_theta='spline',
                     rterm=1e-1,
                     directory='.',
                     number=1):

    f = planar_drone.F(k=1).f
    h = planar_drone.H('h_camera_imu', k=1).h

    tsim = np.arange(0, tsim_length, step=dt)

    theta_curve = planar_drone.generate_smooth_curve(tsim, method=trajectory_shape_theta, 
                                                 smoothness=smoothness_theta, amplitude=amplitude_theta, seed=seed_theta)
    x_curve = planar_drone.generate_smooth_curve(tsim, method=trajectory_shape_x, 
                                                 smoothness=smoothness_x, amplitude=amplitude_x, seed=seed_x)
    z_curve = planar_drone.generate_smooth_curve(tsim, method=trajectory_shape_z, 
                                                 smoothness=smoothness_z, amplitude=amplitude_z, seed=seed_z)
    z_curve += nominal_z

    if np.min(z_curve) <= 0.01:
        z_curve += -1*np.min(z_curve) + 0.01

    NA = np.zeros_like(tsim)
    setpoint = {'theta': theta_curve,
                'theta_dot': NA,
                'x': x_curve,  
                'x_dot': NA,
                'z': z_curve, 
                'z_dot': NA,
                'k': NA,
               }

    t_sim, x_sim, u_sim, y_sim, simulator = planar_drone.simulate_drone(f, h=h, dt=dt, tsim_length=tsim_length,
                                                                        setpoint=setpoint,
                                                                        rterm=rterm) 

    t_sim_df = pandas.DataFrame({'time': t_sim})
    x_sim_df = pandas.DataFrame(x_sim)
    u_sim_df = pandas.DataFrame(u_sim)
    y_sim_df = pandas.DataFrame(y_sim)

    new_names = {key: 'sensor_' + key for key in y_sim_df}
    y_sim_df = y_sim_df.rename(columns=new_names)

    objid = pd.DataFrame({'objid': (number*np.ones_like(tsim)).astype(int)})

    df = pandas.concat([t_sim_df, objid, x_sim_df, u_sim_df, y_sim_df], axis=1)



    df_trimmed = df.iloc[trim_indices:-trim_indices]

    fname = 'trajectory_' + str(number).zfill(5) + '.hdf'
    fname = os.path.join(directory, fname)

    df_trimmed.to_hdf(fname, 'trajec')


if __name__ == '__main__':
    planar_drone = import_local_or_github('planar_drone', directory='../Utility')

    trajectory_shape_options = ['spline', 'noise_filter']

    first_trajec = 0
    N_random_trajecs = 2000
    N_low_altitude_trajecs = 500
    N_squiggle_trajecs = 500

    dt = 0.1
    tsim_length = 10 # how many seconds long is each trajectory
    trim_indices = 5 # how many indices to drop from the beginning and end to remove initialization issues with MPC

    directory = 'trajectories'

    

    total_number = first_trajec

    if 1:
        for i in range(0, N_random_trajecs):
            total_number += 1

            smoothness_x = np.random.uniform(1e-3, 1)
            smoothness_z = np.random.uniform(1e-3, 1)
            smoothness_theta = np.random.uniform(0, 1)

            amplitude_x = np.random.uniform(1e-3, 10)
            amplitude_z = np.random.uniform(1e-3, 10)
            amplitude_theta = np.random.uniform(1e-3, 10)

            rterm = np.random.uniform(1e-4, 1e-1)

            nominal_z = np.random.uniform(1e-3, 20)

            seed_x = np.random.randint(0, 100)
            seed_z = np.random.randint(0, 100)
            seed_theta = np.random.randint(0, 100)

            simulate_snippet(dt, tsim_length, trim_indices, 
                         smoothness_x, smoothness_z, smoothness_theta,
                         amplitude_x, amplitude_z, amplitude_theta,
                         nominal_z,
                         seed_x, seed_z, seed_theta, 
                         trajectory_shape_x=np.random.choice(trajectory_shape_options), trajectory_shape_z=np.random.choice(trajectory_shape_options), trajectory_shape_theta=np.random.choice(trajectory_shape_options),
                         rterm=rterm,
                         directory=directory,
                         number=total_number)

    # generate some low altitude trajectories
    if 1:
        for i in range(0, N_low_altitude_trajecs):
            total_number += 1

            smoothness_x = np.random.uniform(1e-3, 1)
            smoothness_z = np.random.uniform(1e-3, 1)
            smoothness_theta = np.random.uniform(0, 1)

            amplitude_x = np.random.uniform(1e-3, 10)
            amplitude_z = np.random.uniform(0.001)
            amplitude_theta = np.random.uniform(0.001)

            rterm = np.random.uniform(1e-4, 1e-1)

            nominal_z = np.random.uniform(0.5)

            seed_x = np.random.randint(0, 100)
            seed_z = np.random.randint(0, 100)
            seed_theta = np.random.randint(0, 100)

            simulate_snippet(dt, tsim_length, trim_indices, 
                         smoothness_x, smoothness_z, smoothness_theta,
                         amplitude_x, amplitude_z, amplitude_theta,
                         nominal_z,
                         seed_x, seed_z, seed_theta, 
                         trajectory_shape_x=np.random.choice(trajectory_shape_options), trajectory_shape_z=np.random.choice(trajectory_shape_options), trajectory_shape_theta=np.random.choice(trajectory_shape_options),
                         rterm=rterm,
                         directory=directory,
                         number=total_number)

    # generate some squiggle trajectories
    for i in range(0, N_squiggle_trajecs):
        total_number += 1
        number = total_number

        f = planar_drone.F(k=1).f
        h = planar_drone.H('h_camera_imu', k=1).h

        tsim = np.arange(0, tsim_length, step=dt)


        freq_x = np.random.uniform(1e-3, 1)
        freq_z = np.random.uniform(1e-3, 1)

        amplitude_x = np.random.uniform(1e-3, 2)
        amplitude_z = np.random.uniform(1e-3, 2)

        rterm = np.random.uniform(1e-4, 1e-1)
        
        nominal_z = np.random.uniform(1e-3, 20)
        NA = np.zeros_like(tsim)
        setpoint = {'theta': NA,
                    'theta_dot': NA,
                    'x': amplitude_x*np.cos(2*np.pi*tsim*freq_x),  # ground speed changes as a sinusoid
                    'x_dot': NA,
                    'z': amplitude_z*np.sin(2*np.pi*tsim*freq_z)+nominal_z, # altitude also oscillates
                    'z_dot': NA,
                    'k': np.ones_like(tsim),
                   }

        t_sim, x_sim, u_sim, y_sim, simulator = planar_drone.simulate_drone(f, h=h, dt=dt, tsim_length=tsim_length,
                                                                            setpoint=setpoint,
                                                                            rterm=rterm) 

        t_sim_df = pandas.DataFrame({'time': t_sim})
        x_sim_df = pandas.DataFrame(x_sim)
        u_sim_df = pandas.DataFrame(u_sim)
        y_sim_df = pandas.DataFrame(y_sim)

        new_names = {key: 'sensor_' + key for key in y_sim_df}
        y_sim_df = y_sim_df.rename(columns=new_names)

        objid = pd.DataFrame({'objid': (number*np.ones_like(tsim)).astype(int)})

        df = pandas.concat([t_sim_df, objid, x_sim_df, u_sim_df, y_sim_df], axis=1)



        df_trimmed = df.iloc[trim_indices:-trim_indices]

        fname = 'trajectory_' + str(number).zfill(5) + '.hdf'
        fname = os.path.join(directory, fname)

        df_trimmed.to_hdf(fname, 'trajec')
