import numpy as np
import random

import numpy as np
import scipy
from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt
import copy

import cv2

import pynumdiff

import pandas as pd
import os


###########################################################################################
# Terrain functions
###########################################################################################

# From Claude
def generate_smooth_random_function(seed=None):
    """
    Generates a smooth random function for domain [-30, 30] with range [0, 20].
    
    Parameters:
    -----------
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    function : callable
        A function that takes x and returns smooth random values
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate random parameters for multiple sine waves
    n_waves = random.randint(3, 6)
    amplitudes = []
    frequencies = []
    phases = []
    
    for _ in range(n_waves):
        amplitudes.append(random.uniform(0.5, 2.0))
        frequencies.append(random.uniform(0.05, 0.3))
        phases.append(random.uniform(0, 2 * np.pi))
    
    def smooth_function(x):
        """
        Smooth random function of x.
        
        Parameters:
        -----------
        x : float or array-like
            Input value(s) in domain [-30, 30]
            
        Returns:
        --------
        float or array
            Output value(s) in range [0, 20]
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        # Sum of sine waves
        for amp, freq, phase in zip(amplitudes, frequencies, phases):
            result += amp * np.sin(freq * x + phase)
        
        # # Normalize to [0, 20] range
        # min_val = np.min(result) if result.size > 1 else result
        # max_val = np.max(result) if result.size > 1 else result
        
        # # Scale to [0, 20]
        # if max_val != min_val:
        #     result = 20 * (result - min_val) / (max_val - min_val)
        # else:
        #     result = np.full_like(result, 10.0)
        
        return result*2 #+ np.random.normal(0, 0.1, len(x))
    
    return smooth_function 

###########################################################################################
# Terrain
###########################################################################################

class Terrain:
    def __init__(self, x_min=-100, x_max=100, x_resolution=0.01, seed=42):
        self.terrain_function = generate_smooth_random_function(seed=seed)
        self.xs = np.arange(x_min, x_max, x_resolution)
        self.terrain = self.terrain_function(self.xs)

        # high frequency color
        self.color = np.random.uniform(0, 1, len(self.xs))
        self.color, _ = pynumdiff.butterdiff(self.color, 1, [2, 0.04])

        # add some low frequency content
        color_function = generate_smooth_random_function(seed=seed*2)
        self.color += color_function(self.xs)*0.02

        # normalize
        self.color -= np.min(self.color)
        self.color /= np.max(self.color)

        # add some high frequency noise
        noise = np.random.normal(0,0.1,len(self.color))
        noise, _ = pynumdiff.butterdiff(noise, 1, [4, 0.3])
        self.color += noise

        # calculate terrain curve, used for ray tracing
        self.__update_curve_points__()

    def __update_curve_points__(self):
        self.curve_points = np.vstack([self.xs, self.terrain]).T
                                       
    def get_elevation(self, x):
        ix = np.argmin( np.abs(self.xs-x) )
        return self.terrain[ix]

def adjust_trajec_altitude(trajec, terrain, plot=False):

    ventral_elevation = [terrain.get_elevation(trajec.x.values[i]) for i in range(len(trajec.x.values))]
    ventral_elevation_array = np.hstack(ventral_elevation)
    ventral_altitude = trajec.z.values - ventral_elevation_array

    min_altitude = np.min(ventral_altitude)
    altitude_adjustment = 0
    if min_altitude < 0.5:
        altitude_adjustment = np.abs(min_altitude) + 0.5

    if plot:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        
        ax.scatter(terrain.xs, terrain.terrain, c=terrain.color, s=1)
        ax.set_aspect('equal')
        
        ax.plot(trajec.x.values, trajec.z.values + altitude_adjustment, color='red')

        ax.set_title('drone trajec and terrain after adjustment')

    trajec.z += altitude_adjustment
    trajec['ventral_altitude'] = trajec.z.values - ventral_elevation_array

    return trajec

###########################################################################################
# Ray Tracing
###########################################################################################

# from Claude
class RayCurveIntersector:
    """
    Fast ray-curve intersection with precomputation for fixed origin.
    """
    
    def __init__(self, terrain, origin):
        """
        Parameters:
        -----------
        curve_points : np.ndarray, shape (N, 2)
            Array of [x, y] points defining the curve
        origin : tuple or array
            [x, y] starting point of the ray (fixed)
        """
        self.curve_points = terrain.curve_points
        self.origin = np.array(origin)
        
        # Precompute segment vectors
        self.p1 = self.curve_points[:-1]  # Shape (N-1, 2)
        self.p2 = self.curve_points[1:]   # Shape (N-1, 2)
        self.seg_vec = self.p2 - self.p1  # Segment vectors (sx, sy)
        
        # Precompute origin-relative vectors
        self.p1_rel = self.p1 - self.origin  # p1 - origin
        
    def find_intersection(self, angle):
        """
        Find the curve point closest to where a ray intersects the curve.
        
        Parameters:
        -----------
        angle : float
            Angle in radians, where 0 is down (-y direction), 
            positive angles go counterclockwise
        
        Returns:
        --------
        int or None
            Index of the closest curve point to the intersection
        """
        # Ray direction
        angle_standard = angle - np.pi/2
        dx, dy = np.cos(angle_standard), np.sin(angle_standard)
        
        # Vectorized computation for all segments
        sx = self.seg_vec[:, 0]
        sy = self.seg_vec[:, 1]
        p1x_rel = self.p1_rel[:, 0]
        p1y_rel = self.p1_rel[:, 1]
        
        # Compute denominators for all segments
        denom = dx * sy - dy * sx
        
        # Mask for non-parallel segments
        valid = np.abs(denom) > 1e-10
        
        if not np.any(valid):
            return None
        
        # Compute t and s parameters only for valid segments
        t = np.full(len(denom), np.inf)
        s = np.zeros(len(denom))
        
        t[valid] = (p1x_rel[valid] * sy[valid] - p1y_rel[valid] * sx[valid]) / denom[valid]
        s[valid] = (p1x_rel[valid] * dy - p1y_rel[valid] * dx) / denom[valid]
        
        # Find valid intersections (t >= 0, 0 <= s <= 1)
        intersection_mask = valid & (t >= 0) & (s >= 0) & (s <= 1)
        
        if not np.any(intersection_mask):
            return None
        
        # Find closest intersection
        valid_t = t[intersection_mask]
        valid_indices = np.where(intersection_mask)[0]
        valid_s = s[intersection_mask]
        
        closest_idx = np.argmin(valid_t)
        best_segment_idx = valid_indices[closest_idx]
        best_s = valid_s[closest_idx]
        
        # Calculate intersection point
        intersection = self.p1[best_segment_idx] + best_s * self.seg_vec[best_segment_idx]
        
        # Find closer endpoint
        dist1 = np.linalg.norm(intersection - self.p1[best_segment_idx])
        dist2 = np.linalg.norm(intersection - self.p2[best_segment_idx])
        
        if dist1 < dist2:
            return best_segment_idx
        else:
            return best_segment_idx + 1

def get_image(terrain, origin, plot=False, show_image_in_plot=False):
    
    # Create intersector with fixed origin
    intersector = RayCurveIntersector(terrain, origin=origin)
    
    # Now evaluate for many angles efficiently
    angles = np.arange(-np.pi/3, np.pi/3, 0.01)
    idxs = []
    for angle in angles:
        idx = intersector.find_intersection(angle)
        idxs.append(idx)

    # xs and altitudes
    xs = terrain.xs[idxs] - origin[0]
    altitudes = origin[1] - terrain.terrain[idxs]
    

    # build image
    image = np.array([terrain.color[idx] for idx in idxs])

    # get distance for each ray
    ray_distances = []
    for ix in idxs:
        terrain_location = terrain.curve_points[ix,:]
        ray_distances.append(np.linalg.norm( np.array(origin) - terrain_location))
    ray_distances = np.array(ray_distances)
    
    if plot:
        fig = plt.figure(figsize=(8,5))
        ax2 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        
        ax2.scatter(terrain.xs, terrain.terrain, c=terrain.color, s=1)
        ax2.set_aspect('equal')
        
        for ix in idxs:
            terrain_location = terrain.curve_points[ix,:]
            ray = np.vstack([origin, terrain_location])
        
            ax2.plot(ray[:,0], ray[:,1], color='red', linewidth=0.2)
        
        if show_image_in_plot: 
            ax1 = fig.add_axes([0.2, 0.9, 0.7, 0.05])
            ax1.imshow(np.atleast_2d(image))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('image, drone and terrain')

    return angles, xs, altitudes, image, ray_distances


def get_imgs_and_analytic_optic_flows(trajec_adj, terrain):
    '''
    Calibration factor was determined empirically. Depends on "camera resolution" and time step.
    '''
    
    imgs = []
    ray_distances = []
    analytic_optic_flows = []
    for i in range(trajec_adj.shape[0]):
        pos = [trajec_adj.x.values[i], trajec_adj.z.values[i]]
        xdot = trajec_adj.x_dot.values[i]
        zdot = trajec_adj.z_dot.values[i]

        angles, xs, altitudes, img, ray_distance = get_image(terrain, pos, plot=False)
        imgs.append(img)
        ray_distances.append(ray_distance)

        analytic_optic_flow = (xdot / altitudes - xs*zdot/altitudes**2)*np.cos(angles)**2
        analytic_optic_flows.append(analytic_optic_flow)

    imgs = np.vstack(imgs).T
    analytic_optic_flows = np.vstack(analytic_optic_flows).T
    ray_distances = np.vstack(ray_distances).T
    
    return imgs, -1*analytic_optic_flows[1:], ray_distances



if __name__ == '__main__':

    directory = '/home/caveman/Sync/LAB_Private/COURSES/Nonlinear_Estimation/2025_fall/Nonlinear_and_Data_Driven_Estimation/Data/planar_drone_trajectories_opticflow'

    ##################################################################################################
    import sys
    import requests
    import importlib

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

    planar_drone = import_local_or_github('planar_drone', directory='../Utility')
    plot_tme = import_local_or_github('plot_utility', 'plot_tme', directory='../Utility')
    generate_training_data_utility = import_local_or_github('generate_training_data_utility', directory='../Utility')
    keras_ann_utility = import_local_or_github('keras_ann_utility', directory='../Utility')
    keras_advanced_utility = import_local_or_github('keras_advanced_utility', directory='../Utility')
    ##################################################################################################

    ##################################################################################################
    def save_data(data, directory, fname):
        df = pd.DataFrame(data)
        df.to_hdf(os.path.join(directory, fname), fname.split('_')[0])
    ##################################################################################################

    ##################################################################################################
    # Load data
    generate_training_data_utility.download_data('planar_drone_trajectories.zip')
    traj_list = generate_training_data_utility.load_trajectory_data('planar_drone_trajectories')
    #traj_list = traj_list[0:10]
    #traj_list = generate_training_data_utility.clean_trajectory_data(traj_list)
    #traj_list = generate_training_data_utility.add_noise_to_trajectory_data(traj_list, 0.02)

    ##################################################################################################
    # run loop for trajecs
    for trajec in traj_list:
        #print(trajec.objid.unique()[0])
        
        # generate terrain (using obj id as seed to make reproducible)
        terrain = Terrain(seed=trajec.objid.unique()[0])

        # adjust trajectory height
        trajec_adj = adjust_trajec_altitude(trajec, terrain, plot=False)
        
        # calculate images, optic flows
        try:
            imgs, analytic_optic_flows, ray_distances = get_imgs_and_analytic_optic_flows(trajec_adj, terrain)
        except:
            print('Error -- probably drone went out of range.. skipping: ' + str(trajec.objid.unique()[0]))
            continue

        # Save all the data
        trajec_fname = 'trajectoryadj_' + str(trajec.objid.unique()[0]).zfill(5) + '.hdf'
        optic_flow_fname = 'analyticopticflows_' + str(trajec.objid.unique()[0]).zfill(5) + '.hdf'
        ray_dist_fname = 'raydistances_' + str(trajec.objid.unique()[0]).zfill(5) + '.hdf'
        img_fname = 'imgs_' + str(trajec.objid.unique()[0]).zfill(5) + '.hdf'

        save_data(trajec_adj, directory, trajec_fname)
        save_data(analytic_optic_flows.T, directory, optic_flow_fname)
        save_data(ray_distances.T, directory, ray_dist_fname)
        save_data(imgs.T, directory, img_fname)