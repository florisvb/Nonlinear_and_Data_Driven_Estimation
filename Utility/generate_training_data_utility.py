import zipfile
import os
import requests
import pandas as pd

def download_data(filename, giturl=None, unzip=True):
    # Import functions directly from github
    # Important: note that we use raw.githubusercontent.com, not github.com

    if giturl is None:
      giturl = 'https://raw.githubusercontent.com/florisvb/Nonlinear_and_Data_Driven_Estimation/main/Data/' + filename
    
    r = requests.get(giturl)
    print('Fetching from: ')
    print(r)
    
    # Store the file to the colab working directory
    with open(filename, 'wb') as f:
      f.write(r.content)
    f.close()
    
    if unzip:
        print('unzipping...')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
          zip_ref.extractall('.')  # extracts to new directory

def load_trajectory_data(data_path):
    #data_path = 'planar_drone_trajectories'
    all_fnames = os.listdir(data_path)

    traj_list = []
    for fname in all_fnames:
        fname = os.path.join(data_path, fname)
        trajec = pd.read_hdf(fname)
        traj_list.append(trajec)

    print('Number of trajectories: ')
    print(len(traj_list))

    return traj_list

def clean_trajectory_data(traj_list, max_state_value=100):
    cols_to_analyze = traj_list[0].keys().tolist()
    cols_to_analyze.remove('objid')
    
    bad_trajs = []
    for i in range(len(traj_list)):
        if traj_list[i][cols_to_analyze].abs().max().max() > max_state_value:
            bad_trajs.append(i)
    
    traj_list_good = [obj for i, obj in enumerate(traj_list) if i not in bad_trajs]
    
    print('Number of good trajectories: ')
    print(len(traj_list_good))

    return traj_list_good
