import os
import shutil
import random
from pathlib import Path
from typing import Set

# From Claude
def copy_random_file_sets(source_dir: str, dest_dir: str, num_sets: int = 100, include_imgs=False) -> None:
    """
    Randomly select and copy sets of related files from source to destination directory.
    
    Each set consists of four files with matching numbers:
    - analyticopticflows_XXXXX.hdf
    - imgs_XXXXX.hdf <<<<<<<<<<<<<<<<<<<<<<< Set include_imgs = True to include these
    - raydistances_XXXXX.hdf
    - trajectoryadj_XXXXX.hdf
    
    Args:
        source_dir: Path to the source directory containing the files
        dest_dir: Path to the destination directory where files will be copied
        num_sets: Number of complete sets to copy (default: 100)
    
    Raises:
        ValueError: If not enough complete sets are found in the source directory
        FileNotFoundError: If source directory doesn't exist
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Verify source directory exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Define the expected file prefixes
    if include_imgs:
        prefixes = ['analyticopticflows', 'imgs', 'raydistances', 'trajectoryadj']
    else:
        prefixes = ['analyticopticflows', 'raydistances', 'trajectoryadj']
    
    # Find all unique set numbers by looking for one of the file types
    set_numbers: Set[str] = set()
    
    for file in source_path.glob('*.hdf'):
        for prefix in prefixes:
            if file.name.startswith(prefix + '_'):
                # Extract the number part (XXXXX)
                number_part = file.name[len(prefix) + 1:-4]  # Remove prefix_ and .hdf
                set_numbers.add(number_part)
                break
    
    # Verify each set has all four files
    complete_sets = []
    for number in set_numbers:
        files_in_set = []
        all_files_exist = True
        
        for prefix in prefixes:
            filename = f"{prefix}_{number}.hdf"
            filepath = source_path / filename
            
            if filepath.exists():
                files_in_set.append(filepath)
            else:
                all_files_exist = False
                break
        
        if all_files_exist:
            complete_sets.append(files_in_set)
    
    # Check if we have enough complete sets
    if len(complete_sets) < num_sets:
        raise ValueError(
            f"Not enough complete sets found. Requested: {num_sets}, "
            f"Available: {len(complete_sets)}"
        )
    
    # Randomly select the requested number of sets
    selected_sets = random.sample(complete_sets, num_sets)
    
    # Copy the selected files
    copied_count = 0
    for file_set in selected_sets:
        for filepath in file_set:
            dest_file = dest_path / filepath.name
            shutil.copy2(filepath, dest_file)
        copied_count += 1
    
    print(f"Successfully copied {copied_count} sets ({copied_count * 4} files) "
          f"from '{source_dir}' to '{dest_dir}'")
    print(f"Total complete sets available: {len(complete_sets)}")


if __name__ == "__main__":
    # Example usage
    source_directory = '/home/caveman/Sync/LAB_Private/COURSES/Nonlinear_Estimation/2025_fall/Nonlinear_and_Data_Driven_Estimation/Data/planar_drone_trajectories_opticflow'
    destination_directory = '/home/caveman/Sync/LAB_Private/COURSES/Nonlinear_Estimation/2025_fall/Nonlinear_and_Data_Driven_Estimation/Data/planar_drone_trajectories_opticflow_subset'
    
    try:
        copy_random_file_sets(source_directory, destination_directory, num_sets=300)
    except Exception as e:
        print(f"Error: {e}")