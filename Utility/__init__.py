__version__ = "0.0.36"

from . import install_and_download_requirements

def load_plotting_modules():
    from . import plot_utility

def load_planar_drone_modules():
    from . import planar_drone
    from . import terrain_and_optic_flow_utility
    from . import symbolic_derivatives
    from . import generate_training_data_utility

def load_kalman_filter_modules():
    from . import extended_kalman_filter
    from . import unscented_kalman_filter
    
def load_data_driven_modules():
    from . import pysindy_utility

def load_neural_network_modules():
    from . import keras_ann_utility
    from . import keras_advanced_utility

def check_for_updates(silent=False):
    """Check if a newer version is available on GitHub."""
    import requests
    from packaging.version import Version
    
    # Get current installed version
    current_version = __version__
    
    try:
        # Fetch setup.py to get latest version
        url = "https://raw.githubusercontent.com/florisvb/Nonlinear_and_Data_Driven_Estimation/main/Utility/__init__.py"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # Parse version
        import re
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', response.text)
        
        if match:
            latest_version = match.group(1)
            
            if Version(latest_version) > Version(current_version):
                print(f"⚠ Update available: {current_version} → {latest_version}")
                print(f"   Run: pip install --upgrade git+https://github.com/florisvb/Nonlinear_and_Data_Driven_Estimation.git")
                return (current_version, latest_version, True)
            elif not silent:
                print(f"✓ You have the latest version ({current_version})")
            
            return (current_version, latest_version, False)
    except Exception as e:
        if not silent:
            print(f"Could not check for updates: {e}")
        return (current_version, None, None)
