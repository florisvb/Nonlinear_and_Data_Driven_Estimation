import subprocess
import sys
import importlib
from packaging.version import Version

def check_package_version(package_name, required_version=None):
    imported_package = importlib.import_module(package_name)
    if Version(imported_package.__version__) < Version(required_version):
        raise ValueError(package_name + " version is: " + imported_package.__version__ + "This notebook requires version " + required_version + " or greater.")
        return False
    else:
        print(package_name + " version is: " + imported_package.__version__)
        return True

# From Claude, with my edits
def install_package(package_name, required_version=None):
    """
    Install a Python package using pip, suppressing output. If there is a required (minimum) version, check that we meet that version requirement. 
    
    Args:
        package_name (str): Name of the package to install
        required_version (str): Version string, like "2.0.0"
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        importlib.import_module(package_name)
        success = check_package_version(package_name, required_version=required_version)

    except:
        print(f"Attempting to pip install: " + package_name)
        try:
            # Capture all output from pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully installed {package_name}")

            if required_version is not None:
                success = check_package_version(package_name, required_version=required_version)

        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e.stderr}")
            return False


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

def install_basic_requirements():
    install_package('pynumdiff')

def install_pybounds_requirements():
    install_package(casadi)

def install_data_driven_requirements():
    install_package('pysindy[miosr]')

def install_neural_network_requirements():
    pass