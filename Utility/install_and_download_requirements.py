import subprocess
import sys
import importlib
from packaging.version import Version
from importlib import resources
import re

def load_requirements(requirements_file='requirements_pybounds.txt'):
    """Load requirements file (Python 3.7+ compatible)."""
    try:
        # Try Python 3.9+ API
        from importlib import resources
        return resources.files(f'{__package__}.Requirements').joinpath(requirements_file).read_text()
    except AttributeError:
        # Fallback to Python 3.7-3.8 API
        import importlib.resources as pkg_resources
        import nonlinear_estimation_utilities.Requirements as req_pkg
        return pkg_resources.read_text(req_pkg, requirements_file)

def parse_requirements(requirements_text):
    """
    Parse requirements text into a list of [package_name, version_requirement].
    
    Handles extras like package[extra] or package[extra1,extra2].
    
    Args:
        requirements_text (str): Content of a requirements.txt file
        
    Returns:
        list: List of [package_name, version_requirement] pairs.
              version_requirement is None if not specified.
              Package name includes extras in brackets if present.
    """
    parsed = []
    
    for line in requirements_text.strip().split('\n'):
        # Remove whitespace
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Remove inline comments
        if '#' in line:
            line = line.split('#')[0].strip()

        # handle git+ installs 
        if 'git+' in line:
            parsed.append([line, None])
            
        else:
            # Match package name (with optional extras) and version specifier
            # Pattern: package_name[extras]version_spec
            # Examples: numpy, pandas[excel], scipy>=1.0, requests[security]>=2.0
            match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[a-zA-Z0-9_,-]+\])?)(.*?)$', line)
            
            if match:
                package_name = match.group(1)
                version_spec = match.group(2).strip()
                
                # If no version specified, set to None
                if not version_spec:
                    version_spec = None
                
                parsed.append([package_name, version_spec])
    
    return parsed

def check_version_requirement(installed_version, requirement):
    """
    Check if an installed version meets a requirement.
    Handles complex requirements like '>=1.0,<2.0'.
    
    Args:
        installed_version (str): The installed version (e.g., '1.2.3')
        requirement (str): Version requirement (e.g., '>=1.0,<2.0')
        
    Returns:
        bool: True if requirement is met, False otherwise
    """
    if requirement is None:
        return True
    
    # Clean up installed version
    installed_version = str(installed_version).strip()

    # make robust to dev version
    installed_version = installed_version.split('.dev')[0]

    # clean up requirement version
    requirement = str(requirement).strip()
    
    try:
        installed = Version(installed_version)
    except Exception as e:
        print(f"Error parsing installed version '{installed_version}': {e}")
        return False
    
    # Split by comma for compound requirements
    conditions = [c.strip() for c in requirement.split(',')]
    
    for condition in conditions:
        # Parse each condition
        match = re.match(r'^(==|>=|<=|>|<|!=)?(.+)$', condition)
        
        if not match:
            return False
        
        operator = match.group(1)
        required_version = match.group(2).strip()
        
        # If no operator, assume '=='
        if operator is None:
            operator = '=='
        
        try:
            required = Version(required_version)
        except Exception as e:
            print(f"Error parsing required version '{required_version}': {e}")
            return False
        
        # Check condition
        result = False
        if operator == '==':
            result = (installed == required)
        elif operator == '>=':
            result = (installed >= required)
        elif operator == '<=':
            result = (installed <= required)
        elif operator == '>':
            result = (installed > required)
        elif operator == '<':
            result = (installed < required)
        elif operator == '!=':
            result = (installed != required)
        
        if not result:
            return False
    
    # All conditions passed
    return True

def verify_package(package_name, version_requirement):
    """Check if a package meets version requirements."""
    try:
        # Import the package
        pkg = importlib.import_module(package_name)
        
        # Get installed version
        installed_version = str(pkg.__version__)

        # Check requirement
        meets_requirement = check_version_requirement(installed_version, version_requirement)
        
        if meets_requirement:
            print(f"✓ {package_name} {installed_version} meets requirement {version_requirement}")
        else:
            print(f"✗ {package_name} {installed_version} does NOT meet requirement {version_requirement}")
        
        return meets_requirement
    except ImportError:
        print(f"✗ {package_name} is not installed")
        return False
    except AttributeError:
        print(f"⚠ {package_name} has no __version__ attribute")
        return None

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
    if 'git+' in package_name:
        package_base_name = package_name.split('/')[-1]
    else:
        package_base_name = package_name.split('[')[0]

    try:
        importlib.import_module(package_base_name)
        if required_version is not None:
            success = verify_package(package_base_name, required_version)
            if not success:
                raise ValueError('Wrong package version')
        print(f"Already installed: " + package_name)

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
                success = verify_package(package_base_name, required_version)
                if not success:
                    raise ValueError('Wrong package version')

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

def install_requirements(requirements_file):
    req_raw = load_requirements(requirements_file)
    req_parsed = parse_requirements(req_raw)
    for req in req_parsed:
        install_package(req[0], req[1])

def install_planar_drone_requirements():
    install_requirements(requirements_file='requirements_pybounds.txt')

def install_data_driven_requirements():
    install_requirements(requirements_file='requirements_datadriven.txt')

def install_neural_network_requirements():
    install_requirements(requirements_file='requirements_neuralnetworks.txt')
