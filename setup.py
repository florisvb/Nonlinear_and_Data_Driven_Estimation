from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'Utility', '__init__.py')
    with open(init_path) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.0.0"

def read_requirements():
    with open("requirements_minimal.txt") as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="nonlinear-data-driven-estimation",
    version=get_version(),  # Automatically read from __init__.py
    packages=["nonlinear_estimation_utilities"],
    package_dir={"nonlinear_estimation_utilities": "Utility"},
    include_package_data=True,
    package_data={
        'nonlinear_estimation_utilities': ['Requirements/*.txt', 'Requirements/*'],
    },
    install_requires=read_requirements(),
    python_requires=">=3.7",
    description="Utilities for nonlinear and data-driven estimation",
    author="Floris van Breugel",  
    author_email="fvanbreugel@unr.edu",  
)