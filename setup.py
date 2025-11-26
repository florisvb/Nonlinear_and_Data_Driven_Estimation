from setuptools import setup, find_packages

def read_requirements():
    """Read requirements from requirements.txt, ignoring comments and blank lines."""
    with open("Requirements/requirements_minimal.txt") as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="nonlinear-data-driven-estimation",
    version="0.0.1",
    packages=["nonlinear_estimation_utilities"],
    package_dir={"nonlinear_estimation_utilities": "Utility"},
    include_package_data=True,
    data_files=[('requirements', [  'Requirements/requirements_minimal.txt',
                                    'Requirements/requirements_pybounds.txt',
                                    'Requirements/requirements_datadriven.txt',
                                    'Requirements/requirements_neuralnetworks.txt',
        ])],
    install_requires=read_requirements(),
    python_requires=">=3.7",
    description="Utilities for nonlinear and data-driven estimation",
    author="Floris van Breugel",  
    author_email="fvanbreugel@unr.edu",  
)