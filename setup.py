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
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.7",
    description="Utilities for nonlinear and data-driven estimation",
    author="Floris van Breugel",  
    author_email="fvanbreugel@unr.edu",  
)