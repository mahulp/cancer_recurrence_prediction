# setup.py

from setuptools import setup, find_packages

setup(
    name="cancer_recurrence_prediction",
    version="0.1.0",
    description="Quantum-classical cancer recurrence prediction package",
    author="Mahul Pandey",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pennylane>=0.32.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
) 
