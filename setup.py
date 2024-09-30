from setuptools import setup, find_packages

setup(
    name="digit-classifier",
    version="0.1",
    description="MLP Network for Digit Recognition using MNIST 784 dataset.",
    author="Sebastian Mandal",
    author_email="sebastian.mandal@icloud.com",
    packages=["module"],
    install_requires=[
        "numpy==1.24.2",
        "scikit-learn==1.2.0",
        "matplotlib==3.7.1",
        "pandas==2.0.3",
    ],
)
