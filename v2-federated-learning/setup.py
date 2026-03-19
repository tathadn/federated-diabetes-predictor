"""Package setup for v2-federated-learning."""

from setuptools import find_packages, setup

setup(
    name="federated_diabetes_prediction",
    version="2.0.0",
    description="Federated learning for diabetes prediction using the Flower framework",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "flwr[simulation]>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "pytest-cov>=3.0.0"],
        "notebook": ["jupyter>=1.0.0", "plotly>=5.0.0"],
        "tf": ["tensorflow>=2.7.0"],
    },
)
