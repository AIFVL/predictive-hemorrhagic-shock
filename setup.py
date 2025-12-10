# Este archivo se mantiene por compatibilidad, pero el proyecto ahora usa Poetry
# Consulta pyproject.toml para la configuración principal del proyecto

from setuptools import setup, find_packages

setup(
    name="shock",
    version="0.1.0",
    description="Project for hemorrhagic shock prediction using MLflow",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "scikit-learn>=1.7.2",
        "mlflow>=2.17.1",
        "python-decouple>=3.8",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.28.0",
        ],
    },
)
