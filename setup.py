from setuptools import setup, find_packages

setup(
    name="shock",
    version="0.1.0",
    description="PySpark data analysis project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pyspark>=3.5.0",
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "python-decouple>=3.8",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "pylint>=3.0.3",
            "mypy>=1.8.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shock-etl=etl.main:main",
            "shock-analysis=analysis.main:main",
        ],
    },
)
