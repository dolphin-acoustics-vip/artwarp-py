"""Setup configuration for ARTwarp-py package."""

from setuptools import setup, find_packages
from pathlib import Path

# README file for long description
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="artwarp-py",
    version="2.0.3",
    author="Original ARTwarp by Deecke, V. B. & Janik, V. M. (2006); Python implementation by Pedro Gronda Garrigues (2026)",
    description="High-performance Python implementation of ARTwarp for bioacoustic signal categorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dolphin-acoustics-vip/artwarp-py",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "accelerate": [
            "numba>=0.54.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "all": [
            "numba>=0.54.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "artwarp-py=artwarp.cli.main:main",
        ],
    },
)
