#!/usr/bin/env python3
"""
UGRO: Unified GPU Resource Orchestrator

Personal-scale GPU cluster management and distributed training orchestration.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Project metadata
PROJECT_NAME = "ugro"
VERSION = "0.1.0"
DESCRIPTION = "Personal-scale GPU cluster management and distributed training orchestration"
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/obuitelaar/ugro"
AUTHOR = "Oliver Buitelaar"
AUTHOR_EMAIL = "buitelaar@gmail.com"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.10"

# Core dependencies from pixi.toml
INSTALL_REQUIRES = [
    "click>=8.1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "paramiko>=3.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
    "python-dotenv>=1.2.1,<2",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.0.0",
        "mkdocstrings[python]>=0.24.0",
    ],
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.10.0",
        "coverage>=7.0.0",
    ],
}

# Include all extras in "all"
EXTRAS_REQUIRE["all"] = [
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
]

# Package classification
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: System :: Distributed Computing",
    "Topic :: System :: Systems Administration",
]

# Entry points (CLI commands)
ENTRY_POINTS = {
    "console_scripts": [
        "ugro=ugro.cli:cli",
    ],
}

# Package discovery
PACKAGES = find_packages(where="src")
PACKAGE_DIR = {"": "src"}

# Include data files
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json", "*.md"],
}

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    license_files=["LICENSE"],
    
    # Package configuration
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    
    # Dependencies
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Classification
    classifiers=CLASSIFIERS,
    
    # Additional metadata
    keywords="gpu cluster distributed training machine learning pytorch",
    project_urls={
        "Bug Reports": "https://github.com/obuitelaar/ugro/issues",
        "Source": "https://github.com/obuitelaar/ugro",
        "Documentation": "https://github.com/obuitelaar/ugro/docs",
    },
    
    # Build configuration
    zip_safe=False,
    platforms=["any"],
)
