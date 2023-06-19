#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages  # type: ignore

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy==1.21.5",
    "numba",
    "scipy",
    "networkx",
    "pandas",
    "seaborn",
    "anndata",
    "scikit-learn",
    "statsmodels",
    "biopython",
    "pytest",
    "pytest-runner",
    "xgboost",
    "shap"
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Aaron Scott",
    author_email="aaron.scott@med.lu.se",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Data processing package for the analysis of mass spectrometry proteomics data",
    entry_points={
        "console_scripts": [
            "dpks=dpks.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="dpks",
    name="dpks",
    packages=find_packages(include=["dpks", "dpks.*"], exclude=["tests", "tests.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/arnscott/dpks",
    version="0.1.1",
    zip_safe=False,
)
