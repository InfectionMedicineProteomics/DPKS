#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages  # type: ignore

version = "0.1.2"

requirements = [
    "click",
    "numpy",
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
    "shap",
    "imbalanced-learn",
    "kneed",
    "gseapy",
    "unipressed",
    "jupyterlab",
    "tbb"
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
    entry_points={
        "console_scripts": [
            "dpks=dpks.cli:main",
        ],
    },
    install_requires=requirements,
    long_description="Data processing package for the statistical analysis and application of explainable machine learning for omics data.",
    include_package_data=True,
    keywords="dpks",
    name="dpks",
    packages=find_packages(include=["dpks", "dpks.*"], exclude=["tests", "tests.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/InfectionMedicineProteomics/DPKS",
    version=version,
    zip_safe=False,
)
