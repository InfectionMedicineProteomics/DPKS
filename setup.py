#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages  # type: ignore

version = "0.1.5"

install_requires = [
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
    "inmoose"
]


setup(
    name="dpks",
    author="Aaron Scott",
    author_email="aaron.scott@med.lu.se",
    install_requires=install_requires,
    long_description="Data processing package for the statistical analysis and application of explainable machine learning for omics data.",
    include_package_data=True,
    packages=find_packages(include=["dpks", "dpks.*"]),
    url="https://github.com/InfectionMedicineProteomics/DPKS",
    version=version,
)
