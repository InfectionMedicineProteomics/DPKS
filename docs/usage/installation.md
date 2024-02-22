# Installation

## Install DPKS

The sources for dpks can be downloaded from the [Github repo](git://github.com/InfectionMedicineProteomics/DPKS).

You can clone the repository:

``` shell
git clone git://github.com/InfectionMedicineProteomics/DPKS
```

To install dpks, run this command in your terminal:

``` shell
cd DPKS && pip install .
```

If you are developing DPKS, run this command in your terminal:

``` shell
pip install tox flake8 flake8-html coverage pytest pytest-html pytest-cov black mypy bandit
cd DPKS && pip install -e .
```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.
