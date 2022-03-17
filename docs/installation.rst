.. highlight:: shell

============
Installation
============

Dependencies
------------

On ubuntu::

     apt install libblas3 liblapack3 liblapack-dev libblas-dev libatlas-base-dev gfortran

Install DPKS
------------

The sources for dpks can be downloaded from the `Github repo`_.

.. _Github repo: git://github.com/InfectionMedicineProteomics/DPKS

You can clone the repository::

    git clone git://github.com/InfectionMedicineProteomics/DPKS

To install dpks, run this command in your terminal::

    cd DPKS && pip install .

If you are developing DPKS, run this command in your terminal::

    pip install tox flake8 flake8-html coverage Sphinx sphinx-material sphinx-copybutton sphinx-autodoc-typehints sphinxcontrib-autoyaml pytest-sphinx sphinx-click pytest pytest-html pytest-cov black mypy bandit
    cd DPKS && pip install -e .

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

You can also install it with::

    python setup.py install

Documentation
-------------

To build the documentation::

    sphinx-build -b html docs <path_to_site>

Tests
-----

To test dpks::

    pytest

To test the quantification module::

    pytest -m quantification

All test tags are listed in pytest.ini

DocTests
--------

To execute the doctests for the normalize module::

    python3 -m doctest -v dpks/normalization.py

New Release
-----------

Check the current release::

    git tag

Describe the new release in HISTORY.rst

Bump the version::

    bump2version <relase type, e.g. patch> setup.py

Push the tag to the repository::

    git push origin <version tag>
