=====
Usage
=====

Notebooks
---------

There are notebooks with examples on how to use dpks in `GitHub Notebooks`_; the notebooks are listed in the table below.

.. list-table:: Example notebooks
   :widths: 30 70
   :header-rows: 1

   * - Notebook
     - Description
   * - `Quant Matrix`_
     - Demonstrates the basic use of the QuantMatrix. Start here.
   * - `Differential Expression`_
     - Demonstrates how to compute differences between two experimental conditions.
   * - `RT sliding window normalization`_
     - Demonstrates how to used the sliding retention time normalization.

.. _GitHub Notebooks: https://github.com/InfectionMedicineProteomics/DPKS/tree/main/notebooks
.. _Differential Expression: https://github.com/InfectionMedicineProteomics/DPKS/blob/main/notebooks/differential_expression.ipynb
.. _Quant Matrix: https://github.com/InfectionMedicineProteomics/DPKS/blob/main/notebooks/quant_matrix.ipynb
.. _RT sliding window normalization: https://github.com/InfectionMedicineProteomics/DPKS/blob/main/notebooks/rt_sliding_window_normalization.ipynb


Getting started
---------------

Instanciate a QuantMatrix::

    from dpks.quant_matrix import QuantMatrix

    quant_matrix = QuantMatrix(
        quantification_file="matrix.tsv",
        design_matrix_file="design_matrix.tsv"
    )
    quant_matrix.to_df()
