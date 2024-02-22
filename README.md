<p>
    <img src="/docs/img/logo.png", width="100" />
</p>


# DPKS (Data Processing Kitchen Sink)

[![image](https://img.shields.io/pypi/v/dpks.svg)](https://pypi.python.org/pypi/dpks) [![image](https://img.shields.io/travis/arnscott/dpks.svg)](https://travis-ci.com/arnscott/dpks) [![Documentation Status](https://readthedocs.org/projects/dpks/badge/?version=latest)](https://dpks.readthedocs.io/en/latest/?badge=latest)

DPKS provides easily accesible data processing and explainable machine learning for omics data.

-   Free software: MIT license
-   Documentation: [here](https://infectionmedicineproteomics.github.io/DPKS/).


## Overview

DPKS is a comprehensive python library for statistical analysis and explainable machine learning
for omics data. DPKS allows for easily configurable and reproducible analysis pipelines that will simplify
workflows and allow for exploration. Additionally, it exposes advances explainable machine learning techniques
with a simple API allowing them to be used by non-machine learning practicioners in the field.

<figure>
    <img src="/docs/img/dpks_overview_figure.png">
    <figcaption>An overview of DPKS and some of its main functionality.</figcaption>
</figure>


From the abstract from our preprint:


The application of machine learning algorithms to facilitate the
understanding of changes in proteome states has emerged as a promising
methodology in proteomics research. Unfortunately, these methods can prove
difficult to interpret, as it may not be immediately obvious how models reach
their predictions. We present the data processing kitchen sink (DPKS) which provides
reproducible access to classic statistical methods and advanced explainable
machine learning algorithms to build highly accurate and fully interpretable
predictive models. In DPKS, explainable machine learning methods are used to
calculate the importance of each protein towards the prediction of a model for a
particular proteome state. The calculated importance of each protein can enable the
identification of proteins that drive phenotypic change in a data-driven manner
while classic techniques rely on arbitrary cutoffs that may exclude important
features from consideration. DPKS is a free and open source Python package available at [https://github.com/InfectionMedicineProteomics/DPKS](https://github.com/InfectionMedicineProteomics/DPKS). [^1]

## Example

DPKS leverages method chaining, allowing for easily customizable pipelines to be created with minimal lines of code.
Below is an example of how an analysis might be conducted by combining normalization, protein quantification, differential abundance analysis,
explainable machine learning, and pathway enrichment analysis utilizing  using DPKS.

```python
import xgboost
from dpks.quant_matrix import QuantMatrix

quant_data = "quant_data.tsv"
design_matrix = "design_matrix.tsv"

clf = xgboost.XGBClassifier(
    max_depth=2,
    reg_lambda=2,
    objective="binary:logistic",
    seed=42
)

qm = (
    QuantMatrix(
        quantification_file=quant_data,
        design_matrix_file=design_matrix
    )
    .filter()
    .normalize(
        method="mean",
        use_rt_sliding_window_filter=True
    )
    .quantify(
        method="maxlfq",
        threads=10,
        top_n=5
    )
    .compare(
        method="linregress",
        min_samples_per_group=2,
        comparisons=[(2, 1), (3, 1)]
    )
    .explain(
        clf,
        comparisons=[(2, 1), (3, 1)],
        n_iterations=100,
        downsample_background=True
    )
    .annotate()
)

enr = qm.enrich(
    method="overreptest",
    libraries=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022'],
    filter_shap=True
)

```

Here, we parse in the quantitative data and a design matrix, filter out contaminants and precursors below a 1% FDR threshold,
normalize the data using the mean of a retention time sliding window filter, quantify proteins using relative quantification,
perform differential abundance analysis using linear regression between 2 different groups of comparisons, explain those comparisons
using explainable machine learning with 100 iterations of a downsampled bootstrap interpreter, and then annotate the uniprot IDs
with their corresponding gene names. Finally, we take the output from the above analysis and perform pathway enrichment overrepresentation
statistical tests using 3 different pathway databases on only proteins considered important during classification from the `explain` step.

DPKS makes complicated analysis easy, and allows you to explore multiple analytical avenues in a clean and concise manner.

## Getting started

-   Take a look at the [Usage](usage/installation.md)
    section for instructions on how to get started.

# Contributors

- Aaron Scott aaron.scott@med.lu.se
- Erik Hartman erik.hartman@med.lu.se
- Lars Malmstrom lars.malmstrom@med.lu.se

# Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.

# References

[^1]: Aaron M. Scott, Erik Hartman, Johan Malmström, Lars Malmström. Explainable machine learning for the identification of proteome states via the data processing kitchen sink.
bioRxiv 2023.08.30.555506; doi: https://doi.org/10.1101/2023.08.30.555506
