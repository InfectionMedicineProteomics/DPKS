# Quantification

Generally in bottom-up LC-MS/MS proteomics, you quantify precursors, which are broken up pieces of a protein (peptide)
that contain some charge state. To make biological sense of quantified signal, it is useful to combine these precursors
into their parent protein for downstream analysis. This process of protein quantification can be tricky, as there is no
set standard that should always be used.

!!! tip
    Our research tends to focus on using DIA-MS, and we restrict the spectral libraries used to analyze the data to
    proteotypic peptides, meaning that each precursor in the library is only linked to 1 protein. This makes protein
    quantification easier as no assumptions need to be made about precursors shared between proteins.

DPKS provides 2 main protein quantification methods:

1. __Absolute Quantification__: Using the `top_n` method.
2. __Relative Quantification__: Using the `maxlfq` method.

## Absolute Quantification

Absolute quantification is performed using the `top_n` method by combining a specified number of the most abundant
precursors for each protein using a summarization method (`sum`, `mean`, or `median`).

!!! note
    This is particularly useful if you want to compare proteins to other proteins in an experiment, like with a protein
    rank plot, to see what proteins are most abundant in your samples.

Absolute quantification can be performed as follows:

```python
qm = qm.quantify(
    method="top_n",
    top_n=3
)
```

The `top_n` parameter indicates how many of the precursors you want to use per protein for quantification.

## Relative Quantification

DPKS uses the iq[^1] implementation of the MaxLFQ algorithm[^2] to extract optimal ratios between samples for each
protein and combines them into a resulting protein quantity.

!!! note
    Since this relative quantification approach uses all precursors for a protein, this is not suitable for protein
    rank plots, as certain proteins will have their absolute abundance underestimated. It is, however, considered
    state-of-the-art when measuring the differences in protein abundance between 2 experimental groups.

Relative quantification can be performed as follows:

```python
qm = qm.quantify(
    method="maxlfq",
    level="protein",
    threads=5
)
```

Relative quantification takes much longer to process than absolute quantification, so we have optimized for performance
using Numba for JIT compilation of the relative quantification algorithms. To enable multithreading, indicate the
desired number of threads using the `threads` parameter.

The `level` parameter indicates what level you want to quantify, whether it is proteins, peptides, or precursors. It is
possible to quantify peptides using precursors, and precursors using fragment quantities (for DIA experiments) if the
correct columns are supplied to the `QuantMatrix`.

## Combined Quantification

To get the best of both worlds, a combined quantification approach is possible that applies relative quantification to
the indicated `top_n` percursors for each protein. This allows for the overall abundance rank of the protein to remain
intact while reaping the benefits of signal smoothing and ratio extraction used for relative quantification.

Combined quantification can be performed simply by providing a value `>0` to the `top_n` parameter of the `quantify()`
method:

```python
qm = qm.quantify(
    method="maxlfq",
    level="protein",
    threads=5,
    top_n=3
)
```


[^1]: Thang V Pham, Alex A Henneman, Connie R Jimenez, iq: an R package to estimate relative protein abundances from ion quantification in DIA-MS-based proteomics, Bioinformatics, Volume 36, Issue 8, April 2020, Pages 2611–2613, [https://doi.org/10.1093/bioinformatics/btz961](https://doi.org/10.1093/bioinformatics/btz961)
[^2]: Cox, Jürgen et al. Accurate Proteome-wide Label-free Quantification by Delayed Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ. Molecular & Cellular Proteomics, Volume 13, Issue 9, 2513 - 2526, [https://doi.org/10.1074/mcp.M113.031591](https://doi.org/10.1074/mcp.M113.031591)


