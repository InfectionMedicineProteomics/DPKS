# Normalization

Normalization a standard method used in omic analysis to minimize batch effect and technical variation between samples
prior to further downstream analysis. This is particularly important to ensure that the signal you are seeing in your
analysis is actual biological signal and not just noise. DPKS exposes 3 main methods for normalizing your quantitative
matrices:

* `tic` - Total ion chromatogram normalization: The sum of all signal in each sample is used to scale each measured intensity, and the median of the sums is used to rescale the data.
* `median` - Median normalization: The median of each sample is used to scale each intensity and the mean of the medians is used to rescale the data.
* `mean` - Mean normalization: The mean of each sample is used to scale each intensity and the mean of the means is used to rescale the data.

!!! tip
    An implementation of retention time sliding window normalization [^1] can be applied if your input data was LC-MS/MS data and a RetentionTime column is available. This is recommended.

The data is then log2 transformed to allow for interpretable downstream analysis.

!!! note
    You should always (generally) log2 transform your data. Especially if you are doing differential expression analysis.
    This is a default in DPKS, so you do not need to worry about this.

Normalization can be performed as follows:

```python
from dpks.quant_matrix import QuantMatrix

qm = QuantMatrix(
    quantification_file="../tests/input_files/de_matrix.tsv",
    design_matrix_file="../tests/input_files/de_design_matrix.tsv"
).filter()

qm = qm.normalize(
    method="mean",
    log_transform=True,
    use_rt_sliding_window_filter=True,
    minimum_data_points=100,
    stride=5,
    use_overlapping_windows=True,
    rt_unit="seconds"
)
```

Many of the set paramters above are defaults and do not need to be passed everytime.


[^1]: Jakob Willforss, Aakash Chawade, and Fredrik Levander. NormalyzerDE: Online Tool for Improved Normalization of Omics Expression Data and High-Sensitivity Differential Expression Analysis.
Journal of Proteome Research 2019 18 (2), 732-740
<a href="https://doi.org/10.1021/acs.jproteome.8b00523" target="_blank">DOI: 10.1021/acs.jproteome.8b00523</a>

