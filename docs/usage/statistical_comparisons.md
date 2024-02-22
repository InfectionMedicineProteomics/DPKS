# Statistical Comparisons

DPKS allows for a number of different statistical tests to be performed between experimental groups in your data.
Currently, it is possible to compare your samples using:

1. `T-test` - 2-sided t-test for the independent samples.
2. `Linear Regression` - 2-sided test to calculate a linear least-squares regression for the abundances betweeen experimental groups.
3. `ANOVA` - One-way ANOVA to compare the means of 2 groups.
4. `Paired T-test` - 2-sided t-test for 2 related samples (need to indicate a Pairs column in the Design Matrix).

!!! note
    The `scipy` implementations of the above methods are currently used, but it is easy to add new comparison methods if
    a desired one is not yet available.

## T-test

```python
qm = qm.compare(
    method="ttest",
    comparisons=(2,1),
    min_samples_per_group=10,
    level="protein",
    multiple_testing_correction_method="fdr_tsbh"
)
```

## Linear Regression

```python
qm = qm.compare(
    method="linregress",
    comparisons=(2,1),
    min_samples_per_group=10,
    level="protein",
    multiple_testing_correction_method="fdr_tsbh"
)
```

## ANOVA

```python
qm = qm.compare(
    method="anova",
    comparisons=(2,1),
    min_samples_per_group=10,
    level="protein",
    multiple_testing_correction_method="fdr_tsbh"
)
```

## Paired T-test

In order to perform paired t-tests with your data, you first need to pass in a "Pair" column with your design matrix:

| Sample | Group | Pair |
|--------|-------|------|
| s1     | 1     | s2   |
| s2     | 2     | s1   |
| s3     | 2     | s4   |
| s4     | 1     | s3   |

Here, you need to make sure that your Sample is paired with another valid Sample in the list, but each sample pairing
should be unique.

```python
qm = qm.compare(
    method="ttest_paired",
    comparisons=(2,1),
    min_samples_per_group=10,
    level="protein",
    multiple_testing_correction_method="fdr_tsbh"
)
```

## Multiple Comparisons

It is possible to perform multiple comparisons if you have multiple groups in your data by passing a list of tuples in
as the `comparison` parameter:

```python
qm = qm.compare(
    method="ttest_paired",
    comparisons=[(2,1), (3, 1), (4, 1)],
    min_samples_per_group=10,
    level="protein",
    multiple_testing_correction_method="fdr_tsbh"
)
```

The above will separately compare groups 2, 3, and 4 to group 1 and write results columns for each of the 3 different
comparisons.


## Example

There is a jupyter notebook with some examples of how to use this functionality and some possible plots.

[Differential Expression](https://github.com/InfectionMedicineProteomics/DPKS/blob/main/notebooks/differential_expression.ipynb): Demonstrates how to compute differences between two experimental conditions.

