# Loading Data

DPKS can load data from a variety of different proteomic processing pipelines directly. If you have a filetype that you would like to be able to parse directly into DPKS, please let us know.

The `QuantMatrix` is the main entry point to all analysis in DPKS. A new `QuantMatrix` object can be instantiated with your input data and a design matrix by passing the file paths:

```python
quant_matrix = QuantMatrix(
    quantification_file="path_to_quant_file.tsv",
    design_matrix_file="path_to_design_matrix_file.tsv"
)
```

Or by passing in a `pandas` `DataFrame`:

```python

quant_data = pd.read_csv(
    "path_to_quant_file.tsv",
    sep="\t"
)

design_matrix = pd.read_csv(
    "path_to_design_matrix_file.tsv",
    sep="\t"
)

quant_matrix = QuantMatrix(
    quantification_file=quant_data,
    design_matrix_file=design_matrix
)
```

This is particularly useful if you want to process your data (reformat, filter, etc.) in someway before loading it into DPKS. The ability to pass in files or `DataFrames` directly to the `QuantMatrix` object provides some flexibility in the type of data that you can load, making it easy to write custom parsers for new result file types.

!!! tip
    If you encounter errors during parsing, it is useful to first load your data as `DataFrame`s to first verify that
    everything is formatted correctly

## Generic Input

### Quantitative Data

DPKS accepts a generic results file that you can reformat your own data to if there is not a built-in parser available.

| Column                        | Description                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------|
| PrecursorId                   | A unique identifier generally composed of the Peptide Sequence (with mods) and the charge. |
| Charge                        | The precursor charge.                                                                      |
| PeptideSequence               | The modified peptide sequence.                                                             |
| Decoy (Optional)              | Indicating if the precursor is a decoy (used for filtering).                               |
| RetentionTime                 | The retention time of the precursor.                                                       |
| Protein                       | The protein accession code associated with the precursor.                                  |
| PeptideQValue (Optional)      | The global peptide level q-value (used for filtering).                                     |
| ProteinQValue (Optional)      | The global protein level q-value (used for filtering).                                     |
| Sample Columns (Many Columns) | All other columns containing quantification data for your samples.                         |

If you already have controlled for the global FDR, you do not need to include the Decoy, PeptideQValue, or ProteinQValue columns.

A generic file format may look like this:

| PeptideSequence  | Charge | Decoy | Protein | RetentionTime | PeptideQValue | ProteinQValue | SAMPLE_1.osw | SAMPLE_2.osw | SAMPLE_3.osw |
|------------------|--------|-------|---------|---------------|---------------|---------------|--------------|--------------|--------------|
| PEPTIK           | 4      | 0     | P00352  | 5736.15       | 7.81e-06      | 0.0001169     | 29566.2      | 59295.7      | 24536.4      |
| EFMEEVIQR        | 2      | 0     | P04275  | 3155.5        | 9.41e-06      | 0.0001169     | 69900.3      | 195571.0     | 403947.0     |
| SSSGTPDLPVLLTDLK | 2      | 0     | P00352  | 5386.69       | 7.815e-06     | 0.000116      | 115684.0     | 132524.0     | 217962.0     |


!!! note
    If you want to pass already quantified Proteins you could do this:

    | Protein | SAMPLE_1.osw | SAMPLE_2.osw | SAMPLE_3.osw |
    |---------|--------------|--------------|--------------|
    | P00352  | 29566.2      | 59295.7      | 24536.4      |
    | P04275  | 69900.3      | 195571.0     | 403947.0     |
    | P00352  | 115684.0     | 132524.0     | 217962.0     |

### Design Matrix

A basic design matrix will have 2 main columns:

| Column            | Description                                                                                                      |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| Sample (Required) | A list of the samples. This helps differentiate between sample columns and annotation columns in the QuantMatrix |
| Group  (Optional) | The group the sample belongs to. Used in differential testing and explainable machine learning.                  |

A minimal design matrix for the above input examples could look like this:

| Sample       |
|--------------|
| SAMPLE_1.osw |
| SAMPLE_2.osw |
| SAMPLE_3.osw |

And an example using the `Group` column:

| sample        | group |
|---------------|-------|
| AAS_P2009_167 | 6     |
| AAS_P2009_169 | 4     |
| AAS_P2009_176 | 6     |
| AAS_P2009_178 | 4     |
| AAS_P2009_187 | 4     |
| AAS_P2009_194 | 6     |
| AAS_P2009_196 | 4     |
| AAS_P2009_203 | 6     |
| AAS_P2009_205 | 4     |
| AAS_P2009_212 | 6     |
| AAS_P2009_214 | 4     |
| AAS_P2009_221 | 6     |
| AAS_P2009_230 | 6     |
| AAS_P2009_232 | 4     |
| AAS_P2009_239 | 6     |
| AAS_P2009_241 | 4     |
| AAS_P2009_248 | 6     |
| AAS_P2009_250 | 4     |


## DIANN

You can load data directly from DIA-NN using the long-format `diann-output.tsv` file that is generated. The samples in your design matrix column should match the `Run` column in the DIA-NN output, but other columns can be indicated if desired.

Additionally, if you have used MBR, the correct columns will be used to filter precursors at the indicated FDR threshold.

```python
quant_matrix = QuantMatrix(
    quantification_file=quant_file,
    design_matrix_file=simple_design,
    quant_type="diann",
    diann_qvalue=0.01
)
```

