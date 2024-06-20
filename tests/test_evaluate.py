from dpks.quant_matrix import QuantMatrix
import pytest
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def quantified_data(paths):

    clf = LogisticRegression()

    qm = QuantMatrix(
        quantification_file=str(paths["sepsis_matrix_path"]),
        design_matrix_file=str(paths["sepsis_design_path"]),
    )
    quantified_data = (
        qm.normalize(method="mean")
        .quantify(method="top_n", summarization_method="mean")
        .impute(method="constant", constant=0)
        .append(method="shuffle", feature_column="Protein")
        .compare(method="linregress", min_samples_per_group=10, comparisons=[(2, 1)])
        .explain(clf, comparisons=[(2, 1)], n_iterations=10, downsample_background=True, fillna=True)
    )
    return quantified_data


def test_evaluate(quantified_data: QuantMatrix):

    quantified_data.evaluate(method="basic", comparisons=[(2, 1)])
