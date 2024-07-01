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
        qm.normalize(
            method="mean",
        )
        .quantify(method="top_n", summarization_method="mean")
        .impute(method="constant", constant=0)
    )
    return quantified_data


def test_append(quantified_data: QuantMatrix):

    feature_length = quantified_data.row_annotations.shape[0]
    expected_length = feature_length * 2

    quantified_data_decoys = quantified_data.append(method="shuffle", feature_column="Protein")

    assert (quantified_data_decoys.row_annotations.shape[0] == expected_length)
