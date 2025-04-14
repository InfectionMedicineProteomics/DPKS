from dpks.quant_matrix import QuantMatrix
import pytest

@pytest.fixture
def quantified_data(paths):

    qm = QuantMatrix(
        quantification_file=str(paths["sepsis_matrix_path"]),
        design_matrix_file=str(paths["sepsis_design_path"]),
    )
    quantified_data = (
        qm.filter()
        .normalize(method="mean", use_rt_sliding_window_filter=True, rt_unit="second", stride=5, minimum_data_points=200)
        .quantify(method="top_n", top_n=5, summarization_method="mean").impute(method="neighborhood")
        .annotate()
        .compare(method="linregress", min_samples_per_group=10, comparisons=[(2, 1)])
    )
    return quantified_data


def test_evaluate(quantified_data: QuantMatrix):

    clustered_data = quantified_data.cluster()

    assert 130 < clustered_data.row_annotations['FeatureCluster'].unique().shape[0] < 150


