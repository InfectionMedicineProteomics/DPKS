from pathlib import Path
from typing import Annotated, Optional, List

import typer

import numpy as np

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix, parse_comparisons

class ExplainError(Exception):
    pass

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

# Supported classifiers and their import paths
_CLASSIFIERS = {
    "logistic_regression":("sklearn.linear_model","LogisticRegression"),
}

# TODO: left it open to possibly add more models, but this needs to be optimized somehow and should
# probably be done in a notebook outside of the command line. As it stands, only basic logistic regression is
# implemented here
# For example, how would we optimize hyperparameters for more complex models? We could implement an optimize step
# that builds an optimized model from a dataset, and then pass those parameters and the model to this step.
# TODO: Similar to this, should we add a CLI command for enrich()
def _build_classifier(name: str):
    entry = _CLASSIFIERS.get(name)
    if entry is None:
        raise ExplainError(
            f"Unknown classifier '{name}'. "
            f"Choose from: {', '.join(_CLASSIFIERS)}"
        )
    module_name, class_name = entry
    try:
        import importlib
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except ImportError:
        ExplainError(
            f"Could not import '{module_name}'. "
            f"Install it with: pip install {module_name.split('.')[0]}"
        )

    if name == "logistic_regression":
        return cls(
            max_iter=1000,
            random_state=42,
            n_jobs=1
        )

    return cls()


@app.command(
    name="explain",
    help="Explainable machine learning analysis",
    no_args_is_help=True,
)
def explain(
    quant_file: Annotated[Path, typer.Option(
        "--quant_file",
        help="Path to the quantification file"
    )],
    design_matrix: Annotated[Path, typer.Option(
        "--design_matrix",
        help="Path to the design matrix"
    )],
    output: Annotated[Path, typer.Option(
        "--output",
        help="Path to the output file"
    )],

    comparisons: Annotated[List[str], typer.Option(
        "--comparisons",
        help= (
            "Group comparison in 'A-B' format (group A vs group B). "
            "Use the integer group IDs from the design matrix. "
            "Repeat the flag for multiple comparisons, e.g. -c 2-1 -c 3-1"
        ),
    )],

    classifier: Annotated[str, typer.Option(
        "--classifier",
        help=f"Classifier to use. Options: {', '.join(_CLASSIFIERS)}.",
    )],

    n_iterations: Annotated[int, typer.Option(
        "--iterations",
        help="Number of bootstrap iterations for feature importance estimation."
    )] = 100,
    downsample: Annotated[bool, typer.Option(
        "--downsample/--no-downsample",
        help="Disable background downsampling."
    )] = True,

    feature_column: Annotated[str, typer.Option(
        "--feature_column",
        help="Column used to identify features (e.g. 'Protein' or 'Gene').",
    )] = "Protein"
) -> None:

    comparison_tuples = parse_comparisons(comparisons)
    info(f"Comparisons: {comparison_tuples}")
    info(f"Classifier:  {classifier}")
    info(f"Iterations:  {n_iterations}")
    info(f"Downsample: {downsample}")

    qm =load_quant_matrix(
        quant_file=quant_file,
        design_matrix=design_matrix,
    )
    clf = _build_classifier(classifier)

    qm = qm.explain(
        clf=clf,
        comparisons=comparison_tuples,
        n_iterations=n_iterations,
        downsample_background=downsample,
        feature_column=feature_column,
    )

    for g1, g2 in comparison_tuples:
        importance_col = f"MeanImportance{g1}-{g2}"
        protein_col = "Gene" if "Gene" in qm.row_annotations.columns else "Protein"
        if importance_col in qm.row_annotations.columns:
            top5 = (
                qm.row_annotations[[protein_col, importance_col]]
                .dropna()
                .drop_duplicates(protein_col)
                .nlargest(5, importance_col)[protein_col]
                .tolist()
            )
            info(f"Top proteins (Group {g1} vs {g2}): {', '.join(str(p) for p in top5)}")

    save_quant_matrix(qm, output)
