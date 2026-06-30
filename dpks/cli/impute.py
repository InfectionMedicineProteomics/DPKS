from pathlib import Path
from typing import Annotated, Optional

import typer

import numpy as np

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix

class QuantifyError(Exception):
    pass

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)


@app.command(
    name="impute",
    help="Impute missing values in the quantitative matrix",
    no_args_is_help=True,
)
def impute(
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

    method: Annotated[str, typer.Option(
        "--method",
        help=(
            "Imputation method:\n"
            "  'uniform_percentile' — draw from [0, percentile of observed values].\n"
            "  'uniform_range'       — draw uniformly from [min-value, max-value]."
        ),
    )] = "uniform_percentile",

    percentile: Annotated[float, typer.Option(
        "--percentile",
        help="Percentile cutoff (uniform_percentile method only). E.g. 0.1 = 10th percentile.",
    )] = 0.1,

    min_value: Annotated[int, typer.Option(
        "--min_value",
        help="Lower bound for uniform random imputation (uniform_range method only).",
    )] = 0,
    max_value: Annotated[int, typer.Option(
        "--max_value",
        help="Upper bound for uniform random imputation (uniform_range method only).",
    )] = 1

) -> None:

    qm = load_quant_matrix(
        quant_file=quant_file,
        design_matrix=design_matrix,
    )

    X = qm.quantitative_data.X
    n_missing = int(np.sum(np.isnan(X)))
    pct = (n_missing / X.size) * 100
    info(f"Missing values before imputation: {n_missing} ({pct:.1f}%)")

    impute_kwargs: dict = dict(method=method)

    if method == "uniform_percentile":
        impute_kwargs["percentile"] = float(percentile)
    elif method == "uniform_range":
        impute_kwargs["minvalue"] = int(min_value)
        impute_kwargs["maxvalue"] = int(max_value)

    qm = qm.impute(**impute_kwargs)

    save_quant_matrix(qm, output)
