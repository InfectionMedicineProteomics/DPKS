from pathlib import Path
from typing import Annotated, Optional

import typer

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix

class QuantifyError(Exception):
    pass

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

@app.command(
    name="quantify",
    help="Quantify proteins/peptides from precursor intensities",
    no_args_is_help=True,
)
def quantify(
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
        help="Quantification method: 'maxlfq' or 'top_n"
    )] = "maxlfq",
    level: Annotated[str, typer.Option(
        "--level",
        help="Aggregation level: 'protein' or 'peptide'",
    )] = "protein",

    #Maxlfq options
    threads: Annotated[int, typer.Option(
        "--threads",
        help="Number of parallel threads for quantification"
    )] = 1,
    minimum_subgroups: Annotated[int, typer.Option(
        "--minimum_subgroups",
        help="Minimum number of samples per quant group for relative quantification"
    )] = 1,

    #Share options
    top_n: Annotated[int, typer.Option(
        "--top_n",
        help=(
            "Number of top-intensity precursors per protein.\n"
            "For relative quantification, 0 = use all."
        )
    )] = 5,

    #TopN options
    summarization_method: Annotated[str, typer.Option(
        "--summarization_method",
        help="How to combine top-N precursors: 'sum', 'mean', or 'median' (top_n method only).",
    )] = "mean",

) -> None:

    qm = load_quant_matrix(
        quant_file=quant_file,
        design_matrix=design_matrix,
    )

    quant_kwargs: dict = dict(method=method, level=level)

    if method == "maxlfq":
        quant_kwargs.update(
            threads=threads,
            minimum_subgroups=minimum_subgroups,
            top_n=top_n,
        )
    elif method == "top_n":
        quant_kwargs.update(
            top_n=top_n,
            summarization_method=summarization_method,
        )

    info(f"Running {method} quantification at {level} level…")

    qm = qm.quantify(**quant_kwargs)

    info(f"Result: {qm.num_rows} proteins × {qm.num_samples} samples.")
    save_quant_matrix(qm, output)
