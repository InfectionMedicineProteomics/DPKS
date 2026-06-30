from pathlib import Path
from typing import Annotated, Optional

import typer

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix

class BatchCorrectionError(Exception):
    pass

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

@app.command(
    name="correct",
    help="Batch effect correction",
    no_args_is_help=True,
)
def normalize(
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
    quant_type: Annotated[str, typer.Option(
        "--quant_type",
        help="Input format: 'standard' or 'diann'."
    )] = "standard",
    diann_qvalue: Annotated[float, typer.Option(
        "--diann_qvalue",
        help="DIA-NN q-value (diann input only)"
    )] = 0.01,

    method: Annotated[str, typer.Option(
"--method", "-m",
        help="Correction method: 'combat' (empirical Bayes) or 'mean' (reference batch subtraction).",
    )] = "mean",
    reference_batch: Annotated[Optional[str], typer.Option(
"--reference-batch", "-r",
        help="Reference batch label (required when --method mean).",
    )] = None,
) -> None:

    qm = load_quant_matrix(quant_file=quant_file, design_matrix=design_matrix, quant_type=quant_type, diann_qvalue=diann_qvalue)

    if "batch" not in qm.sample_annotations.columns:
        raise BatchCorrectionError(
            "No 'batch' column found in the design matrix."
        )

    batches = list(qm.sample_annotations["batch"].unique()) if "batch" in qm.sample_annotations.columns else []
    info(f"Batches detected: {batches}")

    correct_kwargs: dict = dict(method=method)

    if method == "mean":
        if reference_batch is None:
            raise BatchCorrectionError("--reference-batch is required when --method mean.")
        correct_kwargs["reference_batch"] = reference_batch

    with typer.progressbar(length=1, label=f"Correcting ({method})") as progress:
        qm = qm.correct(**correct_kwargs)
        progress.update(1)

    save_quant_matrix(qm, output)
