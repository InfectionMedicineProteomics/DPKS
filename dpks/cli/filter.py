from pathlib import Path
from typing import Annotated, Optional

import typer

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix

from math import ceil

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

@app.command(
    name="filter",
    help="Filter protein/peptide/precursor rows",
    no_args_is_help=True
)
def filter(
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
    #Q-value thresholds
    peptide_q_value: Annotated[float, typer.Option(
        "--peptide_q_value", "-pq",
        help="Peptide level q-value threshold"
    )] = 0.01,
    protein_q_value: Annotated[float, typer.Option(
        "--protein_q_value", "-prq",
        help="Protein level q-value threshold"
    )] = 0.01,
    #Boolean flags
    remove_decoys: Annotated[bool, typer.Option(
        "--remove-decoys/--keep-decoys", "-rd",
        help="Remove decoys"
    )] = True,
    remove_contaminants: Annotated[bool, typer.Option(
        "--remove-contaminants/--keep-contaminants", "-rc",
        help="Remove contaminants"
    )] = True,
    remove_non_proteotypic: Annotated[bool, typer.Option(
        "--remove-non-proteotypic/--keep-non-proteotypic", "-rp",
        help="Remove non_proteotypic"
    )] = True,
    remove_zero_rows: Annotated[bool, typer.Option(
        "--remove-zero-rows/--keep-zero-rows", "-rz",
        help="Remove rows with all zero values"
    )] = True,
    # Sparse-row filter
    remove_sparse_rows: Annotated[bool, typer.Option(
        "--remove-sparse-rows", "-sr",
        help="Remove rows with more than --max-zeros zero values"
    )] = False,
    max_zeros: Annotated[Optional[float], typer.Option(
        "--max-zeros",
        help="Percentage of maximum zero values allowed per row (requires --remove-sparse-rows)"
    )] = None,
    # Input format
    quant_type: Annotated[str, typer.Option(
        "--quant_type",
        help="Input format: 'standard' or 'diann'."
    )] = "standard",
    diann_qvalue: Annotated[float, typer.Option(
        "--diann_qvalue",
        help="DIA-NN q-value (diann input only)"
    )] = 0.01
) -> None:

    qm = load_quant_matrix(quant_file, design_matrix, quant_type=quant_type, diann_qvalue=diann_qvalue)

    rows_before = qm.num_rows

    if remove_sparse_rows and max_zeros is None:
        typer.echo(
            typer.style(
                "⚠  --remove-sparse-rows requires --max-zeros. Using default of 0.",
                fg=typer.colors.YELLOW,
            ),
            err=True,
        )
        max_zeros = 0

    filter_kwargs: dict = dict(
        peptide_q_value=peptide_q_value,
        protein_q_value=protein_q_value,
        remove_decoys=remove_decoys,
        remove_contaminants=remove_contaminants,
        remove_non_proteotypic=remove_non_proteotypic,
        remove_zero_rows=remove_zero_rows,
        remove_n_zero_rows=remove_sparse_rows,
    )
    if remove_sparse_rows and max_zeros is not None:
        filter_kwargs["max_n_zeros"] = ceil(rows_before * max_zeros)

    with typer.progressbar(length=1, label="Filtering") as progress:
        qm = qm.filter(**filter_kwargs)
        progress.update(1)

    removed = rows_before - qm.num_rows

    info(
        f"Removed {removed} rows ({removed / max(rows_before, 1) * 100:.1f}%). {qm.num_rows} rows retained. "
        f"({len(qm.proteins)} proteins)."
     )

    save_quant_matrix(qm, output)
