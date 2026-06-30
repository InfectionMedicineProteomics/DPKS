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
    name="normalize",
    help="Normalize sample intensities",
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
    #Normalization
    method: Annotated[str, typer.Option(
        "--method",
        help="Base normalisation method: 'mean', 'median', 'tic', or 'log2'.",
    )] = "mean",
    log_transform: Annotated[bool, typer.Option(
        "--log-transform/--no-log-transform",
        help="Skip the log₂ transform that is applied by default after normalisation.",
    )] = True,

    #RT Sliding Window Options
    rt_sliding_window: Annotated[bool, typer.Option(
        "--rt_sliding_window/--no-rt_sliding_window",
        help="Use a retention-time sliding window filter to normalize intensities."
    )] = False,
    rt_minimum_data_points: Annotated[int, typer.Option(
        "--rt_minimum_data_points",
        help="Minimum number of data points in each RT window."
    )] = 100,
    rt_stride: Annotated[int, typer.Option(
        "--rt_stride",
        help="Stride used for sliding window filter"
    )] = 1,
    rt_overlapping: Annotated[bool, typer.Option(
        "--rt_overlapping",
        help="Use overlapping sliding windows"
    )] = True,
    rt_unit: Annotated[str, typer.Option(
        "--rt_unit",
        help="'minute' or 'second'."
    )] = "minute",
) -> None:

    qm = load_quant_matrix(quant_file, design_matrix, quant_type=quant_type, diann_qvalue=diann_qvalue)

    norm_kwargs: dict = dict(
        method=method,
        log_transform=log_transform,
    )

    if rt_sliding_window:
        norm_kwargs.update(
            use_rt_sliding_window_filter=True,
            minimum_data_points=rt_minimum_data_points,
            stride=rt_stride,
            use_overlapping_windows=rt_overlapping,
            rt_unit=rt_unit,
        )

    qm = qm.normalize(**norm_kwargs)

    save_quant_matrix(qm, output)
