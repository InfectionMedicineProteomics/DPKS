from pathlib import Path
from typing import Annotated, Optional, List

import typer

import numpy as np

from dpks.cli.utils import load_quant_matrix, info, save_quant_matrix, parse_comparisons

class QuantifyError(Exception):
    pass

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

@app.command(
    name="compare",
    help="Differential abundance analysis",
    no_args_is_help=True,
)
def compare(
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

    method: Annotated[str, typer.Option(
        "--method",
        help="Statistical test: 'linregress', 'ttest', 'ttest_paired', or 'anova'.",
    )] = "linregress",

    min_samples: Annotated[int, typer.Option(
        "--min_samples",
        help="Minimum number of samples per group required to test a protein.",
    )] = 3,

    level: Annotated[str, typer.Option(
        "--level",
        help="Analysis level: 'protein' or 'peptide'"
    )] = "protein",

    correction: Annotated[str, typer.Option(
        "--correction",
        help=(
            "Multiple-testing correction method: "
            "'fdr_tsbh' (recommended), 'fdr_bh', 'bonferroni', etc.\n"
            "(any available method from 'statsmodels')"
        ),
    )] = "fdr_tsbh",

) -> None:

    comparison_tuples = parse_comparisons(comparisons)

    info(f"Comparisons: {comparison_tuples}")

    qm = load_quant_matrix(
        quant_file=quant_file,
        design_matrix=design_matrix,
    )

    qm = qm.compare(
        method=method,
        comparisons=comparison_tuples,
        min_samples_per_group=min_samples,
        level=level,
        multiple_testing_correction_method=correction,
    )

    for g1, g2 in comparison_tuples:
        col = f"CorrectedPValue{g1}-{g2}"
        significant_proteins = (qm.row_annotations[col] < 0.05).sum()
        info(f"Group {g1} vs {g2}: {significant_proteins} proteins below a corrected pvalue < 0.05")

    save_quant_matrix(qm, output)
