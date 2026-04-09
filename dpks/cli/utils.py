import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from dpks import QuantMatrix

class ComparisonError(Exception):
    pass

class IOError(Exception):
    pass

# ── Console helpers ───────────────────────────────────────────────────────────

def success(msg: str) -> None:
    typer.echo(typer.style(f"SUCCESS: {msg}", fg=typer.colors.GREEN))


def info(msg: str) -> None:
    typer.echo(typer.style(f"INFO: {msg}", fg=typer.colors.CYAN))


def warn(msg: str) -> None:
    typer.echo(typer.style(f"WARNING: {msg}", fg=typer.colors.YELLOW), err=True)


# def error(msg: str) -> None:
#     typer.echo(typer.style(f"✗  {msg}", fg=typer.colors.RED, bold=True), err=True)

#
# def abort(msg: str, exit_code: int = 1) -> None:
#     """Print an error message and exit."""
#     #error(msg)
#     raise typer.Exit(exit_code)

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_quant_matrix(quant_file: Path, design_matrix: Path, quant_type: str = "standard", diann_qvalue: float = 0.01,
                      annotation_fasta: Optional[Path] = None):
    """
    Instantiate and return a QuantMatrix from file paths.
    Exits with a helpful message on failure.
    """

    _require_file(quant_file, "Quantification file")
    _require_file(design_matrix, "Design Matrix file")

    kwargs: dict = dict(
        quantification_file=str(quant_file),
        design_matrix_file=str(design_matrix),
        quant_type=quant_type,
    )

    if quant_type == "diann":
        kwargs["diann_qvalue"] = diann_qvalue

    if annotation_fasta is not None:
        _require_file(annotation_fasta, "Annotation FASTA")
        kwargs["annotation_fasta_file"] = str(annotation_fasta)

    try:
        qm = QuantMatrix(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        raise exc

    info(
        f"Loaded {qm.num_rows} rows × {qm.num_samples} samples "
        f"({len(qm.proteins)} proteins)."
    )

    return qm


def save_quant_matrix(qm, output: Path) -> None:
    """Write a QuantMatrix to a TSV file, creating parent dirs as needed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qm.write(str(output))
        success(f"Saved → {output}")
    except Exception as exc:
        raise IOError(f"Failed to write output: {exc}")

#
# def load_tsv(path: Path, label: str = "file") -> pd.DataFrame:
#     _require_file(path, label)
#     try:
#         return pd.read_csv(path, sep="\t")
#     except Exception as exc:
#         abort(f"Could not read {label} '{path}': {exc}")


def parse_comparisons(raw: list[str]) -> list[tuple[str, str]]:
    """
    Parse a list of 'A-B' strings into (A, B) integer tuples.

    Example
    -------
    >>> parse_comparisons(["2-1", "3-1"])
    [(2, 1), (3, 1)]
    """
    comparisons: list[tuple[str, str]] = []
    for comparison in raw:
        groups = comparison.split("-")
        if len(groups) != 2:
            raise ComparisonError(
                f"Invalid comparison '{comparison}'. "
                "Use the format 'A-B', e.g. '2-1'."
            )
        try:
            comparisons.append((int(groups[0]), int(groups[1])))
        except ValueError:
            raise ComparisonError(
                f"Invalid comparison '{comparison}': group IDs must be integers."
            )
    return comparisons


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise IOError(f"{label} not found: {path}")
    if not path.is_file():
        raise IOError(f"{label} is not a file: {path}")
