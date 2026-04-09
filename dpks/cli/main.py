import typer

from typing import Annotated
from pathlib import Path
from typing import Optional

from dpks.cli import (
    filter,
    normalize,
    correct,
    quantify,
    impute,
    compare,
    explain,
    gui
)

app = typer.Typer(
    name="dpks",
    help=(
        "DPKS - Data Processing Kitchen Sink.\n\n"
        "A command-line interface for proteomics data processing, "
        "statistical analysis, and explainable machine learning.\n\n"
        "Run any sub-command with --help for detailed usage, e.g.:\n\n"
        "    dpks filter --help"
    ),
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False
)


app.add_typer(filter.app)
app.add_typer(normalize.app)
app.add_typer(correct.app)
app.add_typer(quantify.app)
app.add_typer(impute.app)
app.add_typer(compare.app)
app.add_typer(explain.app)
app.add_typer(gui.app)

if __name__ == "__main__":
    app()
