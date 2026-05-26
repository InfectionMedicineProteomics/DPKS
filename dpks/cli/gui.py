import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional, List

import typer
from dpks.cli.utils import info

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=True,
)

def _find_gui_app() -> Path:
    candidate = Path(__file__).parent.parent / "gui" / "app.py"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate the DPKS GUI app.py. "
    )

@app.command(
    name="gui",
    help="Launch DPKS GUI",
    no_args_is_help=False,
)
def gui(
    port: Annotated[int, typer.Option(
        "--port",
        help="Port to run Streamlit app on",
    )] = 8501,
) -> None:

    app_path = _find_gui_app()

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port)
    ]

    info(f"Starting DPKS GUI on http://localhost:{port} ... (Ctrl+C to quit)")

    subprocess.run(cmd, check=True)
