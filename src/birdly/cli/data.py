import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()


def _run_stage(stage: str) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    code = subprocess.call(
        [sys.executable, "-m", "dvc", "repro", stage],
        cwd=repository_root,
    )
    raise typer.Exit(code=code)


@app.command()
def fetch() -> None:
    """Download images for the dataset manifest."""
    _run_stage("download_images")


@app.command()
def materialize() -> None:
    """Build and validate the dataset manifest."""
    _run_stage("validate_dataset")
