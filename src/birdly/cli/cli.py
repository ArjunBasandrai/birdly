import subprocess
import sys
from pathlib import Path

import typer

from birdly.cli.data import app as data_app
from birdly.cli.initialize import init_birdly

app = typer.Typer()
app.add_typer(data_app, name="data")

_SCRIPT_OPTIONS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def _run_script(script: str, arguments: list[str]) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    code = subprocess.call(
        [sys.executable, str(repository_root / "scripts" / script), *arguments]
    )
    raise typer.Exit(code=code)


@app.command("load-test", context_settings=_SCRIPT_OPTIONS)
def load_test(ctx: typer.Context) -> None:
    _run_script("run_load_test.py", ctx.args)


@app.command("reliability-test", context_settings=_SCRIPT_OPTIONS)
def reliability_test(ctx: typer.Context) -> None:
    _run_script("run_reliability_tests.py", ctx.args)


@app.command()
def init(
    data_root: str = typer.Option(
        ...,
        "--data-root",
        help="Path to the data root directory.",
    ),
    image_data_root: str = typer.Option(
        ...,
        "--image-data-root",
        help="Path to the downloaded image directory.",
    ),
):
    init_birdly(
        data_root=data_root,
        image_data_root=image_data_root,
    )


if __name__ == "__main__":
    app()
