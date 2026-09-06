import typer

from birdly.cli.data import app as data_app
from birdly.cli.initialize import init_birdly

app = typer.Typer()
app.add_typer(data_app, name="data")


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
