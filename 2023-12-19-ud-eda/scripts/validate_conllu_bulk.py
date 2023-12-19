import subprocess
from pathlib import Path

import typer
from wasabi import msg


def validate_conllu_bulk(
    # fmt: off
    input_dir: Path = typer.Argument(..., help="Directory to search for CoNLLu files."),
    output_dir: Path = typer.Argument(..., help="Directory to save the logs."),
    validator_script: Path = typer.Argument(..., help="Path to the Universal Dependencies validator script."),
    # fmt: on
):
    pass


if __name__ == "__main__":
    typer.run(validate_conllu_bulk)
