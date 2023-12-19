import sys
import subprocess
from pathlib import Path

import typer
from wasabi import msg
from tqdm import tqdm


def validate_conllu_bulk(
    # fmt: off
    input_dir: Path = typer.Argument(..., help="Directory to search for CoNLLu files."),
    output_dir: Path = typer.Argument(..., help="Directory to save the logs."),
    validator_script: Path = typer.Argument(..., help="Path to the Universal Dependencies validator script."),
    verbose: bool = typer.Option(False, help="Display more information."),
    # fmt: on
):
    for dir in sorted(input_dir.iterdir()):
        conllu_files = sorted(list(dir.glob("*.conllu")))
        msg.info(f"Found {len(conllu_files)} files from {dir}")

        for file in tqdm(conllu_files):
            result = subprocess.run(
                f"cat {file} | {sys.executable} {validator_script} --lang tl --max-err=0",
                capture_output=True,
                shell=True,
            )
            output = result.stderr.decode("utf-8")
            msg.text(output, show=verbose)


if __name__ == "__main__":
    typer.run(validate_conllu_bulk)
