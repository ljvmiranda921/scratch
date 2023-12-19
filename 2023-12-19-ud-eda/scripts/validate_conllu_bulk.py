import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import typer
import polars as pl
from wasabi import msg
from tqdm import tqdm


def validate_conllu_bulk(
    # fmt: off
    input_dir: Path = typer.Argument(..., help="Directory to search for CoNLLu files."),
    output_dir: Path = typer.Argument(..., help="Directory to save the logs as JSONL."),
    validator_script: Path = typer.Argument(..., help="Path to the Universal Dependencies validator script."),
    verbose: bool = typer.Option(False, help="Display more information."),
    # fmt: on
):
    for dir in sorted(input_dir.iterdir()):
        conllu_files = sorted(list(dir.glob("*.conllu")))
        msg.info(f"Found {len(conllu_files)} files from {dir}")

        for file in tqdm(conllu_files):
            # Run validator and parse the results
            result = subprocess.run(
                f"cat {file} | {sys.executable} {validator_script} --lang tl --max-err=0",
                capture_output=True,
                shell=True,
            )
            raw = result.stderr.decode("utf-8")
            msg.text(raw, show=verbose)
            parsed_output = parse_error_log(raw)

            # Save the parsed results
            df = pl.DataFrame(parsed_output)
            output_file = output_dir / file.parent.name / f"{file.stem}.csv"
            df.write_csv(output_file)


def parse_error_log(raw: str) -> List[Dict[str, str]]:
    """Parse the raw error log and transform it into a parseable sequence

    [Line 14 Sent 2]: [L4 Morpho feature-value-upos-not-permitted] Value remt of feature
    |--- location --|  |--------------- error type ---------------| |--- description---|
    """
    pattern = r"\[([^\]]+)\]: \[([^\]]+)\] (.+)"
    lines = raw.splitlines()

    parsed_outputs = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            parsed_outputs.append(
                {
                    "location": match.group(1),
                    "error_code": match.group(2),
                    "error_description": match.group(3),
                }
            )
    return parsed_outputs


if __name__ == "__main__":
    typer.run(validate_conllu_bulk)
