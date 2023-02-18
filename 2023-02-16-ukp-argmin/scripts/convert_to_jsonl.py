from pathlib import Path

import srsly
import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def convert_to_jsonl(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the TSV file of UKP annotations."),
    output_path: Path = Arg(...,  help="Path to the output JSONL file for Prodigy."),
    dataset: str = Opt("test", help="Dataset type to convert to JSONL."),
    # fmt: on
):
    """Convert the raw annotations into JSONL to use for Prodigy"""
    tsv_file = input_path.open("r")
    next(tsv_file)  # skip first line (headers)
    records = []
    for line in tsv_file:
        text, _, dataset_ = line.strip().split("\t")[4:]
        if dataset_ == dataset:
            records.append({"text": text})

    msg.info(f"Saved {len(records)} records ({dataset}) to {output_path} ")
    srsly.write_jsonl(output_path, records)


if __name__ == "__main__":
    typer.run(convert_to_jsonl)
