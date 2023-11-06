from pathlib import Path
from typing import Optional

import typer
from wasabi import msg


def _callback_list_opts(arg):
    return arg.split(",") if arg else None


def benchmark(
    # fmt: off
    output_dir: Path = typer.Argument(..., help="Save the computed metrics (using the filename, [dataset]_[model]_[question].json) in the provided directory.", dir_okay=True),
    seed: int = typer.Option(42, help="Set the random seed.", show_default=True),
    on_models: Optional[str] = typer.Option(None, "--on-models", help="Run the experiments only on the provided models.", callback=_callback_list_opts),
    on_datasets: Optional[str] = typer.Option(None, "--on-datasets", help="Run the experiments only on the provided datasets.", callback=_callback_list_opts),
    ignore_models: Optional[str] = typer.Option(None, "--ignore-models", help="Ignore the following models.", callback=_callback_list_opts),
    ignore_datasets: Optional[str] = typer.Option(None, "--ignore-datasets", help="Ignore the following datasets.", callback=_callback_list_opts),
    # fmt: on
):
    """Run a benchmark experiment

    By default, it will run a benchmark for each model, dataset, and follow-up
    question. This can balloon quickly (i models x j datasets x k follow-up
    questions), so you can pass values on either the --on-models / --on-datasets
    or --ignore-models / --ignore-datasets to reduce the experimentation matrix.

    Note that the --on-xx commands take the highest priority. If you pass a value
    on an --on-xx command, then the --ignore-xx commands are overriden. This is useful
    if you just want to run on a specific instance of the experiment.
    """
    pass


if __name__ == "__main__":
    typer.run(benchmark)
