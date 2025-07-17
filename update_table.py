# /// script
# dependencies = [ "pandas", "typer", "srsly", "tabulate" ]
# ///

from pathlib import Path
from srsly import read_yaml

import typer
import pandas as pd

DEFAULT_README_PATH = Path("./README.md")
README_TEMPLATE = """# 📓 Scratch

This repository contains Jupyter notebooks and random assortment of projects.
Think of this as a scratch paper for my ideas. Some of these may have found
their way into my [blog](https://ljvmiranda921.github.io). To generate the table
below, run `uv run update_table.py`.

{table}
"""


def update_table(readme_path: Path = DEFAULT_README_PATH):
    """Update the contents table in the README"""

    root = Path(__file__).parent
    meta_files = sorted(root.glob("*/meta.yml"))

    metadata = []

    for fp in meta_files:
        url = f"https://github.com/ljvmiranda921/scratch/tree/master/{str(fp.parent)}"
        meta = read_yaml(fp)
        metadata.append(
            {
                "Name": f"[{meta.get('name')}]({url})",
                "Description": meta.get("description"),
            }
        )

    table = pd.DataFrame(metadata).to_markdown(index=False)

    readme = README_TEMPLATE.format(table=table)
    with readme_path.open("w") as readme_file:
        readme_file.write(readme)


if __name__ == "__main__":
    typer.run(update_table)
