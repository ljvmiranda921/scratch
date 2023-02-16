from pathlib import Path
from srsly import read_yaml

import typer
import pandas as pd

README_DEFAULT_PATH = Path("./README.md")


def update_table(readme_path: Path = README_DEFAULT_PATH):
    """Update the contents table in the README"""

    root = Path(__file__).parent

    metadata = []
    for fp in root.glob("*/meta.yml"):
        url = f"https://github.com/ljvmiranda921/scratch/tree/master/{str(fp.parent)}"
        meta = read_yaml(fp)
        metadata.append(
            {
                "Name": f"[{meta.get('name')}]({url})",
                "Description": meta.get("description"),
            }
        )

    table = pd.DataFrame(metadata)
    table_mk = table.to_markdown(index=False)

    readme = f"""# 🌱 Scratch

This repository contains Jupyter notebooks and random assortment of projects.
Think of this as a scratch paper for my ideas.

## Contents

{table_mk}
    """

    with readme_path.open("w") as f:
        f.write(readme)


if __name__ == "__main__":
    typer.run(update_table)
