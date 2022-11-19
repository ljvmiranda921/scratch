from typing import List
import datetime
from pathlib import Path

import typer
import requests
from bs4 import BeautifulSoup
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

APP_HELP = """
Unofficial karl.gg scraper

This scraper allows you to download individual loadouts from karl.gg for
downstream analyses. First, you need to obtain a list of loadouts using the
`get-links` command. This will produce a TXT file. Then, you must supply this
TXT file to the `get-loadouts` command to obtain more in-depth loadout information.
"""

app = typer.Typer(help=APP_HELP)


@app.command("get-links", help="Scrape loadout links from karl.gg")
def get_loadout_links(
    # fmt: off
    output_dir: Path = typer.Argument(..., help="Output directory to save the scraped files."), 
    page_num: int = typer.Option(950, "--page-num", "-n", show_default=True, help="Max number of page to scrape."), 
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Set verbosity.")
    # fmt: on
) -> List[str]:
    """Scrape all loadout links from Karl.GG"""
    links = []
    timestamp = datetime.datetime.now()
    for p in range(1, page_num + 1):
        msg.text(f"Checking page {p}...", show=verbose)
        url = f"https://karl.gg/browse?sort=updated_at&direction=desc&page={p}"
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")
            for row in table.findAll("tr"):
                # The individual loadout links have the cursor-pointer class.
                link = row.find_all("a", attrs={"class": "cursor-pointer"})
                if link:
                    links.append(link[0].get("href"))
    msg.info(f"Found {len(links)} builds (until page {page_num}) as of {timestamp}")

    if output_dir and len(links) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"loadout_links-page-{page_num}_{timestamp.isoformat()}.txt"
        )
        with open(output_path, "w") as f:
            for link in links:
                f.write("%s\n" % link)
        msg.good(f"Individual loadout links saved to {output_path}")

    return links


@app.command("get-loadouts", help="Get individual loadouts from a list of links.")
def get_individual_loadouts():
    pass


if __name__ == "__main__":
    app()
