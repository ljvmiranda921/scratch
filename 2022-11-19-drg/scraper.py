import json
from typing import List, Dict, Any, Optional
import datetime
from pathlib import Path
from tqdm import tqdm

import typer
import requests
import pandas as pd
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
    # fmt: on
) -> List[str]:
    """Scrape all loadout links from Karl.GG"""
    links = []
    timestamp = datetime.datetime.now()
    for p in tqdm(range(1, page_num + 1)):
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

    if len(links) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = f"loadout_links-page-{page_num}_{timestamp.isoformat()}.txt"
        output_path = output_dir / output_file

        with open(output_path, "w") as f:
            for link in tqdm(links):
                f.write("%s\n" % link)
        msg.good(f"Individual loadout links saved to {output_path}")
        msg.good(
            f"Done scraping loadout links. Use the `get-loadouts` command to download in-depth information."
        )


@app.command("get-loadouts", help="Get individual loadouts from a list of links.")
def get_individual_loadouts(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the TXT file from `get-links`."),
    output_path: Path = typer.Argument(..., help="Output CSV file to save the final dump."),
    start_at: Optional[int] = typer.Option(0, help="Start scraping from the following index.")
    # fmt: on
):
    def _get_loadout(url) -> Dict[str, Any]:
        EMPTY_DATA = "{}"
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.content, "html.parser")

            # The output of the response isn't the actual DOM but rather, a PHP/AJAX call
            # where the parameters are stored in the attrs. That's what we only need
            data = soup.body.find("loadout-preview-page").attrs
            user_loadout = json.loads(data[":loadout-data"])

            salutes = int(soup.find("span", attrs={"class": "salute-count"}).text)

            primary = json.loads(data.get(":primary", EMPTY_DATA))
            secondary = json.loads(data.get(":secondary", EMPTY_DATA))
            equipment = json.loads(data.get(":available-equipment", EMPTY_DATA))
            weapon_mods = user_loadout.get("mods")
            equipment_mods = user_loadout.get("equipment_mods")
            overclocks = json.loads(data.get(":overclocks", EMPTY_DATA))

            if user_loadout.get("creator"):
                username = user_loadout.get("creator").get("name")
            else:
                username = "Anonymous"

            def _get_mods(id: str, key: str, mods: Dict) -> str:
                _mods = sorted(
                    [mod for mod in mods if mod.get(key) == id],
                    key=lambda x: x.get("mod_tier"),
                )
                _mod_text = "".join([m.get("mod_index") for m in _mods])
                return _mod_text

            def _get_overclock(id: str) -> str:
                for overclock in overclocks:
                    if overclock.get("gun_id") == id:
                        return overclock.get("overclock_name")

            def _get_equipment(class_name: Dict, equip_map: Dict):
                for equip in equipment:
                    if equip.get("name") == equip_map.get(class_name):
                        id = equip.get("id")
                        mods = _get_mods(id, "equipment_id", equipment_mods)
                        return {"name": equip.get("name"), "mods": mods}

            # The problem here is that both traversal and support tools are called Support Tools,
            # so we need to disambiguate a bit and hard code a few things
            traversal_map = {
                "Gunner": "Zipline Launcher",
                "Scout": "Grappling Hook",
                "Driller": "Reinforced Power Drills",
                "Engineer": "Platform Gun",
            }
            support_map = {
                "Gunner": "Shield Generator",
                "Scout": "Flare Gun",
                "Driller": "Satchel Charge",
                "Engineer": "LMG Gun Platform",
            }
            traversal = _get_equipment(
                user_loadout.get("character").get("name"), traversal_map
            )
            support = _get_equipment(
                user_loadout.get("character").get("name"), support_map
            )

            # Prepare output
            loadout = {
                # fmt: off
                "name": user_loadout.get("name"),
                "class": user_loadout.get("character").get("name"),
                "patch": user_loadout.get("patch_id"),
                "created_at": user_loadout.get("created_at"),
                "updated_at": user_loadout.get("updated_at", None),
                "description": user_loadout.get("description"),
                "username": username,
                "primary": primary.get("name"),
                "primary_mods": _get_mods(id=primary.get("id"), key="gun_id", mods=weapon_mods),
                "primary_overclock": _get_overclock(id=primary.get("id")),
                "secondary": secondary.get("name"),
                "secondary_mods": _get_mods(id=secondary.get("id"), key="gun_id", mods=weapon_mods),
                "secondary_overclock": _get_overclock(id=secondary.get("id")),
                "throwable": json.loads(data.get(":throwable", "{}")).get("name"),
                "traversal": traversal.get("name"),
                "traversal_mods": traversal.get("mods"),
                "support": support.get("name"),
                "support_mods": support.get("mods"),
                "salutes": salutes,
                # fmt: on
            }

            return loadout

    with input_path.open("r") as file:
        lines = [line.rstrip() for line in file]

    loadouts = []
    if start_at:
        msg.info(f"Starting from index {start_at}")
    lines = lines[start_at:]
    try:
        for idx, line in tqdm(enumerate(lines)):
            loadouts.append(_get_loadout(line))
    except Exception:
        msg.text(f"Encountered error, attempting to save until index {idx}")
        df = pd.DataFrame(loadouts)
        df.to_csv(output_path, index=False)
        msg.fail(f"Failure at URL {line}. Stopped at index {idx}.")
    else:
        df = pd.DataFrame(loadouts)
        df.to_csv(output_path, index=False)
        msg.good(f"Saved to {output_path}")


if __name__ == "__main__":
    app()
