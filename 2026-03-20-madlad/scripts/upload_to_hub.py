"""Upload data/classified/*.csv to HuggingFace, one subset per language."""

from pathlib import Path

from datasets import Dataset

REPO_ID = "ljvmiranda921/low-res-textcat"
CLASSIFIED_DIR = Path(__file__).parent.parent / "data" / "classified"


def pick_files() -> dict[str, Path]:
    """Return {lang: path}, keeping the largest file when duplicates exist."""
    by_lang: dict[str, Path] = {}
    for csv in sorted(CLASSIFIED_DIR.glob("*_classified.csv")):
        lang = csv.name.split("_", 1)[0]
        if lang not in by_lang or csv.stat().st_size > by_lang[lang].stat().st_size:
            by_lang[lang] = csv
    return by_lang


def main() -> None:
    files = pick_files()
    print(f"Uploading {len(files)} language subsets to {REPO_ID}")
    for lang, path in files.items():
        size_kb = path.stat().st_size / 1024
        print(f"  [{lang}] {path.name} ({size_kb:.0f} KB)")

    for lang, path in files.items():
        print(f"\n== {lang}: loading {path.name} ==")
        ds = Dataset.from_csv(str(path))
        print(f"   rows={len(ds)} cols={ds.column_names}")
        ds.push_to_hub(REPO_ID, config_name=lang, split="train")
        print(f"   pushed {lang}")


if __name__ == "__main__":
    main()
