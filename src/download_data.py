"""
Download the transfermarkt-datasets CSVs from Kaggle and copy the tables this
project uses into data/.

Kaggle mirror of github.com/dcaribou/transfermarkt-datasets, used because the
project's own download host (r2.dev) is blocked on some networks.

Run: python src/download_data.py
"""

import shutil
import sys
from pathlib import Path

import kagglehub

# `python src/download_data.py` puts src/ on the import path.
from config import DATA_DIR
from data_prep import TABLES

KAGGLE_DATASET = "davidcariboo/player-scores"


def main():
    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded to {src}")

    DATA_DIR.mkdir(exist_ok=True)
    missing = []
    for name in TABLES:
        matches = [p for ext in (".csv.gz", ".csv") for p in src.rglob(f"{name}{ext}")]
        if not matches:
            missing.append(name)
            continue
        shutil.copy2(matches[0], DATA_DIR / matches[0].name)
        print(f"  copied {matches[0].name}")

    if missing:
        sys.exit(f"Not found in the Kaggle download: {', '.join(missing)}")
    print(f"All {len(TABLES)} tables are in {DATA_DIR}")


if __name__ == "__main__":
    main()
