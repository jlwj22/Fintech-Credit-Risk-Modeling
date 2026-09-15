"""
Download every dataset the project uses into data/raw/.

    python -m src.download_data            # everything
    python -m src.download_data gmsc       # just one source
    python -m src.download_data lending_club fred

Sources
  gmsc          Give Me Some Credit (Kaggle competition). Needs the Kaggle CLI
                with an API token and the competition rules accepted at
                https://www.kaggle.com/c/GiveMeSomeCredit/rules
  lending_club  Lending Club accepted loans 2007-2018Q4 (CC0), mirrored on
                Hugging Face. About 1.7 GB.
  fred          Macro series from the St. Louis Fed (no API key needed):
                national unemployment, credit card charge-off and delinquency rates, and
                the unemployment rate for every state.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

RAW_DIR = "data/raw"
GMSC_PATH = os.path.join(RAW_DIR, "cs-training.csv")
LC_PATH = os.path.join(RAW_DIR, "lending_club", "accepted_2007_to_2018Q4.csv")
LC_URL = (
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/"
    "resolve/main/accepted_2007_to_2018Q4.csv"
)
FRED_DIR = os.path.join(RAW_DIR, "fred")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# National series. CORCCACBS and DRCCLACBS are the quarterly charge-off and
# delinquency rates on credit card loans at all commercial banks, the closest
# public proxy for unsecured consumer credit losses.
FRED_NATIONAL = ["UNRATE", "CORCCACBS", "DRCCLACBS"]

STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS",
    "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]


def _fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, dest)


def download_gmsc() -> None:
    if os.path.exists(GMSC_PATH):
        print(f"gmsc: already have {GMSC_PATH}")
        return
    if shutil.which("kaggle") is None:
        print(
            "gmsc: the Kaggle CLI isn't installed. Either:\n"
            "  pip install kaggle, add your token to ~/.kaggle/kaggle.json, accept\n"
            "  the rules at https://www.kaggle.com/c/GiveMeSomeCredit/rules and rerun, or\n"
            f"  download cs-training.csv from the competition page into {RAW_DIR}/"
        )
        return
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "GiveMeSomeCredit", "-p", RAW_DIR],
        check=True,
    )
    archive = os.path.join(RAW_DIR, "GiveMeSomeCredit.zip")
    with zipfile.ZipFile(archive) as zf:
        zf.extract("cs-training.csv", RAW_DIR)
    os.remove(archive)
    print(f"gmsc: saved {GMSC_PATH}")


def download_lending_club() -> None:
    if os.path.exists(LC_PATH):
        print(f"lending_club: already have {LC_PATH}")
        return
    print("lending_club: downloading ~1.7 GB, this takes a few minutes...")
    _fetch(LC_URL, LC_PATH)
    print(f"lending_club: saved {LC_PATH}")


def download_fred() -> None:
    series = FRED_NATIONAL + [f"{st}UR" for st in STATES]
    for sid in series:
        dest = os.path.join(FRED_DIR, f"{sid}.csv")
        if not os.path.exists(dest):
            _fetch(FRED_URL.format(series=sid), dest)
    print(f"fred: {len(series)} series in {FRED_DIR}/")


SOURCES = {
    "gmsc": download_gmsc,
    "lending_club": download_lending_club,
    "fred": download_fred,
}


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SOURCES)
    for name in wanted:
        if name not in SOURCES:
            sys.exit(f"unknown source {name!r}; choose from {', '.join(SOURCES)}")
        SOURCES[name]()
