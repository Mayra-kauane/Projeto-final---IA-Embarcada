from pathlib import Path
from zipfile import ZipFile

import requests


DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip",
]
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "uci_har_smartphones.zip"
EXTRACT_DIR = DATA_DIR / "uci_har"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print("Baixando dataset UCI HAR...")
        last_error = None
        for url in DATA_URLS:
            try:
                print(f"Tentando: {url}")
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                ZIP_PATH.write_bytes(response.content)
                break
            except requests.RequestException as error:
                last_error = error
                print(f"Falhou nessa URL: {error}")
        else:
            raise RuntimeError("Nao foi possivel baixar o dataset.") from last_error
    else:
        print("Arquivo ZIP ja existe. Pulando download.")

    if EXTRACT_DIR.exists():
        print("Dataset ja extraido. Pulando extracao.")
        return

    print("Extraindo dataset...")
    EXTRACT_DIR.mkdir(exist_ok=True)
    with ZipFile(ZIP_PATH) as outer_zip:
        outer_zip.extractall(EXTRACT_DIR)

    nested_zips = list(EXTRACT_DIR.rglob("*.zip"))
    for nested_zip in nested_zips:
        target = nested_zip.with_suffix("")
        target.mkdir(exist_ok=True)
        with ZipFile(nested_zip) as inner_zip:
            inner_zip.extractall(target)

    print(f"Dataset pronto em: {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
