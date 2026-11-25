"""Prepare the TCGA parquet files used by guarded Varix."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from .load_data import tcga_files


ZENODO_ARCHIVE = (
    "https://zenodo.org/api/records/15518831/files/"
    "AutoencodixZenodoReproducibility_v2.zip/content"
)


def fetch_tcga(folder, source):
    folder = Path(folder).expanduser()
    members = source["members"]
    missing = []
    for member in members.values():
        path = folder / member["name"]
        if path.exists():
            if not path.is_file() or path.stat().st_size != member["size"]:
                raise ValueError(f"Existing TCGA file has an unexpected size: {path}")
        else:
            missing.append(member)
    if not missing:
        return tcga_files(folder, members)

    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcga-download-", dir=folder) as temporary:
        temporary = Path(temporary)
        archive = temporary / "tcga.zip"
        url = source.get("url", ZENODO_ARCHIVE)
        print(f"Downloading TCGA from {url}", flush=True)
        urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as data:
            for member in missing:
                name = member["name"]
                if Path(name).name != name:
                    raise ValueError(f"Expected a TCGA filename: {name}")
                extracted = temporary / name
                with data.open(name) as source_file, extracted.open("wb") as output:
                    shutil.copyfileobj(source_file, output)
                if extracted.stat().st_size != member["size"]:
                    raise ValueError(f"Unexpected extracted TCGA file size: {name}")
                try:
                    os.link(extracted, folder / name)
                except FileExistsError:
                    print(f"Something happened: {FileExistsError}")
    return tcga_files(folder, members)
