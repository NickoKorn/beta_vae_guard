"""Download the three TCGA parquet files once (run this before the first training).

    python -m src.core.fetch_tcga            # fetch missing files, verify, record checksums
    python -m src.core.fetch_tcga --verify   # re-hash the local files against data/tcga/source.json

Source: AUTOENCODIX reproducibility archive on Zenodo (doi:10.5281/zenodo.15518831), one ZIP of
~3.3 GB (our three files are ~2.0 GB of it, compressed). The whole ZIP is downloaded in one stream
(resumed with an HTTP Range request after an interruption), the three members are extracted with
zipfile (which checks their CRC-32) and the ZIP is deleted. Member names and sizes:
configs/varix.json -> data.tcga.source. The SHA-256 of every file is written to
data/tcga/source.json, so --verify also detects corrupted files of the right size.
Training never downloads: src/core/load_data.py only checks that the files exist with the right size.
"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .load_config import CONFIG, ROOT, load_config

_TCGA = load_config(CONFIG)["data"]["tcga"]
TCGA_DIR = ROOT / _TCGA["root"]
ZENODO = _TCGA["source"]
RECORD = "source.json"
ARCHIVE = "zenodo_archive.zip.part"


def download(url, target, retries=10):
    """Stream `url` to `target`; after an interruption continue from the current size (Range)."""
    for attempt in range(retries + 1):
        have = target.stat().st_size if target.exists() else 0
        headers = {"User-Agent": "tcga-loader/2"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if have and response.status != 206:
                    have = 0                     # server ignored the range: start over
                total = have + int(response.headers.get("Content-Length", 0))
                with target.open("ab" if have else "wb") as output:
                    while block := response.read(16 * 1024 * 1024):
                        output.write(block)
                        if total:
                            print(f"{target.name}: {output.tell() / total:.1%}", end="\r", flush=True)
            print()
            return target
        except (OSError, TimeoutError) as error:
            if attempt == retries:
                raise
            print(f"\nDownload interrupted ({error}); resuming, attempt {attempt + 1}", flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(folder=TCGA_DIR, url=None):
    """Fetch missing files, check sizes and record SHA-256 checksums; returns the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    members = ZENODO["members"]
    for member in members.values():
        path = folder / member["name"]
        if path.is_symlink() and not path.exists():
            path.unlink()                        # dangling link to an old local copy
    missing = [m["name"] for m in members.values() if not (folder / m["name"]).exists()]
    if missing:
        archive = folder / ARCHIVE
        if shutil.disk_usage(folder).free < 4 * 1024**3 + sum(m["size"] for m in members.values()):
            raise OSError(f"Not enough free disk space in {folder} for the ~3.3 GB archive and the files")
        download(url or ZENODO["url"], archive)
        with zipfile.ZipFile(archive) as zipped:
            for name in missing:
                zipped.extract(name, folder)     # raises zipfile.BadZipFile on a CRC mismatch
        archive.unlink()
    record = {**ZENODO, "sha256": {}}
    paths = {}
    for key, member in members.items():
        path = folder / member["name"]
        if path.stat().st_size != member["size"]:
            raise ValueError(f"{path} has the wrong size; delete it and run this script again")
        record["sha256"][member["name"]] = sha256(path)
        paths[key] = path
    (folder / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    return paths


def verify(folder=TCGA_DIR):
    """Re-hash the local files against the recorded SHA-256 checksums; returns the mismatches."""
    folder = Path(folder)
    recorded = json.loads((folder / RECORD).read_text()).get("sha256")
    if not recorded:
        raise ValueError(f"No checksums in {folder / RECORD}; run python -m src.core.fetch_tcga first")
    return [name for name, digest in recorded.items()
            if not (folder / name).exists() or sha256(folder / name) != digest]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only re-hash the local files")
    parser.add_argument("--folder", type=Path, default=TCGA_DIR)
    args = parser.parse_args()
    if args.verify:
        bad = verify(args.folder)
        print("all files match their SHA-256" if not bad else f"MISMATCH: {bad}")
        raise SystemExit(1 if bad else 0)
    for key, path in fetch(args.folder).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
