"""Generate GUDHI data or load, align and subset the AUTOENCODIX TCGA data (Zenodo).

TCGA source: AUTOENCODIX reproducibility archive, doi:10.5281/zenodo.15518831.
Only the three needed parquet members are fetched with HTTP range requests and
verified against the archive CRC/size (logic from vae-beta-control
``datasets/fetch_tcga_pancancer.py``). Existing files in ``data/tcga`` are used.

Groups (configs/varix.json -> data.tcga.groups, sources in group_sources) select
samples only. Feature filtering (k_filter) and scaling are left to AUTOENCODIX,
which fits them on the training split of whatever samples it receives;
``*_lowvar`` groups only flip the variance ranking (``feature_order``).
Labels are never an estimator or model input.
"""
import argparse
import json
import shutil
import struct
import time
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "varix.json"
# Data location, origin and groups come from the single config (section "data").
_DATA = json.loads(CONFIG.read_text())["data"]
_TCGA = _DATA["tcga"]
TCGA_DIR = ROOT / _TCGA["root"]

LABEL = _TCGA["label"]
DETAIL = _TCGA["detail"]

# Spec: acronym -> None (whole class) or list of allowed CANCER_TYPE_DETAILED values.
# None as a whole means Pan-Cancer. A plain list of acronyms is accepted as shorthand.
# Sources per group: configs/varix.json -> data.tcga.group_sources.
GROUPS = {name: (None if spec is None else list(spec)) for name, spec in _TCGA["groups"].items()}

# Weak-feature variants: same samples, but AUTOENCODIX's VAR filter keeps the k_filter
# features with the LOWEST non-zero training variance instead of the highest
# (per modality, RNA and METH; still fitted on the training split only).
LOW_VARIANCE = _TCGA["lowvar_suffix"]
GROUPS.update({name + LOW_VARIANCE: spec for name, spec in list(GROUPS.items())})


def feature_order(group):
    """'low' for a *_lowvar group (lowest-variance features), else 'top' (AUTOENCODIX default)."""
    return "low" if str(group).endswith(LOW_VARIANCE) else "top"


ZENODO = _TCGA["source"]


def synthetic():

    gudhi = _DATA["gudhi"]
    folder = ROOT / gudhi["root"]
    path = folder / "sphere.parquet"
    if path.exists():
        return {"sphere": pd.read_parquet(path)}, None

    from gudhi.datasets.generators import points

    np.random.seed(gudhi["numpy_seed"])
    frame = pd.DataFrame(points.sphere(n_samples=gudhi["n_samples"], ambient_dim=gudhi["ambient_dim"]),
                         columns=["x", "y", "z"])
    frame.index = [f"sphere_{i}" for i in range(len(frame))]
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return {"sphere": frame}, None


def _fetch_member(member, destination):
    """Download one ZIP member by byte range, inflate it and verify CRC/size."""
    target = destination / member["name"]
    cache = destination / ".download"
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / (member["name"] + ".zipmember")
    total = member["end_offset"] - member["offset"] + 1
    if shutil.disk_usage(destination).free < member["size"] + total + 256 * 1024**2:
        raise OSError(f"Insufficient disk space for {member['name']}")
    retries = 0
    while not partial.exists() or partial.stat().st_size < total:
        have = partial.stat().st_size if partial.exists() else 0
        start = member["offset"] + have
        end = min(member["end_offset"], start + 16 * 1024 * 1024 - 1)
        request = Request(ZENODO["url"], headers={"Range": f"bytes={start}-{end}",
                                                  "User-Agent": "tcga-loader/1"})
        try:
            with urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Server did not honour the requested byte range")
                with partial.open("ab") as output:
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError("Incomplete range response")
                        output.write(block)
                        remaining -= len(block)
            retries = 0
            print(f"{member['name']}: {partial.stat().st_size / total:.1%}", flush=True)
        except (OSError, TimeoutError) as error:
            retries += 1
            print(f"Keeping partial download after {error}; retry {retries}", flush=True)
            if retries > 10:
                raise
            time.sleep(min(30, retries * 3))
    temporary = target.with_suffix(".tmp")
    with partial.open("rb") as stream, temporary.open("wb") as output:
        header = struct.unpack("<4s5H3I2H", stream.read(30))
        if header[0] != b"PK\x03\x04" or header[3] != 8:
            raise ValueError("Unexpected ZIP member header/compression")
        stream.read(header[-2] + header[-1])
        decoder = zlib.decompressobj(-15)
        remaining, crc, size = member["compressed_size"], 0, 0
        while remaining:
            block = stream.read(min(256 * 1024, remaining))
            if not block:
                raise ValueError("Truncated ZIP member")
            remaining -= len(block)
            decoded = decoder.decompress(block)
            output.write(decoded)
            crc, size = zlib.crc32(decoded, crc), size + len(decoded)
        decoded = decoder.flush()
        output.write(decoded)
        crc, size = zlib.crc32(decoded, crc), size + len(decoded)
    if crc != member["crc"] or size != member["size"] or not decoder.eof:
        temporary.unlink()
        raise ValueError(f"CRC/size check failed for {member['name']}")
    temporary.replace(target)
    return target


def tcga_files(folder=TCGA_DIR):
    """Paths of the Zenodo parquet files; missing ones are downloaded and verified."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, member in ZENODO["members"].items():
        path = folder / member["name"]
        if not path.exists():
            path = _fetch_member(member, folder)
        elif path.stat().st_size != member["size"]:
            raise ValueError(f"{path} has the wrong size; delete it to re-download")
        paths[key] = path
    (folder / "source.json").write_text(json.dumps(ZENODO, indent=2) + "\n")
    return paths


def _normalise(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        return _normalise(GROUPS[spec]) if spec in GROUPS else {spec: None}
    if isinstance(spec, Mapping):
        return {str(k): (None if v is None else tuple(v)) for k, v in spec.items()}
    if isinstance(spec, Iterable):
        return {str(k): None for k in spec}
    raise TypeError("group must be None, a GROUPS key, an acronym, a list or a dict")


def select_samples(annotation, group=None):
    """Sample IDs of a group in the given order; ``group`` may be a GROUPS key."""
    rules = _normalise(group)
    if rules is None:
        return annotation.index
    unknown = set(rules) - set(annotation[LABEL].dropna())
    if unknown:
        raise ValueError(f"Unknown {LABEL} values: {sorted(unknown)}")
    mask = pd.Series(False, index=annotation.index)
    for acronym, details in rules.items():
        hit = annotation[LABEL] == acronym
        if details is not None:
            missing = set(details) - set(annotation.loc[hit, DETAIL].dropna())
            if missing:
                raise ValueError(f"No {DETAIL} {sorted(missing)} within {acronym}")
            hit &= annotation[DETAIL].isin(details)
        mask |= hit
    return annotation.index[mask.to_numpy()]


def tcga(group=None, folder=TCGA_DIR):
    """RNA and methylation for samples present in both, optionally one group.

    Returns ({"RNA": frame, "METH": frame}, annotation), all with the same
    sample index in RNA-file order (deterministic, no set ordering). As in
    AUTOENCODIX's BulkDataReader, Pan-Cancer keeps samples without a clinical
    row (annotation all NaN); a group selection drops them.
    """
    paths = tcga_files(folder)
    rna, meth = pd.read_parquet(paths["RNA"]), pd.read_parquet(paths["METH"])
    annotation = pd.read_parquet(paths["clinical"])
    common = rna.index[rna.index.isin(meth.index)]
    if not common.is_unique:
        raise ValueError("Duplicate TCGA sample IDs")
    keep = select_samples(annotation.reindex(common), group)
    if len(keep) < 32:
        raise ValueError(f"Only {len(keep)} TCGA samples selected")
    return {"RNA": rna.loc[keep], "METH": meth.loc[keep]}, annotation.reindex(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gudhi", "tcga", "both"], default="both")
    parser.add_argument("--group", choices=sorted(GROUPS), default="all_classes")
    args = parser.parse_args()
    names = ["gudhi", "tcga"] if args.dataset == "both" else [args.dataset]
    for name in names:
        modalities, annotation = synthetic() if name == "gudhi" else tcga(args.group)
        for modality, frame in modalities.items():
            print(f"{name}/{modality}: {frame.shape[0]} samples x {frame.shape[1]} features")
        if annotation is not None:
            print(f"{name}/clinical: {len(annotation)} aligned samples")
            print(annotation[LABEL].value_counts().to_string())


if __name__ == "__main__":
    main()
