"""Generate GUDHI data or load, align and subset the AUTOENCODIX TCGA data (Zenodo).

TCGA source: AUTOENCODIX reproducibility archive, doi:10.5281/zenodo.15518831.
The three parquet files must already be in ``data/tcga``: download them once with
``python -m src.core.fetch_tcga`` (byte-range fetch, CRC and SHA-256 checks). Loading
never downloads; it only checks that the files exist with the right size.

Groups (configs/varix.json -> data.tcga.groups, sources in group_sources) select
samples only. Feature filtering (k_filter) and scaling are left to AUTOENCODIX,
which fits them on the training split of whatever samples it receives;
``*_lowvar`` groups only flip the variance ranking (``feature_order``).
Labels are never an estimator or model input.
"""
import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from .load_config import CONFIG, ROOT, load_config

# Data location, origin and groups come from the single config (section "data").
_DATA = load_config(CONFIG)["data"]
_TCGA = _DATA["tcga"]
TCGA_DIR = ROOT / _TCGA["root"]

LABEL = _TCGA["label"]            # CANCER_TYPE_ACRONYM: selects the groups and is the class target
DETAIL = _TCGA["detail"]          # CANCER_TYPE_DETAILED: stays in the returned annotation for analyses

# Group -> None (Pan-Cancer, all samples) or a list of CANCER_TYPE_ACRONYM values.
# Sources per group: configs/varix.json -> data.tcga.group_sources.
GROUPS = {name: (None if spec is None else list(spec)) for name, spec in _TCGA["groups"].items()}

# Weak-feature variants: same samples, but AUTOENCODIX's VAR filter keeps the k_filter
# features with the LOWEST non-zero training variance instead of the highest
# (per modality, RNA and METH; still fitted on the training split only).
LOW_VARIANCE = _TCGA["lowvar_suffix"]
GROUPS.update({name + LOW_VARIANCE: spec for name, spec in list(GROUPS.items())})
ZENODO = _TCGA["source"]

def feature_order(group):
    """'low' for a *_lowvar group (lowest-variance features), else 'top' (AUTOENCODIX default)."""
    return "low" if str(group).endswith(LOW_VARIANCE) else "top"

@contextlib.contextmanager
def variance_order(order="top"):
    """Within the block, AUTOENCODIX's VAR filter keeps the k lowest-variance features.

    ``order="top"`` changes nothing (AUTOENCODIX: k highest variance). ``"low"``
    replaces DataFilter._filter_by_variance by the ascending ranking; zero-variance
    features are still removed before (AUTOENCODIX does that first). Yields a list
    that records the k of every call, so the caller can check the filter ran.
    """
    if order not in ("top", "low"):
        raise ValueError("feature_order must be 'top' or 'low'")
    calls = []
    if order == "top":
        yield calls
        return
    import numpy as np
    import pandas as pd
    from autoencodix.data._filter import DataFilter

    original = DataFilter._filter_by_variance

    def lowest(self, df, k):
        if k is None or k > df.shape[1]:
            return original(self, df, k)
        calls.append(k)
        var = pd.Series(np.var(df, axis=0), index=df.columns)
        return df[var.sort_values(ascending=True, kind="stable").index[:k]]

    DataFilter._filter_by_variance = lowest
    try:
        yield calls
    finally:
        DataFilter._filter_by_variance = original


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


def tcga_files(folder=TCGA_DIR):
    """Paths of the three TCGA parquet files; they must exist with the size of the Zenodo archive."""
    folder = Path(folder)
    paths, problems = {}, []
    for key, member in ZENODO["members"].items():
        path = folder / member["name"]
        if not path.exists():
            problems.append(f"missing: {path}")
        elif path.stat().st_size != member["size"]:
            problems.append(f"wrong size: {path} ({path.stat().st_size} != {member['size']} bytes)")
        paths[key] = path
    if problems:
        raise FileNotFoundError("TCGA data not ready (" + "; ".join(problems) + "). "
                                "Download once with: python -m src.core.fetch_tcga")
    return paths


def select_samples(annotation, group=None):
    """Sample IDs of a configured group (configs/varix.json -> data.tcga.groups), in the given order."""
    if group is None:
        return annotation.index
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group!r}; choose from {sorted(GROUPS)}")
    acronyms = GROUPS[group]
    if acronyms is None:                       # all_classes: Pan-Cancer
        return annotation.index
    unknown = set(acronyms) - set(annotation[LABEL].dropna())
    if unknown:
        raise ValueError(f"Unknown {LABEL} values: {sorted(unknown)}")
    return annotation.index[annotation[LABEL].isin(acronyms).to_numpy()]


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
