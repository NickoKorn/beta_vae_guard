"""Load TCGA data for Varix."""

import contextlib
from pathlib import Path


def select_samples(annotation, group, tcga_config):
    """Select configured TCGA sample IDs while preserving annotation order."""
    if group is None:
        return annotation.index

    groups = tcga_config["groups"]
    suffix = tcga_config["lowvar_suffix"]
    base_group = group[:-len(suffix)] if suffix and group.endswith(suffix) else group
    if base_group not in groups:
        available = sorted(groups) + sorted(name + suffix for name in groups)
        raise ValueError(f"Unknown TCGA group {group!r}; choose from {available}")

    acronyms = groups[base_group]
    if acronyms is None:
        return annotation.index

    label = tcga_config["label"]
    unknown = set(acronyms) - set(annotation[label].dropna())
    if unknown:
        raise ValueError(f"Unknown {label} values: {sorted(unknown)}")
    return annotation.index[annotation[label].isin(acronyms).to_numpy()]


def tcga_files(folder, members):
    """Resolve the three TCGA files described by the selected configuration."""
    folder = Path(folder).expanduser()
    paths, problems = {}, []
    for key in ("RNA", "METH", "clinical"):
        member = members[key]
        path = folder / member["name"]
        paths[key] = path
        if not path.is_file():
            problems.append(f"missing file: {path}")
        else:
            size = path.stat().st_size
            if size != member["size"]:
                problems.append(f"wrong size: {path} ({size} != {member['size']} bytes)")
    if problems:
        raise FileNotFoundError("TCGA data not ready (" + "; ".join(problems) + ")")
    return paths


def tcga(group, folder, tcga_config):
    """Load and align configured TCGA modalities and annotations for one group."""
    import pandas as pd

    members = tcga_config["source"]["members"]
    paths = tcga_files(folder, members)
    rna = pd.read_parquet(paths["RNA"])
    meth = pd.read_parquet(paths["METH"])
    annotation = pd.read_parquet(paths["clinical"])

    common = rna.index[rna.index.isin(meth.index)]
    if not common.is_unique:
        raise ValueError("Duplicate TCGA sample IDs in the RNA/METH intersection")

    aligned_annotation = annotation.reindex(common)
    sample_ids = select_samples(aligned_annotation, group, tcga_config)
    if len(sample_ids) < 32:
        raise ValueError(f"Only {len(sample_ids)} TCGA samples selected for group {group!r}")

    return {
        "RNA": rna.loc[sample_ids],
        "METH": meth.loc[sample_ids],
    }, aligned_annotation.reindex(sample_ids)


def gudhi(topology, folder, gudhi_config):
    """Load or generate a GUDHI point cloud with known circle or sphere topology.

    ``circle`` is GUDHI's 1-sphere in R² (Betti numbers H0=1, H1=1).
    ``sphere`` is GUDHI's 2-sphere in R³ (Betti numbers H0=1, H1=0, H2=1).
    Returns a one-modality mapping and an index-only annotation. AUTOENCODIX's
    paired multi-bulk preprocessor requires annotation sample IDs even without labels.
    """
    import numpy as np
    import pandas as pd
    from gudhi.datasets.generators import points

    dimensions = {"circle": 2, "sphere": 3}
    if topology not in dimensions:
        raise ValueError(f"Unknown GUDHI topology {topology!r}; choose from {sorted(dimensions)}")
    generator = gudhi_config.get("generator", "gudhi.datasets.generators.points.sphere")
    if generator != "gudhi.datasets.generators.points.sphere":
        raise ValueError(f"Unsupported GUDHI generator: {generator!r}")
    n_samples = int(gudhi_config["n_samples"])
    seed = int(gudhi_config["numpy_seed"])
    radius = float(gudhi_config.get("radius", 1.0))
    if n_samples < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("GUDHI n_samples and radius must be positive")

    folder = Path(folder).expanduser()
    path = folder / f"{topology}_n{n_samples}_r{radius:g}_seed{seed}.parquet"
    if path.is_file():
        frame = pd.read_parquet(path)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            coordinates = points.sphere(
                n_samples=n_samples, ambient_dim=dimensions[topology], radius=radius,
            )
        finally:
            np.random.set_state(random_state)
        columns = [f"x{i}" for i in range(dimensions[topology])]
        index = [f"{topology}_{i}" for i in range(n_samples)]
        frame = pd.DataFrame(coordinates, columns=columns, index=index)
        frame.to_parquet(path)

    expected_shape = (n_samples, dimensions[topology])
    if frame.shape != expected_shape:
        raise ValueError(f"Cached GUDHI data has shape {frame.shape}, expected {expected_shape}: {path}")
    annotation = pd.DataFrame(index=frame.index)
    return {topology: frame}, annotation


@contextlib.contextmanager
def variance_order(order="top"):
    """Temporarily select highest- or lowest-variance features in AUTOENCODIX."""
    if order not in ("top", "low"):
        raise ValueError("order must be 'top' or 'low'")
    calls = []
    if order == "top":
        yield calls
        return

    import numpy as np
    import pandas as pd
    from autoencodix.data._filter import DataFilter

    original = DataFilter._filter_by_variance

    def select_lowest(self, frame, k):
        if k is None or k > frame.shape[1]:
            return original(self, frame, k)
        calls.append(k)
        variances = pd.Series(np.var(frame, axis=0), index=frame.columns)
        selected = variances.sort_values(ascending=True, kind="stable").index[:k]
        return frame[selected]

    DataFilter._filter_by_variance = select_lowest
    try:
        yield calls
    finally:
        DataFilter._filter_by_variance = original
