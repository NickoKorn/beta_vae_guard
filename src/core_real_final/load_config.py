"""Load the experiment JSON and expand its training configurations (one module for both).

Axes of a run (only the first one is a design choice):
  1. model factors   robustness.factors (one factor at a time around `training`) and
                     optional `interactions` blocks (itertools.product inside each block only)
                     -> training_configs() (a list)
  2. repetitions     groups x training seeds x split seeds (`experiment`): always fully crossed
                     with every configuration, done by the runner, not here
  3. branches        early stop / freeze / ControlVAE from a stored state: nested per annealing
                     (src/experiments), never crossed with the factors
  4. protocol        set points, gains, collapse criteria (`guard`, `protocol`): fixed, not varied

Crossing all model factors (robustness.design = "cartesian") is still possible but explodes
(e.g. 5*5*3*3*2*2*2 = 1800 configurations); interaction blocks keep it to chosen pairs.
"""

import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "varix.json"

# Top-level sections of configs/varix.json (a typo such as "trainig" fails instead of being ignored).
# "interactions" is optional (factor blocks for training_configs); "status" is output and dropped below.
CONFIG_SECTIONS = {
    "dataset", "skip_evaluation", "target", "experiment", "training", "datasets",
    "guard", "robustness", "interactions", "protocol", "data", "runs",
}


def load_config(path=CONFIG):
    """Read a JSON configuration, excluding generated run status."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise TypeError("Settings must be a JSON object")
    # "status" is output, not input: the progress table written by src/experiments/run_status.py.
    # Dropped so it is never checked as a setting, never compared with saved run settings
    # (it changes on every refresh) and never copied into frozen study configs.
    settings.pop("status", None)
    unknown = set(settings) - CONFIG_SECTIONS
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}; allowed: {sorted(CONFIG_SECTIONS)}")
    if settings.get("dataset", "both") not in ("tcga", "gudhi", "both"):
        raise ValueError("dataset must be tcga, gudhi or both")
    for name in ("training", "datasets"):
        if not isinstance(settings.get(name, {}), dict):
            raise ValueError(f"{name} must be an object")
    for name, values in settings.get("datasets", {}).items():
        if name not in ("tcga", "gudhi") or not isinstance(values, dict):
            raise ValueError("datasets must map tcga/gudhi to parameter objects")
    return settings


def config_name(parameters, base):
    """Folder name of a training configuration: every parameter that differs from the base, as
    <parameter><value>, joined by '__'; 'base' if nothing differs.

    Examples: base with drop_p 0.3 -> 'drop_p0.3'; with beta 2 and latent_dim 29 -> 'latent_dim29__beta2'.
    The run folders of the existing grid use exactly these names, so do not change the format.
    """
    changes = []
    for key, value in parameters.items():
        if value != base.get(key):
            # "g": shortest form, so 2.0 -> "2" and 0.0 -> "0" (folder names beta2, drop_p0); keeps
            # 6 significant digits, so levels differing only beyond that would share a name (checked
            # in training_configs)
            level = format(value, "g") if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
            changes.append(f"{key}{level}")
    return "__".join(changes) or "base"


def _check_factors(factors, base, where):
    """Catch silent mistakes: a misspelt parameter, or a single value instead of a list of levels."""
    for key, values in factors.items():
        if key not in base:
            raise ValueError(f"{where}: {key!r} is not a training parameter")
        if not isinstance(values, list):
            raise ValueError(f"{where}: levels of {key!r} must be a list, e.g. [{values!r}]")


def training_configs(settings, *, grid=True, dataset="tcga"):
    """List of unique (name, parameters) training configurations, in run order.

    grid=False: only the base (`training` plus the selected dataset's overrides).
    robustness.factors: one factor at a time around the base (default design, base first), or
    only the points of the full itertools.product if robustness.design == "cartesian".
    interactions: list of factor blocks; itertools.product inside each block, the other
    factors stay at the base. A configuration that appears twice (e.g. the base inside a block)
    is kept once; two different configurations with the same folder name are an error.
    """
    base = {**settings["training"], **settings.get("datasets", {}).get(dataset, {})}
    configs = {}                                   # name -> parameters, in insertion order

    def add(parameters):
        name = config_name(parameters, base)
        if name in configs and configs[name] != parameters:
            raise ValueError(f"Configurations differ but share the name {name!r}; use coarser levels")
        configs.setdefault(name, parameters)

    robustness = settings.get("robustness", {}) if grid else {}
    factors = robustness.get("factors", {})
    design = robustness.get("design", "one-factor-at-a-time")
    if design not in ("cartesian", "one-factor-at-a-time"):
        raise ValueError(f"Unknown robustness design: {design!r}")
    if design != "cartesian" or not factors:
        add(dict(base))                            # OFAT is centred on the base; a product only has its points
    if factors:
        _check_factors(factors, base, "robustness.factors")
        if design == "cartesian":
            for values in product(*factors.values()):
                add({**base, **dict(zip(factors, values))})
        else:
            for key, values in factors.items():
                for value in values:
                    add({**base, key: value})
    for index, block in enumerate(settings.get("interactions", []) if grid else []):
        block = {key: value for key, value in block.items() if not key.startswith("_")}
        _check_factors(block, base, f"interactions[{index}]")
        for values in product(*block.values()):
            add({**base, **dict(zip(block, values))})
    return list(configs.items())
