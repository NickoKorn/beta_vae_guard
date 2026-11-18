"""Uniform access to saved runs and the cross-study run register (analysis only, no training).

Every study in configs/varix.json -> runs.studies has a path, a role (primary, reference,
baseline, superseded) and a layout. The register has one row per saved run (annealing
or continuation branch) with study, role, method, configuration, group, split, seed,
checkpoint and path; every analysis selects its runs from it.

    python -m src.analysis.runs          # rebuild results/run_register.parquet (+ .csv)

Methods: annealing (native schedule, no intervention), early_stop, freeze, controlvae,
or the form of the old guard (dynamicvae, nupi, log). Configuration names follow the
threshold ladder (base, drop_p0.5, loss_reductionsum, ...); runs of configurations that
are no longer in robustness.factors get role "outside_grid".
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd

from src.core.train_varix import ROOT

CONFIG = ROOT / "configs" / "varix.json"
COLUMNS = ["study", "role", "layout", "method", "branch", "config", "group", "split", "seed", "checkpoint",
           "complete", "signal", "target", "beta_start", "ki_step", "path"]


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _level(value):
    return format(value, "g") if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def grid_configs(config=None):
    """Names of the configurations of the current robustness grid (base + one factor at a time)."""
    config = config or json.loads(CONFIG.read_text())
    base, factors = config["training"], config["robustness"]["factors"]
    return {"base"} | {f"{f}{_level(v)}" for f, values in factors.items() for v in values if v != base[f]}


def canonical_config(name, factors):
    """'learning_rate_1e-05' (grid/last-axis spelling) -> 'learning_rate1e-05' (ladder spelling)."""
    for factor in sorted(factors, key=len, reverse=True):
        if name.startswith(factor + "_"):
            return factor + name[len(factor) + 1:]
    return name


def _method_from_branch(branch):
    for method in ("early_stop", "controlvae", "freeze", "annealing"):
        if branch.startswith(method):
            return method
    return "unguarded" if "unguarded" in branch else branch


# --------------------------------------------------------------------------------------- layouts
def _threshold_ladder(study_dir, protocol):
    """runs/<group>/seed<s>/... | runs/<config>/<group>/seed<s>/... | runs/split<k>/<config>/<group>/seed<s>/..."""
    rows = []
    runs = study_dir / "runs"
    for done in list(runs.rglob("annealing/annealing_complete.json")) + list(runs.rglob("*/run_complete.json")):
        parts = done.relative_to(runs).parts
        if parts[0].startswith("split"):
            split, config, group, rest = int(parts[0][5:]), parts[1], parts[2], parts[3:]
        elif "variants" in protocol:
            split, config, group, rest = protocol.get("split_seed", 2), parts[0], parts[1], parts[2:]
        else:
            split, config, group, rest = protocol.get("split_seed", 2), "base", parts[0], parts[1:]
        if not rest or not rest[0].startswith("seed"):
            continue
        seed = int(rest[0][4:])
        if rest[1] == "annealing":
            rows.append(dict(method="annealing", branch="annealing", checkpoint=None, config=config, group=group,
                             split=split, seed=seed, path=done.parent, complete=True))
        else:
            meta = _read(done) or {}
            rows.append(dict(method=meta.get("method", rest[2]), branch=rest[2], checkpoint=rest[1], config=config,
                             group=group, split=split, seed=seed, path=done.parent, complete=True,
                             signal=meta.get("signal"), target=meta.get("target"), beta_start=meta.get("beta_start"),
                             ki_step=meta.get("ki_step")))
    return rows


def _last_axis(study_dir, protocol, factors):
    """runs/tcga/<group>/<config>/split<k>/seed<s>/<branch>/ (branch = annealing, early_stop, freeze_*, controlvae_*)."""
    rows = []
    for done in (study_dir / "runs" / "tcga").glob("*/*/split*/seed*/*/run_complete.json"):
        branch_dir = done.parent
        group, config, split, seed, branch = branch_dir.parts[-5:]
        meta = _read(done) or {}
        rows.append(dict(method=_method_from_branch(branch), branch=branch, checkpoint="last_axis_onset-3",
                         config=canonical_config(config, factors), group=group, split=int(split[5:]),
                         seed=int(seed[4:]), path=branch_dir, complete=True, signal="lucas_max" if
                         branch.startswith("controlvae") else None, target=meta.get("pi_target"),
                         beta_start=meta.get("beta"), ki_step=None))
    for done in (study_dir / "runs" / "tcga").glob("*/*/split*/seed*/annealing/guard_summary.json"):
        group, config, split, seed = done.parent.parts[-5:-1]
        rows.append(dict(method="annealing", branch="annealing", checkpoint=None,
                         config=canonical_config(config, factors), group=group, split=int(split[5:]),
                         seed=int(seed[4:]), path=done.parent, complete=True))
    return rows


def _sweep(study_dir, protocol, factors):
    """<phase>/<variant>/seed<s>/ with variant.json (guard dict) and guard_summary.json."""
    rows = []
    for done in study_dir.glob("*/*/seed*/guard_summary.json"):
        meta = _read(done.parent / "variant.json") or {}
        guard = meta.get("guard", {})
        training = guard.get("training") or {}
        config = "base" if not training else "".join(f"{k}{_level(v)}" for k, v in training.items())
        if guard.get("enabled") is False:
            method = "annealing"
        else:
            method = guard.get("form", "controlvae")
        rows.append(dict(method=method, branch=meta.get("variant", done.parent.parent.name),
                         checkpoint=f"phase {meta.get('phase', done.parts[-4])}", config=config,
                         group=guard.get("group", "all_classes"), split=guard.get("split", 2),
                         seed=meta.get("seed", int(done.parent.name[4:])), path=done.parent, complete=True,
                         signal=guard.get("signal", "total (training batches)"), target=guard.get("control_value", guard.get("trip")),
                         ki_step=guard.get("ki")))
    return rows


def _generic(study_dir, protocol, factors):
    """Fallback: every folder with run_complete.json or guard_summary.json; method from the path."""
    rows, seen = [], set()
    for name in ("run_complete.json", "guard_summary.json"):
        for done in study_dir.rglob(name):
            if done.parent in seen or "source" in done.relative_to(study_dir).parts:
                continue
            seen.add(done.parent)
            rel = done.parent.relative_to(study_dir).parts
            method = next((_method_from_branch(p) for p in reversed(rel)
                           if _method_from_branch(p) in ("early_stop", "controlvae", "freeze", "annealing",
                                                         "unguarded")), None)
            summary = _read(done.parent / "guard_summary.json") or {}
            if method is None and summary.get("form"):
                method = summary["form"] if summary.get("guard_enabled", True) else "annealing"
            if method is None and rel and rel[-1].endswith("_check"):
                method = "check"
            config = next((canonical_config(p, factors) for p in rel
                           if p == "base" or canonical_config(p, factors) != p), "base")
            group = next((p for p in rel if p in ("all_classes", "ontix_standard", "weak_signal_lowvar",
                                                  "weak_signal")), None)
            split = next((int(p[5:]) for p in rel if p.startswith("split") and p[5:].isdigit()), None)
            seed = next((int(p[4:]) for p in rel if p.startswith("seed") and p[4:].isdigit()), None)
            rows.append(dict(method=method or "unknown", branch=rel[-1] if rel else "", checkpoint=None,
                             config=config, group=group, split=split, seed=seed, path=done.parent, complete=True,
                             signal=summary.get("signal"), target=summary.get("control_value", summary.get("trip")), ki_step=summary.get("ki")))
    return rows


LAYOUTS = {"threshold_ladder": lambda d, p, f: _threshold_ladder(d, p), "last_axis": _last_axis,
           "sweep": _sweep, "generic": _generic}


def build_register(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    factors = config["robustness"]["factors"]
    grid = grid_configs(config)
    rows = []
    for study, spec in config["runs"]["studies"].items():
        layout, role = spec.get("layout"), spec.get("role")
        study_dir = ROOT / spec["path"]
        if layout not in LAYOUTS or not study_dir.exists():
            continue
        protocol = _read(study_dir / "protocol.json") or {}
        for row in LAYOUTS[layout](study_dir, protocol, factors):
            row.update(study=study, layout=layout,
                       role=role if role != "primary" or row["config"] in grid else "outside_grid",
                       path=str(Path(row["path"]).relative_to(ROOT)))
            rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    return table.sort_values(["role", "study", "config", "group", "split", "seed", "checkpoint", "branch"],
                             na_position="first").reset_index(drop=True)


def load_register(path=None):
    path = Path(path) if path else ROOT / json.loads(CONFIG.read_text())["runs"]["register"]
    return pd.read_parquet(path)


# --------------------------------------------------------------------------------------- one run
@dataclass
class Run:
    """All saved measurements of one run folder, loaded the same way for every study."""
    path: Path
    meta: dict = field(default_factory=dict)
    history: pd.DataFrame | None = None          # collapse_history.jsonl, one row per validation epoch
    triggers: dict | None = None                 # ladder annealing only
    posteriors: list = field(default_factory=list)

    def steps(self):
        """Per training batch (KL, error, P, I, beta); loaded on demand, can be large."""
        path = self.path / "steps.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def posterior(self, index=-1):
        """mu/logvar of train and valid at a stored checkpoint (default: the last one)."""
        import numpy as np
        if not self.posteriors:
            return None
        return dict(np.load(self.path / self.posteriors[index]["path"]))


def load_run(path):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    meta = {}
    for name in ("run_complete.json", "annealing_complete.json", "variant.json", "guard_summary.json",
                 "start_state_check.json"):
        value = _read(path / name)
        if value:
            meta[name.replace(".json", "")] = value
    history = None
    if (path / "collapse_history.jsonl").exists():
        history = pd.read_json(path / "collapse_history.jsonl", lines=True)
    index = path / "posterior_index.jsonl"
    posteriors = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    return Run(path=path, meta=meta, history=history, triggers=_read(path / "triggers.json"), posteriors=posteriors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    config = json.loads(CONFIG.read_text())
    table = build_register()
    out = ROOT / config["runs"]["register"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    table.to_csv(out.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 200)
    print(table.groupby(["role", "study", "method"]).size().rename("runs").to_string())
    print(f"\n{len(table)} runs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
