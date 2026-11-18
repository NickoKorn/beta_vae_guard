"""Prepare GUDHI/TCGA data and run separate Autoencodix Varix trainings."""
import argparse
import contextlib
import json
import math
import os
# deterministic cuBLAS on the GPU (required by torch for reproducible runs; set before CUDA starts)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from typing import get_args


ROOT = Path(__file__).resolve().parents[2]

def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def load_settings(path, preset=None):
    """Read editable training settings before loading datasets.

    The single config file configs/varix.json holds named presets; select one with
    ``preset=`` or as ``configs/varix.json#<preset>``. A preset overrides only the
    values it lists (deep merge, lists replaced); keys starting with '_' are notes.
    """
    path = str(path)
    if '#' in path:
        path, preset = path.split('#', 1)
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object")
    presets = settings.pop("presets", {})
    settings.pop("status", None)          # run status (output of src/experiments/run_status.py), never an input
    if preset is not None:
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(presets)}")
        settings = _merge(settings, {k: v for k, v in presets[preset].items() if not k.startswith("_")})
    unknown = set(settings) - {"dataset", "group", "skip_evaluation", "target", "training", "datasets",
                               "collapse_monitor", "experiment", "robustness", "guard", "protocol", "data", "runs"}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    if settings.get("dataset", "both") not in ("tcga", "gudhi", "both"):
        raise ValueError("dataset must be tcga, gudhi or both")
    for name in ("training", "datasets"):
        if not isinstance(settings.get(name, {}), dict):
            raise ValueError(f"{name} must be an object")
    for name, values in settings.get("datasets", {}).items():
        if name not in ("tcga", "gudhi") or not isinstance(values, dict):
            raise ValueError("datasets must map tcga/gudhi to parameter objects")
    return settings


def main():
    from autoencodix.configs.varix_config import VarixConfig

    schedules = get_args(VarixConfig.model_fields["anneal_function"].annotation)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "varix.json")
    parser.add_argument("--dataset", choices=["gudhi", "tcga", "both"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--anneal", choices=schedules)
    parser.add_argument("--skip-evaluation", action="store_true", default=None)
    parser.add_argument("--target")
    from .load_data import GROUPS, feature_order, synthetic, tcga
    parser.add_argument("--group", choices=sorted(GROUPS), help="TCGA sample group (default: config, else all_classes)")
    args = parser.parse_args()
    settings = load_settings(args.config)
    selected = args.dataset or settings.get("dataset", "both")
    common = settings.get("training", {})
    overrides = {key: value for key, value in {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "beta": args.beta, "anneal_function": args.anneal,
    }.items() if value is not None}
    group = args.group or settings.get("group", "all_classes")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {sorted(GROUPS)}")
    for name in (["gudhi", "tcga"] if selected == "both" else [selected]):
        parameters = {**common, **settings.get("datasets", {}).get(name, {}), **overrides}
        # latent_dim may be a list: one separate run per latent dimension
        dims = parameters.pop("latent_dim", None)
        dims = dims if isinstance(dims, list) else [dims]
        lists = sorted(key for key, value in parameters.items() if isinstance(value, list))
        if lists:
            raise ValueError(f"Only latent_dim may be a list; got lists for {lists}")
        modalities, annotation = synthetic() if name == "gudhi" else tcga(group)
        for dim in dims:
            run_parameters = parameters if dim is None else {**parameters, "latent_dim": int(dim)}
            folder = ROOT / "local" / "runs" / name / (group if name == "tcga" else "") / f"latent_dim_{dim}"
            train_varix(modalities, annotation, dataset=name, config_overrides=run_parameters,
                        output_dir=folder, guard=settings.get("guard"),
                        feature_order=feature_order(group) if name == "tcga" else "top",
                        skip_evaluation=(args.skip_evaluation if args.skip_evaluation is not None
                                         else settings.get("skip_evaluation", False)),
                        target=args.target or settings.get("target", "CANCER_TYPE_ACRONYM"))


def drop_checkpoint_reconstructions(result):
    """Remove reconstructions of all samples stored at training checkpoints before saving.

    They are most of pipeline.pkl (every checkpoint holds train/valid x features) and
    nothing here uses them; the test reconstructions of predict() (epoch -1), which
    Autoencodix reads, are kept. mu, logvar and z of every checkpoint stay.
    """
    data = result.reconstructions._data
    for epoch in [e for e in data if e != -1]:
        del data[epoch]


def save_latents(result, path):
    """mu and logvar of train/valid (last stored epoch) and test (predict) as one parquet."""
    import numpy as np
    import pandas as pd

    last = max(e for e in result.mus._data if e >= 0)
    frames = []
    for split, epoch in (("train", last), ("valid", last), ("test", -1)):
        mu = result.mus.get(epoch=epoch, split=split)
        if mu is None or np.asarray(mu).size == 0:
            continue
        mu, logvar = np.asarray(mu, float), np.asarray(result.sigmas.get(epoch=epoch, split=split), float)
        frame = pd.DataFrame(mu, columns=[f"mu_{j}" for j in range(mu.shape[1])])
        frame[[f"logvar_{j}" for j in range(mu.shape[1])]] = logvar
        frame.insert(0, "sample_id", np.asarray(result.sample_ids.get(epoch=epoch, split=split)).astype(str))
        frame.insert(0, "epoch", epoch)
        frame.insert(0, "split", split)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


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


def train_varix(modalities, annotation=None, *, dataset="tcga", epochs=30,
                output_dir=None, skip_evaluation=False, config_overrides=None,
                target="CANCER_TYPE_ACRONYM", guard=None, split_seed=None, train_seed=None,
                on_epoch=None, save_pipeline=True, feature_order="top"):
    """Preprocess, train, predict and save Varix; return the trained pipeline.

    modalities maps modality names to sample-indexed DataFrames. ``beta`` and
    ``anneal_function`` go into ``config_overrides``; the schedule is
    Autoencodix's native one (VarixConfig defaults apply if omitted).
    Training always runs through GuardedTrainer.

    ``split_seed`` drives the Autoencodix split (preprocess), ``train_seed`` the
    initialisation and batches (fit): Autoencodix uses the one ``global_seed``
    for both, so it is set to split_seed before preprocess and to train_seed
    before fit. If omitted, ``global_seed`` from the config is used for both.

    ``feature_order="low"`` keeps the k_filter lowest-variance features instead of
    the highest (``load_data.feature_order(group)`` for *_lowvar groups); same k_filter.
    """
    if dataset not in ("tcga", "gudhi"):
        raise ValueError("dataset must be 'tcga' or 'gudhi'")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    import autoencodix as acx
    from autoencodix.configs.default_config import DataCase
    from autoencodix.configs.varix_config import VarixConfig
    from autoencodix.data.datapackage import DataPackage

    output = Path(output_dir).resolve() if output_dir is not None else ROOT / "local" / "runs" / dataset
    output.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    try:
        os.chdir(output)
        parameters = dict(data_case=DataCase.MULTI_BULK, epochs=epochs,
                          device="cpu", latent_dim=3 if dataset == "gudhi" else 16,
                          n_layers=1 if dataset == "gudhi" else 3,
                          batch_size=256, reproducible=True, global_seed=42,
                          **({"k_filter": 2000} if dataset == "tcga" else {}))
        extra = dict(config_overrides or {})
        unknown = set(extra) - set(VarixConfig.model_fields)
        if unknown:
            raise ValueError(f"Unknown Varix parameters: {sorted(unknown)}")
        parameters.update(extra)
        if split_seed is not None:
            parameters["global_seed"] = int(split_seed)
        config = VarixConfig(**parameters)
        if not math.isfinite(config.beta):  # VarixConfig only enforces beta >= 0
            raise ValueError("beta must be finite")
        (output / "config.json").write_text(config.model_dump_json(indent=2))
        data = DataPackage(multi_bulk=modalities,
                           annotation={"paired": annotation} if annotation is not None else None)
        print(f"Starting Varix: {dataset}\n{data}", flush=True)
        model = acx.Varix(data=data, config=config)
        with variance_order(feature_order) as filtered:
            model.preprocess()
        if feature_order == "low" and not filtered:
            raise ValueError("feature_order='low' needs the VAR filter with k_filter set")
        manifest = dict(
            sample_ids={split: [str(v) for v in getattr(model.preprocessed_data, split).sample_ids]
                        for split in ("train", "valid", "test")
                        if getattr(model.preprocessed_data, split) is not None},
            feature_ids=[str(v) for v in model.preprocessed_data.train.feature_ids],
            pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        )
        (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        split_used = model.config.global_seed
        if train_seed is not None:
            model.config.global_seed = int(train_seed)  # Autoencodix reseeds from it in fit()
        (output / "run_info.json").write_text(json.dumps(dict(
            dataset=dataset, split_seed=int(split_used), train_seed=int(model.config.global_seed),
            n_train=len(model.preprocessed_data.train.sample_ids),
            n_valid=len(model.preprocessed_data.valid.sample_ids),
            n_test=len(model.preprocessed_data.test.sample_ids),
            n_features=len(model.preprocessed_data.train.feature_ids),
            feature_order=feature_order,
            loss_reduction=config.loss_reduction, latent_dim=config.latent_dim, beta_max=config.beta), indent=2) + "\n")
        if config.default_vae_loss != "kl":
            raise ValueError("Collapse monitoring currently requires default_vae_loss='kl'")
        from src.core.collapse_guard import current_options
        guard = current_options(guard)
        guard_enabled = bool(guard.pop("enabled", True))
        collapse_threshold = float(guard.pop("collapse_value", 0.01))
        if not math.isfinite(collapse_threshold) or collapse_threshold <= 0:
            raise ValueError("guard.collapse_value must be finite and positive")
        if "band" in guard:
            guard["band"] = tuple(guard["band"])
        from src.core.guarded_trainer import GuardedTrainer
        from functools import partial
        history = output / "collapse_history.jsonl"
        if history.exists():
            raise FileExistsError(f"Use a fresh output_dir to preserve existing history: {history}")
        model._trainer_type = partial(GuardedTrainer, collapse_threshold=collapse_threshold,
                                      collapse_output=str(output), guard_enabled=guard_enabled,
                                      guard_options=guard, on_epoch=on_epoch)
        model.fit()
        from src.analysis.plots import plot_run
        plot_run(output)  # guard_diagnostic.pdf next to the history
        model.predict()
        drop_checkpoint_reconstructions(model.result)
        # The raw input (all ~25k RNA/METH features, ~2 GB for all_classes) is most of the
        # pickle; evaluation needs only the preprocessed splits. It is reloadable from the
        # Zenodo files with load_data.tcga(group).
        model.raw_user_data = None
        if not save_pipeline:
            # sweeps/notebook: ~1 MB of latents instead of a ~850 MB pipeline per run;
            # the evaluation still runs on the in-memory model
            save_latents(model.result, output / "latents.parquet")
            if not skip_evaluation:
                from src.analysis.latent import evaluate_pipeline
                evaluate_pipeline(model, output, target=target if annotation is not None else None)
            print(f"Finished Varix: {dataset} (latents only)", flush=True)
            return model
        model.save(str(output / "pipeline.pkl"), save_all=True)
        if not skip_evaluation:
            from src.analysis.latent import evaluate_pipeline
            evaluate_pipeline(model, output, target=target if annotation is not None else None)
        print(f"Finished Varix: {dataset}", flush=True)
        return model
    finally:
        os.chdir(previous_dir)

if __name__ == "__main__":
    main()
