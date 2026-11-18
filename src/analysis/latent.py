"""Quality of the latent space of a trained pipeline (analysis only, no training).

    latent_metrics(pipeline, mu)       AUTOENCODIX-native total correlation and coverage
    measure_preservation(original, z)  ZADU: LCMC (local), Spearman (global), steadiness/cohesiveness
    evaluate_pipeline(pipeline, out)   test-split report: KL per dimension, TC, coverage,
                                       preservation, downstream (AUTOENCODIX evaluate), native plots
    python -m src.analysis.latent PIPELINE [--output DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.core.guarded_trainer import KL_per_dim
from src.analysis.plots import plot_all

def as_array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Expected a finite nonempty samples-by-features matrix")
    return array


def measure_preservation(original, embedding, ks=(5, 20), seed=42):
    """Rows must describe exactly the same samples in exactly the same order.

    LCMC is chance-corrected neighborhood overlap (best: 1-k/(n-1)).
    Spearman measures global distance-rank agreement (best: 1).
    Steadiness/cohesiveness measures false/missing cluster structure (best: 1).
    """

    from zadu import ZADU

    original = np.asarray(original, dtype=float)
    embedding = np.asarray(embedding, dtype=float)
    if (original.ndim != 2 or embedding.ndim != 2 or
            len(original) != len(embedding) or min(original.shape) < 1 or
            min(embedding.shape) < 1):
        raise ValueError("Original and embedding must have matching nonempty sample rows")
    if not np.isfinite(original).all() or not np.isfinite(embedding).all():
        raise ValueError("Preservation inputs must be finite")
    if not ks or any(not isinstance(k, int) or not 1 <= k < len(original) for k in ks):
        raise ValueError("Each k must be an integer with 1 <= k < number of samples")
    specs, groups = [], []
    for k in ks:
        specs.append({"id": "lcmc", "params": {"k": k}})
        groups.append(("local", k))
    specs.append({"id": "srho"})
    groups.append(("global", None))
    for k in ks:
        specs.append({"id": "snc", "params": {"k": k, "random_state": seed,
                                                "clustering_strategy": "dbscan"}})
        groups.append(("clustering", k))
    scores = ZADU(specs, original).measure(embedding)
    if len(scores) != len(specs):
        raise RuntimeError("Unexpected ZADU result count")
    report = {"samples": len(original), "embedding": "posterior_mean",
              "original_space": "preprocessed_features", "seed": seed,
              "local": [], "global": [], "clustering": []}
    for spec, (group, k), values in zip(specs, groups, scores):
        values = {key: float(value) for key, value in values.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite ZADU result for {spec['id']}; inspect degenerate inputs")
        entry = {"metric": spec["id"], "scores": values}
        if k is not None:
            entry["k"] = k
        if group == "local":
            entry["best_possible"] = 1 - k / (len(original) - 1)
        report[group].append(entry)
    return report


def latent_metrics(pipeline, mu):
    """Use Autoencodix's Coverage-Correlation visualizer metrics."""
    import pandas as pd

    frame = pd.DataFrame(as_array(mu))
    metrics = {}
    for key, method in (("total_correlation", "_total_correlation"),
                        ("coverage", "_coverage_calc")):
        compute = getattr(pipeline.visualizer, method, None)
        if not callable(compute):
            raise RuntimeError(f"Autoencodix visualizer lacks {method}; check package version")
        value = float(compute(frame))
        metrics[key] = {"value": value if np.isfinite(value) else None,
                        "status": "ok" if np.isfinite(value) else "undefined",
                        "method": f"autoencodix.visualizer.{method}",
                        "space": "posterior_means", "samples": len(frame),
                        "latent_dimensions": frame.shape[1]}
        if key == "coverage" and not np.isfinite(value):
            metrics[key]["reason"] = "Native coverage needs at least two bins per dimension"
    return metrics


def evaluate_pipeline(pipeline, output, target="CANCER_TYPE_ACRONYM"):
    """Evaluate final held-out predictions and save reports beside the model."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = pipeline.result
    logvar = as_array(result.sigmas.get(epoch=-1, split="test"))
    raw_mu = np.asarray(result.mus.get(epoch=-1, split="test"))
    # Autoencodix stores a buffer only if its sum is nonzero (maybe_add in _general_trainer).
    # After total collapse Varix sets every posterior mean to exactly 0, so mu is missing;
    # its true value is the zero matrix of the stored logvar shape.
    mu_all_zero = raw_mu.size == 0
    mu = np.zeros_like(logvar) if mu_all_zero else as_array(raw_mu)
    import torch

    R_j, v_j = KL_per_dim(torch.as_tensor(mu), torch.as_tensor(logvar))
    report = {"split": "test", "mu_all_zero_not_stored_by_autoencodix": bool(mu_all_zero),
              "posterior_collapse": {
                  "kl_per_dimension_nats": R_j.tolist(),
                  "mean_total_kl_nats": float(R_j.sum()),
                  "half_posterior_mean_sample_variance": v_j.tolist(),
              },
              **latent_metrics(pipeline, mu)}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    # Match test posterior rows to preprocessed input rows by sample ID.
    ids = np.asarray(result.sample_ids.get(epoch=-1, split="test")).astype(str)
    data_ids = np.asarray(result.datasets.test.sample_ids).astype(str)
    if (ids.ndim != 1 or data_ids.ndim != 1 or len(ids) != len(mu) or
            len(set(ids)) != len(ids) or len(set(data_ids)) != len(data_ids) or
            set(ids) != set(data_ids)):
        raise ValueError("Test posterior and input sample IDs must match uniquely")
    positions = {sample: i for i, sample in enumerate(data_ids)}
    original = as_array(result.datasets.test.data)[[positions[sample] for sample in ids]]
    if np.ptp(mu, axis=0).max() == 0:
        # total collapse: all posterior means identical, all latent distances 0
        preservation = {"status": "undefined",
                        "reason": "all test posterior means are identical (total collapse)"}
    else:
        preservation = measure_preservation(original, mu)
    (output / "preservation.json").write_text(json.dumps(preservation, indent=2, allow_nan=False))
    report["preservation"] = preservation
    # Preserve diagnostics even if the downstream evaluator encounters a problem.
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if target is None:
        report["downstream"] = {"status": "skipped", "reason": "No target annotation"}
    else:
        from sklearn.linear_model import LogisticRegression

        evaluated = pipeline.evaluate(params=[target], split_type="use-split",
                                      ml_model_class=LogisticRegression(max_iter=2000, random_state=42),
                                      metric_class="roc_auc_ovo", reference_methods=[])
        evaluated.embedding_evaluation.to_csv(output / "downstream.csv", index=True)
        report["downstream"] = {"target": target, "metric": "roc_auc_ovo",
                                "estimator": "LogisticRegression", "split": "use-split",
                                "report": "downstream.csv"}
    report["plots"] = plot_all(pipeline, output, target=target)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path, help="Locally saved Autoencodix pipeline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", default="CANCER_TYPE_ACRONYM")
    parser.add_argument("--no-downstream", action="store_true")
    args = parser.parse_args()

    import autoencodix as acx

    pipeline = acx.Varix.load(str(args.pipeline))
    evaluate_pipeline(pipeline, args.output or args.pipeline.parent,
                      target=None if args.no_downstream else args.target)


if __name__ == "__main__":
    main()
