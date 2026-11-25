import argparse
import json
from pathlib import Path

import numpy as np

from src.core_real_final.guarded_trainer import kl_per_sample, lucas_collapse

def reconstruction_r2(x, recon):
    x, recon = np.asarray(x, dtype=np.float64), np.asarray(recon, dtype=np.float64)
    if x.shape != recon.shape or x.ndim != 2:
        raise ValueError("x and recon must be matching samples-by-features matrices")
    sum_squared_error = (x - recon) ** 2
    total_sum_of_squares_per_sample = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    total_sum_of_squares_per_feature = ((x - x.mean(0)) ** 2).sum(0)
    ok_s, ok_f = total_sum_of_squares_per_sample > 0, total_sum_of_squares_per_feature > 0
    return dict(r2_sample=float(np.mean(1 - sum_squared_error.sum(1)[ok_s] / total_sum_of_squares_per_sample[ok_s])),
                r2_feature=float(np.mean(1 - sum_squared_error.sum(0)[ok_f] / total_sum_of_squares_per_feature[ok_f])),
                r2_weighted=float(1 - sum_squared_error.sum() / total_sum_of_squares_per_feature.sum()))


def pca_baseline(data, k):

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import get_scorer

    pca = PCA(n_components=k).fit(np.vstack([data[s]["x"] for s in ("train", "valid", "test")]))
    z_train, z_valid = (pca.transform(data[s]["x"]) for s in ("train", "valid"))
    classifier = LogisticRegression(max_iter=1000).fit(z_train, data["train"]["y"])
    return dict(k=k, explained_variance=float(pca.explained_variance_ratio_.sum()),
                auroc_ovo=float(get_scorer("roc_auc_ovo")(classifier, z_valid, data["valid"]["y"])),
                **reconstruction_r2(data["valid"]["x"], pca.inverse_transform(z_valid)))


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

    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    kl_samples = kl_per_sample(mu_t, torch.as_tensor(logvar))
    R_j = kl_samples.mean(dim=0)
    v_j = 0.5 * mu_t.var(dim=0, correction=int(len(mu) > 1))
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
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def quality_over_time(branch, original, sample_ids, *, ks=(5, 20), epsilon=0.01, delta=0.01):
    """TC, coverage and ZADU preservation of the validation posterior means at every exported epoch.
    """
    import pandas as pd
    import torch
    from autoencodix.visualize.visualize import Visualizer

    branch, original = Path(branch), np.asarray(original, dtype=float)
    order = {str(sid): i for i, sid in enumerate(sample_ids)}
    rows = []
    for line in (branch / "posterior_index.jsonl").read_text().splitlines():
        entry = json.loads(line)
        data = np.load(branch / entry["path"], allow_pickle=True)
        ids = [str(sid) for sid in data["valid_ids"]]
        x = original[[order[sid] for sid in ids]]               # inputs in the order of the posteriors
        mu, logvar = data["valid_mu"].astype(float), data["valid_logvar"].astype(float)
        kl = kl_per_sample(torch.as_tensor(mu), torch.as_tensor(logvar))
        active = [j for j in range(mu.shape[1])
                  if j not in lucas_collapse(kl, epsilon=epsilon, delta=delta)["collapsed_dimensions"]]
        # Varix sets mu < 1e-7 to exactly 0: a high share means many tied coordinates, which also
        # affects every distance-based score below (LCMC, steadiness/cohesiveness, Spearman)
        row = dict(epoch_number=entry["epoch_number"], beta=entry.get("beta"), active_axes=len(active),
                   clamped_mu_share=float((mu == 0).mean()))
        for label, columns in (("all", list(range(mu.shape[1]))), ("active", active)):
            frame = pd.DataFrame(mu[:, columns])
            with np.errstate(all="ignore"):
                row[f"tc_{label}"] = (float(Visualizer._total_correlation(frame))
                                      if len(columns) >= 2 else float("nan"))
                row[f"coverage_{label}"] = float(Visualizer._coverage_calc(frame)) if columns else float("nan")
        try:
            report = measure_preservation(x, mu, ks=ks)
        except ValueError as error:                 # e.g. a fully collapsed (constant) embedding
            row["preservation"] = f"undefined: {error}"
            rows.append(row)
            continue
        row["preservation"] = "ok"
        for entry_ in report["local"]:
            row[f"lcmc_k{entry_['k']}"] = entry_["scores"]["lcmc"]
        row["spearman_global"] = report["global"][0]["scores"]["spearman_rho"]
        for entry_ in report["clustering"]:
            row[f"steadiness_k{entry_['k']}"] = entry_["scores"]["steadiness"]
            row[f"cohesiveness_k{entry_['k']}"] = entry_["scores"]["cohesiveness"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch_number").reset_index(drop=True)


def diagrams(points, dims=(0, 1, 2), n=500):
    """Persistent homology H0/H1/H2 via farthest-point subsampling and GUDHI RipsPersistence."""
    from gudhi.sklearn.rips_persistence import RipsPersistence
    from gudhi.subsampling import choose_n_farthest_points

    points = np.asarray(points, dtype=float)
    if n and len(points) > n:
        points = np.asarray(choose_n_farthest_points(points=points, nb_points=n, starting_point=0))
    out = RipsPersistence(homology_dimensions=list(dims)).fit_transform([points])[0]
    return dict(zip(dims, out))


def n_long_bars(bars, min_ratio=0.25, max_bars=10):
    """Estimate Betti number: infinite bars plus finite bars before the largest length gap."""
    from gudhi.representations import DiagramSelector

    bars = np.asarray(bars).reshape(-1, 2)
    finite = DiagramSelector(use=True, point_type="finite").fit_transform([bars])[0]
    p = np.sort(finite[:, 1] - finite[:, 0])[::-1][:max_bars]
    long = 0
    if len(p) == 1 or (len(p) > 1 and p[0] > 0):
        k = 1 if len(p) == 1 else int(np.argmin(p[1:] / p[:-1])) + 1
        long = min(k, int((p >= min_ratio * p[0]).sum()))
    return len(bars) - len(finite) + long


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
