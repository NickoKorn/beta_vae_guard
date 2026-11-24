"""Descriptive trajectories of the native annealing up to the posterior collapse (analysis only).

    epoch_table(folder)        one row per validation epoch of one annealing (collapse_history.jsonl)
    annealing_tables(roots)    all annealings below the given ladder folders, aligned to the collapse
    median_trajectory(table)   median and quartiles per group and epoch before the collapse
    rate_correlations(table)   Spearman correlation of every metric with the total and last-axis rate

Alignment: epochs_before_onset = onset epoch - epoch number, with the onset of the confirmed full
Lucas collapse (triggers.json, eps = delta = 0.01, three validations). All quantities are measured
on the validation split in eval mode. Metrics per epoch:
    total_rate        sum_j R_j (nats)                 last_axis_rate   max_j R_j
    mean_rate         sum_j R_j / d                    beta             effective beta of the loss
    lucas_active      dimensions not (eps, delta)-collapsed (Lucas et al. 2019)
    mean_active       dimensions with R_j >= 0.01      burda_active     2 * v_j > 0.01 (Burda et al. 2016)
    share_active_max  largest share of patients with KL_j(x) >= eps (1 - fraction below eps)
    q99_max           largest 99 % quantile of KL_j(x) over the dimensions
    valid_recon       validation reconstruction loss (AUTOENCODIX MSE, mean reduction)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 0.01
METRICS = ["total_rate", "last_axis_rate", "mean_rate", "beta", "lucas_active", "mean_active", "burda_active",
           "share_active_max", "q99_max", "valid_recon"]


def epoch_table(folder):
    """Per-epoch metrics of one annealing folder (with collapse_history.jsonl and triggers.json)."""
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / "collapse_history.jsonl").read_text().splitlines()]
    triggers = folder / "triggers.json"
    onset = json.loads(triggers.read_text()).get("onset_epoch_number") if triggers.exists() else None
    table = []
    for r in rows:
        R = np.asarray(r["R_j_nats"], dtype=float)
        v = np.asarray(r["v_j"], dtype=float)
        below = np.asarray(r["lucas"]["fractions_below_epsilon"], dtype=float)
        table.append(dict(
            epoch_number=r["epoch_number"], total_rate=R.sum(), last_axis_rate=R.max(), mean_rate=R.mean(),
            beta=r["effective_beta"], lucas_active=r["lucas"]["n_noncollapsed_dimensions"],
            mean_active=int((R >= EPSILON).sum()), burda_active=int((2 * v > EPSILON).sum()),
            share_active_max=float((1 - below).max()), q99_max=r.get("lucas_q99_max", np.nan),
            valid_recon=r.get("recon_loss", np.nan)))
    table = pd.DataFrame(table)
    table["onset_epoch"] = onset
    table["epochs_before_onset"] = (onset - table.epoch_number) if onset is not None else np.nan
    return table


def annealing_tables(roots):
    """All annealings below the ladder folders: runs/[split<s>/<variant>/]<group>/seed<k>/annealing."""
    frames = []
    for root in map(Path, roots):
        for done in sorted(root.glob("runs/**/annealing/annealing_complete.json")):
            folder = done.parent
            info = json.loads(done.read_text())
            parts = folder.relative_to(root / "runs").parts      # (...)/group/seed<k>/annealing
            frame = epoch_table(folder)
            frame.insert(0, "seed", info.get("seed", int(parts[-2][4:])))
            frame.insert(0, "split", info.get("split_seed", 2))
            frame.insert(0, "variant", info.get("variant", "base"))
            frame.insert(0, "group", parts[-3])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def median_trajectory(table, metrics=METRICS, by=("group",), window=(0, 300)):
    """Long table: per group, epoch before the onset and metric the median, quartiles and run count
    (runs without a confirmed collapse are dropped)."""
    aligned = table.dropna(subset=["epochs_before_onset"])
    aligned = aligned[aligned.epochs_before_onset.between(*window)]
    long = aligned.melt(id_vars=[*by, "epochs_before_onset"], value_vars=list(metrics), var_name="metric")
    grouped = long.groupby([*by, "epochs_before_onset", "metric"]).value
    return pd.DataFrame({"median": grouped.median(), "q25": grouped.quantile(0.25),
                         "q75": grouped.quantile(0.75), "runs": grouped.count()}).reset_index()


def rate_correlations(table, metrics=METRICS, rates=("total_rate", "last_axis_rate"), by=("group",)):
    """Spearman correlation of each metric with the rates: pooled over all epochs of a group, and the
    median of the per-run correlations (within-run: does the metric move with the rate over time?)."""
    rows = []
    run_keys = [k for k in ("group", "variant", "split", "seed") if k in table]
    for keys, part in table.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            for rate in rates:
                if metric == rate:
                    continue
                pooled = part[[metric, rate]].corr(method="spearman").iloc[0, 1]
                within = [run[[metric, rate]].corr(method="spearman").iloc[0, 1]
                          for _, run in part.groupby(run_keys) if run[metric].nunique() > 1]
                rows.append(dict(zip(by, keys), metric=metric, rate=rate, spearman_pooled=pooled,
                                 spearman_within_median=np.nanmedian(within) if within else np.nan,
                                 runs=len(within)))
    return pd.DataFrame(rows)
