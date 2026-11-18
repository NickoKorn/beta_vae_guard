"""Rate R_j over training, in total and per dimension (analysis only, no training).

All numbers come from a run folder (collapse_history.jsonl, guard_summary.json);
nothing is smoothed except where the name says so.

    rate_run(folder)        R (epochs x dims), beta, schedule, epochs, summary of one run
    dimension_table(run)    per dimension: peak, first epoch below collapse_value, sustained death
                            (``sustain`` epochs in a row below collapse_value), returns, beta at each
    monotonicity(run)       share of epochs in which the largest rate rises, raw and smoothed
    elasticity(run)         slope of log R_j against log beta in each dimension's decay phase
    rate_loss(run)          loss speed -dlog(rate)/dlog(beta), total and per dimension
    lead_time(signals, ...) epochs from a threshold crossing to the full-collapse onset
    kl_decomposition(mu, logvar)  information (Var mu), offset (mean mu^2) and variance term
Figures: src/analysis/plots.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.collapse_guard import summary_value

def rate_run(folder):
    folder = Path(folder).resolve()
    rows = [json.loads(line) for line in (folder / "collapse_history.jsonl").read_text().splitlines() if line]
    summary = json.loads((folder / "guard_summary.json").read_text())
    return dict(name="/".join(folder.parts[-4:]), rows=rows, R=np.array([r["R_j_nats"] for r in rows]),
                beta=np.array([r["effective_beta"] for r in rows]),
                schedule=np.array([r["config_beta"] * r["schedule_factor"] for r in rows]),
                epochs=np.array([r["epoch"] for r in rows]), summary=summary)


def _runs_below(mask):
    """(start, length) of every run of True values."""
    runs, start = [], None
    for i, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def dimension_table(run, collapse_value=None, observe_start=None, sustain=10):
    """Per dimension: peak, first crossing below collapse_value, sustained death and returns.

    A dimension counts only after it was above ``observe_start`` (guard_observe_start_value) once.
    sustained_death_epoch: start of the first stretch of >= ``sustain`` epochs below collapse_value.
    returns: how often it came back above collapse_value after having been below it.
    """
    s = run["summary"]
    dead, arm = collapse_value or summary_value(s, "collapse_value"), observe_start or s.get("guard_observe_start_value", s.get("arm"))
    R, beta = run["R"], run["beta"]
    rows = []
    for j in range(R.shape[1]):
        r = R[:, j]
        active = np.flatnonzero(r > arm)
        start = int(active[0]) if active.size else None
        row = dict(dim=j, peak_epoch=int(r.argmax()), peak_R=float(r.max()), final_R=float(r[-1]),
                   ever_active=start is not None)
        if start is not None:
            below = r[start:] < dead
            stretches = [(start + a, n) for a, n in _runs_below(below)]
            first = stretches[0][0] if stretches else None
            sustained = next((a for a, n in stretches if n >= sustain), None)
            row.update(first_below_epoch=first, beta_first_below=None if first is None else float(beta[first]),
                       sustained_death_epoch=sustained,
                       beta_sustained_death=None if sustained is None else float(beta[sustained]),
                       returns=sum(1 for a, n in stretches if a + n < len(r)))
        rows.append(row)
    return pd.DataFrame(rows)


def monotonicity(run, smooth=6):
    """Share of epochs (from the peak of max R to the trigger, or to total collapse) in which max R rises."""
    s = run["summary"]
    m = run["R"].max(axis=1)
    end = s["trigger_epoch"] if s["guard_enabled"] and s["trigger_epoch"] is not None else len(m)
    collapsed = np.flatnonzero(m[:end] < summary_value(s, "collapse_value"))
    end = int(collapsed[0]) if collapsed.size else end      # stop at total collapse
    peak = int(m[:end].argmax())
    raw = np.diff(m[peak:end])
    smoothed = np.diff(np.convolve(m[peak:end], np.ones(smooth) / smooth, mode="valid"))
    steps = np.abs(np.diff(np.log(m[peak:end])))
    return dict(run=run["name"], phase=f"epochs {peak}-{end - 1}", rises_raw=float(np.mean(raw > 0)),
                rises_smoothed=float(np.mean(smoothed > 0)), smooth_window=smooth,
                median_step_pct=float(np.median(steps) * 100), p90_step_pct=float(np.quantile(steps, .9) * 100))


def elasticity(run, collapse_value=None):
    """Slope d log R_j / d log beta from the dimension's peak to its sustained death (or the trigger).

    Least squares on the raw values; describes how fast a dimension loses rate as beta grows.
    """
    s = run["summary"]
    dead = collapse_value or summary_value(s, "collapse_value")
    table = dimension_table(run)
    end_all = s["trigger_epoch"] if s["guard_enabled"] and s["trigger_epoch"] is not None else len(run["R"])
    rows = []
    for _, d in table.iterrows():
        if not d["ever_active"]:
            continue
        stop = int(d["sustained_death_epoch"]) if pd.notna(d.get("sustained_death_epoch")) else end_all
        sl = slice(int(d["peak_epoch"]), stop)
        r, b = run["R"][sl, int(d["dim"])], run["beta"][sl]
        keep = (r > dead) & (b > 0)
        if keep.sum() < 10:
            rows.append(dict(dim=int(d["dim"]), slope=np.nan, n=int(keep.sum())))
            continue
        slope, intercept = np.polyfit(np.log(b[keep]), np.log(r[keep]), 1)
        fit = slope * np.log(b[keep]) + intercept
        r2 = 1 - np.var(np.log(r[keep]) - fit) / np.var(np.log(r[keep]))
        rows.append(dict(dim=int(d["dim"]), slope=float(slope), r2=float(r2), n=int(keep.sum()),
                         beta_from=float(b[keep][0]), beta_to=float(b[keep][-1])))
    return pd.DataFrame(rows)


def rate_loss(run, smooth=6, floor=3e-5):
    """When is most rate lost?  Loss speed = -d log(rate) / d log beta between epochs.

    Computed on the 6-epoch moving average (as the controller), for the total rate
    sum_j R_j and for every dimension (only while it is above ``floor``). Returns
    (per-epoch frame, summary rows: epoch and beta of the steepest loss, and the epoch
    window in which the middle half of the total rate decline happened).
    """
    s, R, beta = run["summary"], run["R"], run["beta"]
    end = s["trigger_epoch"] if s["guard_enabled"] and s["trigger_epoch"] is not None else len(R)
    kernel = np.ones(smooth) / smooth

    def speed(x):
        xs = np.convolve(x[:end], kernel, mode="valid")
        bs = np.convolve(beta[:end], kernel, mode="valid")
        return -np.diff(np.log(np.maximum(xs, floor))) / np.diff(np.log(bs)), xs

    total_speed, total = speed(R.sum(axis=1))
    epochs = np.arange(len(total_speed)) + smooth          # epoch at the right end of the window
    frame = pd.DataFrame(dict(epoch=epochs, beta=beta[epochs], total_rate=total[1:], loss_speed_total=total_speed))
    peak_from = int(np.argmax(total))                      # only the decline after the peak
    decline = total[peak_from:]
    lost = (decline[0] - decline) / (decline[0] - decline[-1])
    q25, q75 = (peak_from + int(np.argmax(lost >= q)) + smooth - 1 for q in (0.25, 0.75))
    rows = [dict(what="total", peak_speed=float(total_speed[peak_from:].max()),
                 epoch_peak=int(epochs[peak_from + np.argmax(total_speed[peak_from:])]),
                 beta_peak=float(beta[epochs[peak_from + np.argmax(total_speed[peak_from:])]]),
                 middle_half_epochs=f"{q25}-{q75}", middle_half_beta=f"{beta[q25]:.4f}-{beta[q75]:.4f}")]
    for j in range(R.shape[1]):
        sp, xs = speed(R[:, j])
        alive = xs[1:] > 10 * floor
        if alive.sum() < 5:
            continue
        sp = np.where(alive, sp, -np.inf)
        k = int(np.argmax(sp))
        frame[f"loss_speed_dim{j}"] = np.where(alive, sp, np.nan)
        rows.append(dict(what=f"dim {j}", peak_speed=float(sp[k]), epoch_peak=int(epochs[k]), beta_peak=float(beta[epochs[k]])))
    return frame, pd.DataFrame(rows)


def sustained_from(mask, end):
    """First index from which ``mask`` holds without interruption up to ``end`` (inclusive)."""
    m = np.asarray(mask[: end + 1], bool)
    if not m.any() or not m[-1]:
        return None
    breaks = np.flatnonzero(~m)
    return int(breaks[-1] + 1) if breaks.size else 0


def lead_time(signals, onset_index, level, factors=(1.25, 2, 5, 10)):
    """Epochs between a signal's sustained crossing of k*level and the full-collapse onset.

    ``signals``: {name: array over epochs}; ``level``: {name: own collapse level} (0.01 for
    a single axis, d*0.01 for the total KL). Fair comparison of early warning at the same
    safety factor k (results/signal_lead_time: total KL warns earliest in every seed).
    """
    rows = []
    for name, s in signals.items():
        s = np.asarray(s, float)
        for k in factors:
            start = sustained_from(s < k * level[name], onset_index)
            rows.append(dict(signal=name, k=k, threshold=k * level[name],
                             lead=None if start is None else onset_index - start, start_index=start))
    return pd.DataFrame(rows)


def kl_decomposition(mu, logvar):
    """Per dimension: mean KL = 0.5 Var(mu) + 0.5 mean(mu)^2 + 0.5 E[sigma^2 - 1 - log sigma^2].

    Separates information (variance of the means over x) from a constant offset and from
    the posterior variance term; identifies 'KL without information' axes.
    """
    mu, logvar = np.asarray(mu, float), np.asarray(logvar, float)
    var_mu = 0.5 * mu.var(0)
    mean_sq = 0.5 * mu.mean(0) ** 2
    variance = 0.5 * (np.exp(logvar) - 1 - logvar).mean(0)
    return pd.DataFrame(dict(dim=np.arange(mu.shape[1]), kl=var_mu + mean_sq + variance, var_mu_term=var_mu,
                             mean_offset_term=mean_sq, variance_term=variance,
                             sigma2_mean=np.exp(logvar).mean(0)))

