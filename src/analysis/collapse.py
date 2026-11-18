"""Posterior collapse per dimension and in total, after three criteria (analysis only).

Input: the validation history of one run (collapse_history.jsonl, via runs.load_run).
Per epoch and latent dimension j an axis counts as active
  mean      R_j > collapse_value                mean KL over the samples (our dimension criterion)
  lucas     fewer than 1-delta of the samples have KL < epsilon (Lucas et al. 2019; per-sample)
  au        A_j = Var_x(E_q[z_j | x]) > threshold (Burda et al. 2016, Sec. 5.2 "active units";
            here A_j = 2 v_j with v_j = 0.5 Var(mu_j) as logged by the trainer)
Primary endpoint (configs/varix.json -> protocol.collapse.primary_endpoint): lucas, i.e. the
confirmed collapse of the last active axis; mean KL, total KL and au are descriptive only.
The criteria answer different questions: mean and lucas count KL, au counts information
in the posterior means. An axis can carry KL without information (constant mu, sigma^2 != 1),
then it is active after mean/lucas and inactive after au.

Events per dimension: first epoch inactive, final death (start of the inactive stretch that
lasts to the end), number of inactive episodes, recovery (inactive, later active again).
Full collapse: no active axis; confirmed after `confirm` consecutive epochs (protocol default 3).
Thresholds default to configs/varix.json -> protocol.collapse. KL decomposition into
information / offset / variance term: rate.kl_decomposition.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.core.train_varix import ROOT

CRITERIA = ("mean", "lucas", "au")


def _defaults():
    c = json.loads((ROOT / "configs" / "varix.json").read_text())["protocol"]["collapse"]
    return dict(collapse_value=c["dimension_collapse_value_nats"], au_threshold=c["active_units_threshold"],
                lucas_delta=c["lucas_delta"], confirm=c["lucas_confirm_epochs"])


def active_matrices(history, *, collapse_value=None, au_threshold=None, lucas_delta=None):
    """{criterion: bool array (epochs x dims)} plus the epoch numbers; lucas is None if not logged."""
    d = _defaults()
    collapse_value = d["collapse_value"] if collapse_value is None else collapse_value
    au_threshold = d["au_threshold"] if au_threshold is None else au_threshold
    lucas_delta = d["lucas_delta"] if lucas_delta is None else lucas_delta
    R = np.vstack(history["R_j_nats"].to_numpy())
    out = dict(mean=R > collapse_value, au=2 * np.vstack(history["v_j"].to_numpy()) > au_threshold, lucas=None)
    if "lucas" in history and history["lucas"].notna().all():
        frac = np.vstack([np.asarray(x["fractions_below_epsilon"]) for x in history["lucas"]])
        out["lucas"] = frac < 1 - lucas_delta
    epochs = history["epoch_number"].to_numpy() if "epoch_number" in history else history["epoch"].to_numpy() + 1
    return out, epochs


def timeline(history, **thresholds):
    """Per epoch: beta, total and max KL, number of active axes after each criterion."""
    active, epochs = active_matrices(history, **thresholds)
    R = np.vstack(history["R_j_nats"].to_numpy())
    table = pd.DataFrame(dict(epoch_number=epochs, beta=history["effective_beta"].to_numpy(),
                              total_KL=R.sum(1), max_R=R.max(1)))
    for name in CRITERIA:
        table[f"n_active_{name}"] = None if active[name] is None else active[name].sum(1)
    return table


def _runs(mask):
    """(start, end) index pairs of consecutive True stretches."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def dimension_events(active, epochs):
    """One row per dimension: ever active, first inactive, final death, episodes, recovery."""
    rows = []
    for j in range(active.shape[1]):
        a = active[:, j]
        inactive = _runs(~a)
        ever = bool(a.any())
        first_active = int(np.argmax(a)) if ever else None
        after = [(s, e) for s, e in inactive if first_active is not None and s > first_active]
        final = after[-1] if after and after[-1][1] == len(a) - 1 else None
        rows.append(dict(dim=j, ever_active=ever,
                         first_inactive=int(epochs[after[0][0]]) if after else None,
                         final_death=int(epochs[final[0]]) if final else None,
                         inactive_episodes=len(after),
                         recovered=bool(len(after) > (1 if final else 0)),
                         inactive_epochs=int(sum(e - s + 1 for s, e in after)),
                         active_at_end=bool(a[-1])))
    return pd.DataFrame(rows)


def full_collapse(active, epochs, confirm=None):
    """Onset/confirmation of the first stretch with no active axis lasting >= confirm epochs."""
    confirm = _defaults()["confirm"] if confirm is None else confirm
    none = active.sum(1) == 0
    stretches = _runs(none)
    confirmed = [(s, e) for s, e in stretches if e - s + 1 >= confirm]
    first = confirmed[0] if confirmed else None
    return dict(onset=int(epochs[first[0]]) if first else None,
                confirmation=int(epochs[first[0] + confirm - 1]) if first else None,
                epochs_without_active_axis=int(none.sum()), episodes=len(stretches),
                episodes_confirmed=len(confirmed), at_end=bool(none[-1]),
                recovered_after_confirmed=bool(first is not None and not none[-1]))


def summarize(history, *, confirm=None, **thresholds):
    """Collapse summary of one run after every criterion (flat dict for tables).

    ``primary_*`` keys repeat the primary endpoint (Lucas) so that comparisons use one name.
    """
    active, epochs = active_matrices(history, **thresholds)
    out = dict(epochs=len(epochs), first_epoch=int(epochs[0]), last_epoch=int(epochs[-1]))
    for name in CRITERIA:
        if active[name] is None:
            continue
        ev = dimension_events(active[name], epochs)
        fc = full_collapse(active[name], epochs, confirm)
        out.update({f"{name}_active_end": int(active[name][-1].sum()),
                    f"{name}_active_max": int(active[name].sum(1).max()),
                    f"{name}_first_death": None if ev.first_inactive.isna().all() else int(ev.first_inactive.min()),
                    f"{name}_recovered_dims": int(ev.recovered.sum()),
                    **{f"{name}_full_{k}": v for k, v in fc.items()}})
    # axes that carry KL but no information (active after mean, not after au) at the end
    out["kl_without_information_end"] = int((active["mean"][-1] & ~active["au"][-1]).sum())
    if active["lucas"] is not None:
        out.update(primary="lucas", primary_full_collapse=out["lucas_full_onset"] is not None,
                   primary_onset=out["lucas_full_onset"], primary_active_end=out["lucas_active_end"])
    return out
