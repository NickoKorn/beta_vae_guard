"""All figures of the analysis (analysis only, no training).

Run diagnostics (from guard_plots):
    draw / plot_run(folder) / LivePlot / compare_runs
Rate (from rate_analysis; input: rate.rate_run(folder)):
    plot_rates, plot_steps, plot_dimensions, plot_guarded, plot_rate_loss, plot_pi_zoom
Downstream: plot_auc(folders)   (auc_curve.csv from downstream.auc_curve)
AUTOENCODIX-native figures of a pipeline: plot_all(pipeline, output)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.rate import dimension_table, rate_loss
from src.core.collapse_guard import summary_value

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES, ACCENT, ALERT = "#2a78d6", "#eb6834", "#d03b3b"
ALERT_RED = ALERT
DIM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # fixed order per run
COMPARE_COLORS = RUN_COLORS

def draw(rows, info, title):
    """Figure from the per-epoch records; ``info`` holds guard_observe_start_value, control_value, collapse_value, config_beta,
    trigger_epoch and stop_epoch (the last two may be None)."""
    h = pd.DataFrame(rows)
    R = np.vstack(h["R_j_nats"].to_numpy())
    epochs = h["epoch_number"].to_numpy()

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[3, 2])
    for ax in (top, bottom):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)

    for j in range(R.shape[1]):
        top.plot(epochs, R[:, j], color=MUTED, linewidth=0.9, alpha=0.6)
    top.plot(epochs, R.max(axis=1), color=SERIES, linewidth=2, label="max$_j$ R$_j$")
    if info.get("signal", "max") == "total":
        label = "Gesamt-KL (Auslösesignal)" if info.get("form") == "freeze" else "Gesamt-KL (Regelgröße)"
        top.plot(epochs, R.sum(axis=1), color=INK, linewidth=1.5, label=label)
    for j, first in enumerate(h["first_below_threshold_epoch"].iloc[-1]):
        if first is not None:
            top.plot(first + 1, R[first, j], "o", color=ALERT, markersize=6, markeredgecolor="#fcfcfb")
    start = info.get("guard_observe_start_value", info.get("arm"))
    for value, name, offset in ((start, "observe start", 1.15), (summary_value(info, "control_value"), "control", 0.87), (summary_value(info, "collapse_value"), "collapse", 1.0)):
        top.axhline(value, color=INK, linestyle=":", linewidth=0.8)
        top.text(epochs[-1], value * offset, f" {name} {value:g}", va="center", fontsize=8, color=INK)
    top.set_yscale("log")
    top.set_ylabel("R$_j$ (Nats, Validierung)", color=INK, fontsize=9)
    top.legend(frameon=False, fontsize=8, loc="lower left")

    bottom.plot(epochs, info["config_beta"] * h["schedule_factor"], color=MUTED, linestyle="--", linewidth=1.2,
                label="Fahrplan logistic-mid")
    bottom.plot(epochs, h["effective_beta"], color=SERIES, linewidth=2, label="β effektiv")
    bottom.set_yscale("log")
    bottom.set_ylabel("β", color=INK, fontsize=9)
    bottom.set_xlabel("Epoche", color=INK, fontsize=9)
    bottom.legend(frameon=False, fontsize=8, loc="lower right")

    for epoch, name, color in ((info.get("trigger_epoch"), "Auslösen", ACCENT),
                               (info.get("stop_epoch"), "Stopp", INK)):
        if epoch is not None:
            for ax in (top, bottom):
                ax.axvline(epoch + 1, color=color, linewidth=1)
            top.text(epoch + 1, top.get_ylim()[1], f" {name}", fontsize=8, color=color, va="top")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_run(folder, show=False):
    """Write guard_diagnostic.pdf; with ``show=True`` also return the figure (notebook)."""
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / "collapse_history.jsonl").read_text().splitlines()]
    summary = json.loads((folder / "guard_summary.json").read_text())
    title = (f"{'/'.join(folder.parts[-4:])}: Status {summary['status']}"
             + (f", β_trigger={summary['beta_trigger']:.3g}" if summary["beta_trigger"] else "")
             + (f", β_hold={summary['beta_hold']:.3g}" if summary["beta_hold"] else ""))
    fig = draw(rows, summary, title)
    out = folder / "guard_diagnostic.pdf"
    fig.savefig(out)
    if show:
        return fig
    plt.close(fig)
    return out


class LivePlot:
    """Callback for GuardedTrainer: redraw the diagnostic figure during training.

    In Jupyter the figure is updated in place (one output area per run); outside a
    notebook nothing is drawn, so scripts and background runs are unaffected.
    """

    def __init__(self, name="", every=5):
        self.name, self.every = name, int(every)
        self.rows, self._handle = [], None
        try:
            from IPython import get_ipython
            self.active = get_ipython() is not None and "IPKernelApp" in get_ipython().config
        except Exception:  # noqa: BLE001 - no IPython available
            self.active = False

    def __call__(self, trainer, epoch, record):
        self.rows.append(record)
        last = trainer.guard.should_stop() or epoch + 1 == trainer.guard.total_epochs
        if not self.active or (epoch % self.every and not last):
            return
        from IPython.display import display

        g = trainer.guard
        info = dict(guard_observe_start_value=g.guard_observe_start_value, control_value=g.control_value, collapse_value=g.collapse_value, signal=g.signal, form=g.form,
                    config_beta=float(trainer._config.beta),
                    trigger_epoch=g.trigger_epoch, stop_epoch=g.stop_epoch)
        fig = draw(self.rows, info, f"{self.name}  Epoche {epoch + 1}, Zustand {g.state}, "
                                    f"β={record['effective_beta']:.3g}, max R={record['max_R']:.3g}")
        if self._handle is None:
            self._handle = display(fig, display_id=True)
        else:
            self._handle.update(fig)
        plt.close(fig)


def compare_runs(folders, labels=None):
    """Runs on one time axis: largest R_j, living dimensions and effective beta.

    Typical use: the same group/split/seed with and without the guard.
    """
    folders = [Path(f).resolve() for f in folders]
    labels = labels or ["/".join(f.parts[-4:-2]) for f in folders]
    if len(folders) > len(COMPARE_COLORS):
        raise ValueError(f"at most {len(COMPARE_COLORS)} runs per comparison")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, height_ratios=[3, 1.4, 2])
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    top, alive, bottom = axes
    summary = None
    for folder, label, color in zip(folders, labels, COMPARE_COLORS):
        rows = [json.loads(line) for line in (folder / "collapse_history.jsonl").read_text().splitlines()]
        summary = json.loads((folder / "guard_summary.json").read_text())
        h = pd.DataFrame(rows)
        R = np.vstack(h["R_j_nats"].to_numpy())
        ep = h["epoch_number"].to_numpy()
        top.plot(ep, R.max(axis=1), color=color, linewidth=1.8, label=label)
        alive.step(ep, (R > summary_value(summary, "collapse_value")).sum(axis=1), where="post", color=color, linewidth=1.8)
        bottom.plot(ep, h["effective_beta"], color=color, linewidth=1.8, label=label)
        if summary["trigger_epoch"] is not None and summary["guard_enabled"]:
            for ax in axes:
                ax.axvline(summary["trigger_epoch"] + 1, color=color, linewidth=0.8, linestyle=":")
    longest = max((pd.read_json(f / "collapse_history.jsonl", lines=True) for f in folders), key=len)
    bottom.plot(longest["epoch_number"], summary["config_beta"] * longest["schedule_factor"], color=MUTED,
                linestyle="--", linewidth=1.1, label="Fahrplan logistic-mid")
    for value, name in ((summary_value(summary, "control_value"), "control"), (summary_value(summary, "collapse_value"), "collapse")):
        top.axhline(value, color=INK, linestyle=":", linewidth=0.8)
        top.text(top.get_xlim()[1], value, f" {name} {value:g}", va="center", fontsize=8, color=INK)
    top.set_yscale("log")
    top.set_ylabel("max$_j$ R$_j$ (Nats)", color=INK, fontsize=9)
    top.legend(frameon=False, fontsize=8, loc="lower left")
    alive.set_ylabel(f"lebende Dim.\n(R$_j$ > {summary_value(summary, 'collapse_value'):g})", color=INK, fontsize=9)
    bottom.set_yscale("log")
    bottom.set_ylabel("β effektiv", color=INK, fontsize=9)
    bottom.set_xlabel("Epoche", color=INK, fontsize=9)
    bottom.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(" vs. ".join(labels) + f"  ({'/'.join(folders[0].parts[-4:-3] + folders[0].parts[-2:])})",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_rates(run, sustain=10):
    """Left: R_j over epochs. Right: R_j over effective beta. o = first below collapse_value, x = sustained death."""
    s, R, beta = run["summary"], run["R"], run["beta"]
    s_control, s_collapse = summary_value(s, "control_value"), summary_value(s, "collapse_value")
    table = dimension_table(run, sustain=sustain)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax in (left, right):
        _style(ax)
        for value, name in ((s.get("guard_observe_start_value", s.get("arm")), "observe start"), (s_control, "control"), (s_collapse, "collapse")):
            ax.axhline(value, color=INK, linestyle=":", linewidth=0.7)
    x_epoch = run["epochs"] + 1
    for j in range(R.shape[1]):
        color = DIM_COLORS[j % len(DIM_COLORS)]
        left.plot(x_epoch, R[:, j], color=color, linewidth=1.1, label=f"Dim {j}")
        right.plot(beta, R[:, j], color=color, linewidth=1.1)
        row = table.iloc[j]
        for key, marker in (("first_below_epoch", "o"), ("sustained_death_epoch", "x")):
            e = row.get(key)
            if e is not None and pd.notna(e):
                e = int(e)
                left.plot(x_epoch[e], R[e, j], marker, color=color, markersize=7, markeredgewidth=1.6)
                right.plot(beta[e], R[e, j], marker, color=color, markersize=7, markeredgewidth=1.6)
    if s["trigger_epoch"] is not None and s["guard_enabled"]:
        left.axvline(s["trigger_epoch"] + 1, color="#eb6834", linewidth=1)
        right.axvline(beta[s["trigger_epoch"]], color="#eb6834", linewidth=1)
    left.set_yscale("log")
    left.set_ylabel("R$_j$ (Nats, Validierung)", color=INK, fontsize=9)
    left.set_xlabel("Epoche", color=INK, fontsize=9)
    right.set_xscale("log")
    right.set_xlabel("β effektiv (log)", color=INK, fontsize=9)
    left.legend(frameon=False, fontsize=7, ncol=4, loc="lower left")
    fig.suptitle(f"{run['name']}: o erstes Unterschreiten von {s_collapse:g}, "
                 f"x dauerhafter Tod (≥ {sustain} Epochen darunter); orange: Auslösen", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_steps(runs):
    """Histogram of |log max R(t+1) - log max R(t)| before the trigger, one line per run."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _style(ax)
    bins = np.linspace(0, 0.6, 31)
    for run, color in zip(runs, DIM_COLORS):
        s, m = run["summary"], run["R"].max(axis=1)
        end = s["trigger_epoch"] if s["guard_enabled"] and s["trigger_epoch"] is not None else len(m)
        peak = int(m[:end].argmax())
        steps = np.abs(np.diff(np.log(m[peak:end])))
        ax.hist(steps, bins=bins, histtype="step", linewidth=1.6, color=color, density=True, label=run["name"])
    ax.set_xlabel("|Δ log max R| von Epoche zu Epoche (0,1 ≈ 10 %)", color=INK, fontsize=9)
    ax.set_ylabel("Dichte", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_dimensions(runs, labels=None, x="epoch"):
    """One panel per latent dimension; every run drawn in each panel (same group/seed expected).

    x = "epoch" or "beta". A dot marks the dimension's peak, a cross its sustained death.
    """
    labels = labels or [r["name"] for r in runs]
    d = runs[0]["R"].shape[1]
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(d / cols)), cols, figsize=(13, 3.0 * np.ceil(d / cols)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    s = runs[0]["summary"]
    s_control, s_collapse = summary_value(s, "control_value"), summary_value(s, "collapse_value")
    for j, ax in enumerate(axes):
        if j >= d:
            ax.set_visible(False)
            continue
        _style(ax)
        for value in (s_control, s_collapse):
            ax.axhline(value, color=INK, linestyle=":", linewidth=0.7)
        for run, label, color in zip(runs, labels, RUN_COLORS):
            xs = run["epochs"] + 1 if x == "epoch" else run["beta"]
            r = run["R"][:, j]
            ax.plot(xs, r, color=color, linewidth=1.2, label=label)
            peak = int(r.argmax())
            ax.plot(xs[peak], r[peak], "o", color=color, markersize=5)
            row = dimension_table(run).iloc[j]
            if pd.notna(row.get("sustained_death_epoch")):
                e = int(row["sustained_death_epoch"])
                ax.plot(xs[e], r[e], "x", color=color, markersize=8, markeredgewidth=1.8)
            t = run["summary"]["trigger_epoch"]
            if t is not None and run["summary"]["guard_enabled"]:
                ax.axvline(xs[t], color=color, linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        if x == "beta":
            ax.set_xscale("log")
        ax.set_title(f"Dim {j}", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.supxlabel("Epoche" if x == "epoch" else "β effektiv (log)", fontsize=9, color=INK)
    fig.supylabel("R$_j$ (Nats, Validierung)", fontsize=9, color=INK)
    fig.suptitle("R$_j$ pro Dimension; o Höchststand, x dauerhafter Tod, gestrichelt Auslösen; "
                 f"Linien bei {s_control:g} und {s_collapse:g}", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_guarded(run, zoom_before=50):
    """All dimensions and the PI phase in one figure.

    Left: whole run; right: from ``zoom_before`` epochs before the trigger to the end.
    Top: R_j of every dimension (one colour each); in the PI phase the controlled rate
    max_j R_j is drawn bold black. Bottom: effective beta against the logistic-mid schedule.
    The PI phase is shaded.
    """
    s, R, beta = run["summary"], run["R"], run["beta"]
    s_control, s_collapse = summary_value(s, "control_value"), summary_value(s, "collapse_value")
    ep = run["epochs"] + 1
    trig = s["trigger_epoch"] if s["guard_enabled"] else None
    start = max(0, (trig or 0) - zoom_before)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex="col", height_ratios=[3, 2],
                             gridspec_kw=dict(width_ratios=[2, 1.2]))
    for col, sl in enumerate((slice(None), slice(start, None))):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            _style(ax)
            if trig is not None:
                ax.axvspan(trig + 1, ep[-1], color="#eb6834", alpha=0.08, linewidth=0)
                ax.axvline(trig + 1, color="#eb6834", linewidth=1)
        for j in range(R.shape[1]):
            top.plot(ep[sl], R[sl, j], color=DIM_COLORS[j % len(DIM_COLORS)], linewidth=1.1,
                     label=f"Dim {j}" if col == 0 else None)
        if trig is not None:
            top.plot(ep[trig:], R[trig:].max(axis=1), color=INK, linewidth=2.2,
                     label="geregelt: max$_j$ R$_j$" if col == 0 else None)
        for value, name in ((s.get("guard_observe_start_value", s.get("arm")), "observe start"), (s_control, "control"), (s_collapse, "collapse")):
            top.axhline(value, color=INK, linestyle=":", linewidth=0.7)
            if col == 1:
                top.text(ep[-1], value, f" {name} {value:g}", va="center", fontsize=8, color=INK)
        top.set_yscale("log")
        bottom.plot(ep[sl], run["schedule"][sl], color=MUTED, linestyle="--", linewidth=1.2,
                    label="Fahrplan logistic-mid" if col == 0 else None)
        bottom.plot(ep[sl], beta[sl], color="#2a78d6", linewidth=2, label="β effektiv" if col == 0 else None)
        bottom.set_yscale("log")
        bottom.set_xlabel("Epoche", color=INK, fontsize=9)
    axes[0, 0].set_ylabel("R$_j$ (Nats, Validierung)", color=INK, fontsize=9)
    axes[1, 0].set_ylabel("β", color=INK, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=5, loc="lower left")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0, 1].set_title(f"Ausschnitt ab Epoche {start + 1}", fontsize=9, color=INK)
    held = "" if trig is None else (f"; PI-Regelung (orange hinterlegt) ab Epoche {trig + 1}, "
                                    f"β_trigger={s['beta_trigger']:.3g}, β_hold={s['beta_hold']:.3g}")
    fig.suptitle(f"{run['name']}: Status {s['status']}{held}", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_auc(folders, labels=None):
    """AUC (valid, from auc_curve.csv) over epochs and over effective beta, one line per run.

    The dashed horizontal line is chance (0.5); a vertical dotted line in each run's
    colour marks its trigger.
    """
    folders = [Path(f).resolve() for f in folders]
    labels = labels or ["/".join(f.parts[-4:-2]) for f in folders]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax in (left, right):
        _style(ax)
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    for folder, label, color in zip(folders, labels, RUN_COLORS):
        curve = pd.read_csv(folder / "auc_curve.csv")
        s = json.loads((folder / "guard_summary.json").read_text())
        style = dict(color=color, linewidth=1.8, marker="o", markersize=3 if len(curve) > 15 else 6, label=label)
        left.plot(curve["epoch"] + 1, curve["auc_valid"], **style)
        right.plot(curve["beta"], curve["auc_valid"], **style)
        if s["guard_enabled"] and s["trigger_epoch"] is not None:
            left.axvline(s["trigger_epoch"] + 1, color=color, linestyle=":", linewidth=1)
            right.axvline(s["beta_trigger"], color=color, linestyle=":", linewidth=1)
    left.set_xlabel("Epoche", color=INK, fontsize=9)
    right.set_xlabel("β effektiv (log)", color=INK, fontsize=9)
    right.set_xscale("log")
    left.set_ylabel("AUC Krebsart (Validierung)", color=INK, fontsize=9)
    left.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Downstream-AUC an den gespeicherten Checkpoints; gestrichelt: Zufall, gepunktet: Auslösen",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_rate_loss(run, smooth=6):
    """Top: total rate sum_j R_j over beta, middle half of its decline shaded.
    Bottom: loss speed -dlog(rate)/dlog(beta) of the total and of every dimension."""
    frame, summary = rate_loss(run, smooth)
    tot = summary.iloc[0]
    q25, q75 = (int(v) for v in tot["middle_half_epochs"].split("-"))
    beta, R = run["beta"], run["R"]
    end = len(frame) + smooth
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, height_ratios=[1.2, 1])
    for ax in (top, bottom):
        _style(ax)
        ax.axvspan(beta[q25], beta[q75], color="#2a78d6", alpha=0.10, linewidth=0)
        ax.set_xscale("log")
    top.plot(beta[:end], R[:end].sum(axis=1), color=INK, linewidth=1.8, label="Gesamtrate Σ R$_j$")
    top.set_yscale("log")
    top.set_ylabel("Rate (Nats)", color=INK, fontsize=9)
    top.text(np.sqrt(beta[q25] * beta[q75]), top.get_ylim()[0] * 3, "mittlere Hälfte\ndes Ratenverlusts",
             ha="center", fontsize=8, color="#1c5cab")
    bottom.plot(frame["beta"], frame["loss_speed_total"], color=INK, linewidth=1.8, label="gesamt")
    for j in range(R.shape[1]):
        col = f"loss_speed_dim{j}"
        if col in frame:
            bottom.plot(frame["beta"], frame[col], color=DIM_COLORS[j % 8], linewidth=0.9, alpha=0.8, label=f"Dim {j}")
    bottom.axhline(1, color=MUTED, linestyle=":", linewidth=0.8)
    bottom.set_yscale("symlog", linthresh=1)
    bottom.set_ylabel("−d log R / d log β\n(1 = R ∝ 1/β)", color=INK, fontsize=9)
    bottom.set_xlabel("β effektiv (log)", color=INK, fontsize=9)
    bottom.legend(frameon=False, fontsize=7, ncol=5, loc="upper left")
    fig.suptitle(f"{run['name']}: wann geht die meiste Rate verloren? (über {smooth} Epochen geglättet)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_pi_zoom(run, before=20):
    """PI phase in detail: controlled rate vs set point, beta, error and Delta beta split
    into its P and I parts (reconstructed from the logged error with the guard's gains)."""
    s = run["summary"]
    s_control, s_collapse = summary_value(s, "control_value"), summary_value(s, "collapse_value")
    t = s["trigger_epoch"]
    if not (s["guard_enabled"] and t is not None):
        raise ValueError("run has no PI phase")
    h = pd.DataFrame(run["rows"])
    sl = slice(max(0, t - before), None)
    ep = h["epoch"].to_numpy()[sl] + 1
    kp, ki = s["kp"], s["ki"]
    e = h["error"].to_numpy(dtype=float)
    sig = lambda x: 1 / (1 + np.exp(x))
    p_part = np.r_[np.nan, kp * (sig(e[1:]) - sig(e[:-1]))]
    i_part = -ki * e
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, height_ratios=[1.4, 1, 1, 1])
    for ax in axes:
        _style(ax)
        ax.axvline(t + 1, color="#eb6834", linewidth=1)
    a0, a1, a2, a3 = axes
    a0.plot(ep, h["max_R"].to_numpy()[sl], color=MUTED, linewidth=0.9, label="max$_j$ R$_j$ (roh)")
    a0.plot(ep, h["R_bar"].to_numpy()[sl], color=INK, linewidth=2, label="geglättet (Regelgröße)")
    a0.axhline(s_control, color="#2a78d6", linestyle="--", linewidth=1, label=f"Sollwert {s_control:g}")
    a0.axhspan(0.04, 0.06, color="#2a78d6", alpha=0.08, linewidth=0)
    a0.axhline(s_collapse, color=ALERT_RED, linestyle=":", linewidth=1, label=f"collapse {s_collapse:g}")
    a0.set_ylabel("Rate (Nats)", fontsize=9, color=INK)
    a0.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    a1.plot(ep, h["effective_beta"].to_numpy()[sl], color="#2a78d6", linewidth=2, label="β effektiv")
    a1.plot(ep, (h["config_beta"] * h["schedule_factor"]).to_numpy()[sl], color=MUTED, linestyle="--", linewidth=1,
            label="Fahrplan")
    a1.set_ylabel("β", fontsize=9, color=INK)
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    a2.plot(ep, e[sl], color=INK, linewidth=1.5)
    a2.axhline(0, color=MUTED, linewidth=0.8)
    a2.set_ylabel("Fehler e = C − y", fontsize=9, color=INK)
    a3.plot(ep, p_part[sl], color="#eb6834", linewidth=1.4, label="P-Anteil")
    a3.plot(ep, i_part[sl], color="#1baf7a", linewidth=1.4, label="I-Anteil")
    a3.plot(ep, h["du"].to_numpy(dtype=float)[sl], color=INK, linewidth=1, linestyle=":", label="Δβ gesamt")
    a3.axhline(0, color=MUTED, linewidth=0.8)
    a3.set_ylabel("Δβ pro Epoche", fontsize=9, color=INK)
    a3.set_xlabel("Epoche", fontsize=9, color=INK)
    a3.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    fig.suptitle(f"{run['name']}: PI-Regelung (K_p={kp:g}, K_i={ki:g}), orange: Auslösen", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_all(pipeline, output, target=None, split="test"):
    """Generate and save available native Autoencodix plots and a status report.

    Missing history or unsupported plots are recorded independently so the
    remaining plots can still be saved. Run after downstream evaluation to
    include the library's ML evaluation plots.
    """
    import warnings
    import matplotlib.pyplot as plt

    folder = Path(output) / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"directory": "plots", "split": split, "color_by": target,
              "steps": {}, "files": []}
    existing_figures = set(plt.get_fignums())

    def attempt(name, action):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                action()
                report["steps"][name] = {"status": "completed"}
            except Exception as error:
                report["steps"][name] = {
                    "status": "failed", "reason": f"{type(error).__name__}: {error}"}
            report["steps"][name]["warnings"] = [str(item.message) for item in caught]

    try:
        with plt.ioff():
            # Native visualize() builds model weights and absolute/relative losses.
            attempt("weights_and_losses", pipeline.visualize)
            for plot_type in ("2D-scatter", "Ridgeline", "Coverage-Correlation"):
                attempt(plot_type, lambda kind=plot_type:
                        pipeline.visualizer.show_latent_space(
                            result=pipeline.result, plot_type=kind, split=split,
                            param=[target] if target is not None else None,
                            n_downsample=10000,
                        ))
            # evaluate() already builds ML_Evaluation figures when labels exist.
            for name in list(pipeline.visualizer.plots):
                attempt("save_" + name, lambda key=name:
                        pipeline.visualizer.save_plots(
                            path=str(folder), which=key, format="png"))
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
    report["files"] = sorted(str(path.relative_to(Path(output)))
                             for path in folder.rglob("*.png"))
    (folder / "plot_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    plt.switch_backend("Agg")
    print(plot_run(sys.argv[1]))


def plot_class_auc(aligned, group, ncols=None):
    """Small multiples: validation AUC of every class over epochs relative to the collapse onset.

    ``aligned``: rows with group, seed, cancer_class, rel (epoch - onset), auc (mu; see
    results/class_timeline). One panel per class, median over seeds with the seed range as
    a band; panels ordered by the AUC 1-5 epochs before the onset (last surviving first).
    The x axis is symlog: coarse checkpoints far before the onset, every epoch near it.
    """
    t = pd.DataFrame(aligned)
    t = t[t.group == group].copy()
    # far from the onset the 100-epoch checkpoints of different seeds sit at different
    # relative epochs: pool them to 50-epoch positions so the median does not zigzag
    t["rel"] = np.where(t.rel < -20, (t.rel / 50).round() * 50, t.rel)
    late = t[t.rel.between(-5, -1)].groupby("cancer_class").auc.median().sort_values(ascending=False)
    classes = list(late.index) + sorted(set(t.cancer_class) - set(late.index))
    ncols = ncols or min(len(classes), 4 if len(classes) <= 8 else 6)
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.ravel()[len(classes):]:
        ax.set_visible(False)
    for ax, cls in zip(axes.ravel(), classes):
        c = t[t.cancer_class == cls].groupby("rel").auc
        x, med, lo, hi = c.median().index, c.median().to_numpy(), c.min().to_numpy(), c.max().to_numpy()
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
        ax.axvline(0, color=INK, linewidth=0.8)
        x = np.asarray(x, float)
        # no measurement in long gaps: solid line only between close points, dotted across gaps
        segments = np.split(np.arange(len(x)), np.flatnonzero(np.diff(x) > 20) + 1)
        for seg in segments:
            ax.fill_between(x[seg], lo[seg], hi[seg], color=SERIES, alpha=0.18, linewidth=0)
            ax.plot(x[seg], med[seg], color=SERIES, linewidth=2, marker="o", markersize=2.5)
        for a, b in zip(segments[:-1], segments[1:]):
            ax.plot([x[a[-1]], x[b[0]]], [med[a[-1]], med[b[0]]], color=SERIES, linewidth=1, linestyle=":")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(f"{cls}  ({late.get(cls, float('nan')):.2f} kurz vor Kollaps)", fontsize=8, color=INK)
    fig.supxlabel("Epochen relativ zum Beginn des Vollkollapses (symlog; 0 = Beginn)", fontsize=9, color=INK)
    fig.supylabel("AUC auf μ (Validierung)", fontsize=9, color=INK)
    fig.suptitle(f"{group}: Klasseninformation je Krebsart, Median über 5 Seeds (Band: Spannweite); "
                 "gestrichelt: Zufall, gepunktet: keine Messung dazwischen", fontsize=10, color=INK)
    fig.tight_layout()
    return fig
