"""Shared KL trigger and beta schedule adapter for AUTOENCODIX VarixLoss.

VarixLoss uses effective_beta = config.beta * get_weight(...). Before activation,
this guard follows AUTOENCODIX's original annealing schedule. Once observing and
triggered by epoch-level validation KL, it keeps beta fixed or delegates updates:

* controlvae: src/core/controlvae_controller.py, positional PI per training batch.
* freeze: constant beta from the trigger; no feedback update or early stop by default.
* early_stop: finish training immediately after the validation trigger.
* dynamicvae: local/complementary/dynamicvae_controller.py, batch or legacy epoch rule.
* nupi: local/complementary/nupi_controller.py, projected multiplier update per batch.
* log: local/complementary/log_pi_controller.py, historical project ablation per epoch.

State (ramp, observing, hold), trigger, measurement filtering, bounds and history stay in this
class. The complementary laws are imported only when selected. The constructor's
historical default remains dynamicvae/epoch; configs/varix.json explicitly selects
controlvae. No complete paper-specific VAE architecture is installed here.

Validation chooses the activation epoch. Signals total/max use mean KL as
before (fed back from the training batches, i.e. with dropout active). The
opt-in held validation signals measure in eval mode and hold the value between
epochs (own feedback adaptation): lucas_max = largest validation 99%-KL order
statistic; valid_max = max_j R_j and valid_total = sum_j R_j of the validation.
step() still logs the actual training KL; batch controllers update beta before
the loss asks get_weight(). stop_on_done=False observes the full horizon.
"""
# ---------------------------------------------------------------------------------------------
# Cases after the hand-over from the annealing (discussion 2026-11-18). beta* = the beta the
# network sustains without full Lucas collapse (ontix ~0.065-0.075: ControlVAE end value of the
# ladder; last-axis study 0.057-0.068). What matters is beta at the hand-over vs. beta*.
#
#  #  method                      beta after the hand-over      start < beta*        start > beta*
#  1  early_stop                  stop, snapshot                stable, maybe         stable, close to
#                                                               undertrained          the collapse
#  2  freeze                      constant                      should hold           collapses later
#                                                                                     (ontix 0/15)
#  3  slow ramp, no cap           rises more slowly             collapses later;      collapses
#                                 (open loop)                   measures beta*
#                                                               quasi-statically
#  4  slow ramp with cap          rises up to beta_cap          holds iff cap <       -
#                                                               beta* (needs beta*)
#  5  beta lowered open loop      step/ramp down                -                     holds iff target <
#                                                                                     beta* (needs beta*)
#  6  ControlVAE from below       falls (to 0), back to beta*   -                     rate recovers, axes
#     (threshold ladder)                                                              do not; ontix k5 5/5
#  7  ControlVAE from above       rises to beta* (sketch        should settle on      -
#     (start before the target)   annealing_prototype.jpg)      beta* without
#                                                               undershoot
#  8  ControlVAE recovery         late start, set point         -                     rate up; do axes
#                                 above the trigger value                             come back?
#
# Only 6 and 7 need no prior knowledge of beta*; 2, 4, 5 are baselines/controls, 3 measures beta*.
# Case 7 needs beta_max="schedule" (logistic-mid as the upper bound) or a number: with
# beta_max="trigger" (configs/varix.json) beta can never rise above the hand-over value.
# Data (src/experiments/varix_case_study.py): 1, 2, 7 from the valid_total_k* states of the
# threshold ladder (on ontix below beta* at k10/k5); 6 = threshold ladder; 3, 4, 5, 8 open.
# weak_signal_lowvar has no fixed beta* (collapse only at epoch 613-989, the net keeps learning).
# ---------------------------------------------------------------------------------------------
from __future__ import annotations

import math
from collections import deque

import numpy as np
from autoencodix.utils._annealer import AnnealingScheduler

from .controlvae_controller import controlvae_step, proportional_term


def _complementary(module, name):
    """Import a complementary controller from local/complementary (not part of the GitHub repo)."""
    import importlib
    try:
        return getattr(importlib.import_module(f"local.complementary.{module}"), name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"form needs local/complementary/{module}.py (complementary approaches are kept local, "
            "not on GitHub); controlvae, freeze and early_stop work without it") from error


# Option names used in configs and run files saved before a rename -> current name.
RENAMED_OPTIONS = {"arm": "guard_observe_start_value", "trip": "control_value",       # renamed 2026-11-18
                   "trip_epochs": "control_value_epochs", "dead": "collapse_value"}


def current_options(options):
    """Guard options with names from older saved configs/runs translated (protocol.json, variant.json)."""
    options = dict(options or {})
    for old, new in RENAMED_OPTIONS.items():
        if old in options:
            if new in options:
                raise ValueError(f"guard options contain both {old!r} and {new!r}")
            options[new] = options.pop(old)
    return options


def summary_value(summary, name):
    """Value from a saved guard_summary/settings dict under its current or its former name."""
    if name in summary:
        return summary[name]
    former = {new: old for old, new in RENAMED_OPTIONS.items()}.get(name)
    return summary[former] if former in summary else summary[name]


# State values saved before 2026-11-18 -> current value (states: ramp -> observing -> hold/done/failed,
# or triggered_unguarded when the guard is disabled).
RENAMED_STATES = {"armed": "observing", "tripped_unguarded": "triggered_unguarded"}


class CollapseGuard:
    # signals whose feedback is the eval-mode validation value, held between epochs
    HELD_SIGNALS = ("lucas_max", "lucas_quantile", "lucas_share", "valid_max", "valid_total")
    # patient-based (per-sample KL, Lucas et al. 2019 Sec. 6.2) signals measured by the trainer
    PATIENT_SIGNALS = ("lucas_max", "lucas_quantile", "lucas_share")

    def __init__(self, *, total_epochs, schedule="logistic-mid", enabled=True, guard_observe_start_value=0.10, control_value=0.05,
                 collapse_value=0.01, control_value_epochs=2, form="dynamicvae", smooth=6, kp=0.01, ki=0.05, beta_min=0.0,
                 max_step=None, beta_scale=1.0, settle_epochs=50, window=100, band=(0.04, 0.06),
                 beta_tolerance=0.05, fail_epochs=20, signal="max", stop_on_done=None, gains_per="step",
                 beta_max="schedule", anti_windup="conditional", nu=0.0, update_per=None,
                 kl_window=1, signal_quantile=0.99):
        if schedule == "no-annealing":
            raise ValueError("Autoencodix bypasses the annealing object for 'no-annealing'")
        if not 0 < collapse_value < control_value < guard_observe_start_value:
            raise ValueError("Need 0 < collapse_value < control_value < guard_observe_start_value")
        self.total_epochs, self.schedule, self.enabled = int(total_epochs), schedule, bool(enabled)
        # guard_observe_start_value: the guard starts observing (ramp -> observing) once the signal was above it;
        # control_value: trigger threshold and set point C; collapse_value: mean-KL threshold per dimension
        self.guard_observe_start_value, self.control_value, self.collapse_value = guard_observe_start_value, control_value, collapse_value
        if form not in ("dynamicvae", "log", "controlvae", "nupi", "freeze", "early_stop"):
            raise ValueError("Unknown form; choose dynamicvae, log, controlvae, nupi, freeze or early_stop")
        if signal not in ("max", "total") + self.HELD_SIGNALS:
            raise ValueError("signal must be max, total, valid_max, valid_total (held validation mean KL), "
                             "lucas_max (largest 99% order statistic of the per-patient KL), lucas_quantile "
                             "(largest signal_quantile order statistic) or lucas_share (largest share of "
                             "patients with KL >= epsilon)")
        if not 0 < signal_quantile <= 1:
            raise ValueError("signal_quantile must be in (0, 1]")
        self.signal_quantile = float(signal_quantile)   # used by signal="lucas_quantile"
        self.signal = signal
        self._validation_feedback = None
        self.stop_on_done = form != "freeze" if stop_on_done is None else bool(stop_on_done)
        if form == "early_stop" and not self.stop_on_done:
            raise ValueError("early_stop requires stop_on_done=True")
        if form in ("freeze", "early_stop"):
            kp = ki = 0.0
        if not isinstance(kl_window, int) or kl_window < 1:
            raise ValueError('kl_window must be a positive integer')
        if kl_window != 1 and form != 'controlvae':
            raise ValueError('kl_window is the optional ControlVAE filter; DynamicVAE uses smooth')
        self.kl_window = kl_window
        self._batch_recent = deque(maxlen=kl_window)
        # Preserve historical epoch-based DynamicVAE runs. New comparisons opt into
        # update_per="batch" explicitly; ControlVAE/nuPI always update per batch.
        self.update_per = update_per or ("epoch" if form in ("dynamicvae", "log", "freeze", "early_stop") else "batch")
        if self.update_per not in ("batch", "epoch") or (
            form in ("controlvae", "nupi") and self.update_per != "batch"
        ) or (form in ("log", "freeze", "early_stop") and self.update_per != "epoch"):
            raise ValueError('incompatible form and update_per')
        self._batch_control = self.update_per == "batch"
        self.controller = None
        if form == "dynamicvae" and self._batch_control:
            DynamicVAEController = _complementary('dynamicvae_controller', 'DynamicVAEController')
            self.controller = DynamicVAEController(target=control_value, kp=kp, ki=ki, smooth=smooth)
        if self.controller is not None and max_step is not None:
            raise ValueError('max_step is only supported by the legacy epoch controllers')
        if gains_per not in ("step", "epoch"):
            raise ValueError("gains_per must be 'step' or 'epoch'")
        # "epoch": ki is given per epoch and divided by the batches per epoch (set by the
        # trainer), so the I action per epoch does not depend on the dataset size
        self.gains_per, self.ki_step = gains_per, ki
        # beta_max of the controlvae form (ControlVAE Algorithm 1 has a fixed beta_max, 1 for
        # language modeling): "schedule" = logistic-mid of the current epoch (moves),
        # "trigger" = beta at the trigger (fixed), or a number = fixed absolute beta
        if not (beta_max in ("schedule", "trigger") or isinstance(beta_max, (int, float))):
            raise ValueError("beta_max must be 'schedule', 'trigger' or a number")
        # anti-windup: "official" = ControlVAE code (Disentangling/P_PID.py): freeze I while the
        # previous UNclamped output was outside [beta_min, beta_max]; "conditional" = that plus
        # integrating when the error drives the output back inside (needed for the moving
        # schedule cap)
        if anti_windup not in ("official", "conditional"):
            raise ValueError("anti_windup must be 'official' or 'conditional'")
        self.beta_max_rule, self.anti_windup = beta_max, anti_windup
        if not 0.0 <= nu < 1.0:
            raise ValueError("nu (EMA coefficient of the nupi form) must be in [0, 1)")
        self.nu, self._xi = float(nu), None          # nupi: EMA of the error (xi_t)
        self.beta_trigger = None
        self.I = None               # integral of the controlvae form
        self._u_prev = None         # previous unclamped controlvae output (anti-windup)
        self._cap = None            # beta_max for the batches of the current epoch
        self._epoch = 0             # epoch the next training batches belong to
        self.steps: list[tuple] = []  # per training batch: epoch, state, KL, e, P, I, beta, clamped
        self.control_value_epochs, self.form, self.kp, self.ki = control_value_epochs, form, kp, ki
        self.beta_min, self.beta_scale = float(beta_min), float(beta_scale)  # beta = beta_scale * factor
        self.max_du = None if max_step is None else math.log(max_step)
        self.settle_epochs, self.window, self.band = settle_epochs, window, band
        self.beta_tolerance, self.fail_epochs = beta_tolerance, fail_epochs
        self._scheduler = AnnealingScheduler()
        self._recent = deque(maxlen=smooth)
        # state:
        #   ramp                 beta = annealing; not observing yet
        #   observing            beta = annealing; signal was > guard_observe_start_value, waiting for <= control_value
        #   hold                 triggered; freeze: beta constant, controlvae: beta regulated
        #   triggered_unguarded  triggered with the guard disabled; beta = annealing
        #   done                 early_stop at the trigger, or KL stable in band (stop_on_done)
        #   failed               max R_j < collapse_value for fail_epochs while in hold (stop_on_done)
        self.state = "ramp"
        self.factor = None          # factor in hold; None means "follow the schedule"
        self.u = self.e_prev = self.beta = None
        self.trigger_epoch = self.stop_epoch = None
        self.first_death_epoch = None
        self._ever_active = None  # dims that were above guard_observe_start_value at least once
        self._below_control_value = self._below_collapse_value = 0
        self.history: list[dict] = []

    def __setstate__(self, state):
        # guards pickled before 2026-11-18 (checkpoint states) use the former attribute and state names;
        # remove together with RENAMED_OPTIONS/RENAMED_STATES once no such checkpoint is loaded any more
        for old, new in {**RENAMED_OPTIONS, "_below_trip": "_below_control_value",
                         "_below_dead": "_below_collapse_value"}.items():
            if old in state and new not in state:
                state[new] = state.pop(old)
        state["state"] = RENAMED_STATES.get(state.get("state"), state.get("state"))
        state.setdefault("signal_quantile", 0.99)       # added 2026-11-18 (signal="lucas_quantile")
        self.__dict__.update(state)

    # --- Autoencodix annealing interface ------------------------------------
    def schedule_factor(self, epoch):
        return self._scheduler.get_weight(epoch_current=epoch, total_epoch=self.total_epochs, func=self.schedule)

    def get_weight(self, *, epoch_current=None, total_epoch=None, func=None):
        """Factor for ``epoch_current``; never changes state."""
        if epoch_current is None:
            return 1.0
        if self.factor is None:
            return self.schedule_factor(epoch_current)
        return self.factor

    # --- control ---------------------------------------------------------------
    def observe(self, epoch, R_j, *, measured_signal=None):
        """Use the validation R_j of ``epoch`` to set the factor of ``epoch + 1``."""
        R_j = np.asarray(R_j, dtype=float)
        max_r, controlled = float(R_j.max()), int(R_j.argmax())
        total = float(R_j.sum())
        if self.signal in self.PATIENT_SIGNALS:
            if measured_signal is None or not math.isfinite(measured_signal) or measured_signal < 0:
                raise ValueError(f"{self.signal} needs a finite nonnegative per-patient validation measure")
            y = self._validation_feedback = float(measured_signal)
        elif self.signal in ("valid_max", "valid_total"):
            y = self._validation_feedback = max_r if self.signal == "valid_max" else total
        else:
            y = max_r if self.signal == "max" else total
        alive = R_j > self.collapse_value
        n_alive = int(alive.sum())
        # second, per-dimension use of the same value: a dimension counts as having died only if
        # its own R_j was above guard_observe_start_value once (initialisation excluded)
        active = R_j > self.guard_observe_start_value
        self._ever_active = active if self._ever_active is None else (self._ever_active | active)
        # a death needs a dimension that was really active before; dims near
        # collapse_value after initialisation do not count
        if self.first_death_epoch is None and (self._ever_active & ~alive).any():
            self.first_death_epoch = epoch
        applied = self.get_weight(epoch_current=epoch)
        self._recent.append(y)
        r_bar = float(np.mean(self._recent))
        e = du = None

        if self.state == "ramp" and y > self.guard_observe_start_value:
            self.state = "observing"
        if self.state == "observing":
            self._below_control_value = self._below_control_value + 1 if y <= self.control_value else 0
            if self._below_control_value >= self.control_value_epochs:
                self.trigger_epoch = epoch
                if self.enabled:
                    self.state = "hold"
                    self.beta = self.beta_scale * applied        # beta(t0): the schedule value
                    self.u = math.log(applied)
                    if self.form == "early_stop":
                        self.beta_trigger = self.beta
                        self.factor = applied
                        self.state, self.stop_epoch = "done", epoch
                    elif self.form == "freeze":
                        self.beta_trigger = self.beta
                    elif self.form == "controlvae":
                        # bumpless start: P(e0) + I + beta_min = beta(t0)
                        e0 = self.control_value - y
                        self.I = self.beta - self._p_term(e0) - self.beta_min
                        self.beta_trigger = self.beta
                        self._u_prev = self.beta
                        self.e_prev = e0
                        # Optional filter ablation: keep all ControlVAE gains and
                        # anti-windup unchanged. Trigger-prefill is our start rule.
                        self._batch_recent.extend([y] * self.kl_window)
                    elif self.form == "nupi":
                        # beta = beta_ref - lambda, lambda(t0) = 0: beta(t0) = schedule beta.
                        # xi starts at the trigger error so the first P step has no kick
                        # (bumpless; adaptation, the paper leaves xi_0 general)
                        self.beta_trigger = self.beta
                        self.I = 0.0
                        self._xi = self.e_prev = self.control_value - y
                    elif self.controller is not None:
                        self.beta_trigger = self.beta
                        self.controller.reset(beta=self.beta, kl=y)
                        self.e_prev = self.control_value - y
                    else:
                        self.e_prev = (self.control_value - r_bar) if self.form == "dynamicvae" else math.log(r_bar / self.control_value)
                else:
                    self.state = "triggered_unguarded"
        if (self.state == "hold" and self.trigger_epoch != epoch
                and not self._batch_control and self.form != "freeze"):
            cap = self.beta_scale * self.schedule_factor(epoch + 1)
            if self.form == "dynamicvae":
                e, du = self._dynamicvae_step(r_bar)
                self.beta = min(max(self.beta + du, self.beta_min), cap)
            else:
                log_pi_step = _complementary('log_pi_controller', 'log_pi_step')
                e, du, self.u, self.beta = log_pi_step(
                    r_bar, target=self.control_value, kp=self.kp, ki=self.ki,
                    error_previous=self.e_prev, log_factor=self.u,
                    beta_scale=self.beta_scale, beta_max=cap, max_delta=self.max_du)
            self.e_prev = e
        if self.state == "hold":
            self.factor = self.beta / self.beta_scale
            self._below_collapse_value = self._below_collapse_value + 1 if max_r < self.collapse_value else 0
            if self._below_collapse_value >= self.fail_epochs and self.stop_on_done:
                self.state, self.stop_epoch = "failed", epoch
        self._epoch = epoch + 1
        self._cap = self.beta_scale * self.schedule_factor(epoch + 1)   # beta_max of the next epoch

        self.history.append(dict(
            epoch=epoch, state=self.state, schedule_factor=self.schedule_factor(epoch), applied_factor=applied,
            next_factor=self.get_weight(epoch_current=epoch + 1), max_R=max_r, total_KL=total, R_bar=r_bar,
            controlled_dim=controlled, n_alive=n_alive, error=e, du=du, integral=self.I))
        if self.state == "hold" and self.stop_on_done and self._stable(epoch):
            self.state, self.stop_epoch = "done", epoch
            self.history[-1]["state"] = "done"
        return self.history[-1]

    # --- controlvae: one PI step per training batch -----------------------------
    def set_batches_per_epoch(self, n):
        """Called by the trainer; converts a per-epoch ki into the per-batch step."""
        self.batches_per_epoch = int(n)
        self.ki_step = self.ki / n if self.gains_per == "epoch" else self.ki
        if self.controller is not None:
            self.controller.ki = self.ki_step

    def _beta_max(self):
        if self.beta_max_rule == "trigger":
            return self.beta_trigger
        if self.beta_max_rule == "schedule":
            return self._cap if self._cap is not None else self.beta
        return float(self.beta_max_rule)

    def _p_term(self, e):
        return proportional_term(e, self.kp)

    def step(self, kl_batch):
        """Feed training KL; batch modes update beta before loss weighting.

        ``kl_batch`` is the KL per latent dimension (mean over the batch) or already its
        sum. The controlled value follows ``signal``: the sum (total KL) or the largest
        dimension (max_j R_j). Called from the training loss before ``get_weight``, so
        the beta of this batch already uses its own KL (ControlVAE Algorithm 1: sample
        KL, then set beta(t)). Logged for every form and state; only batch modes in
        "hold" change beta. DynamicVAE logs smoothed error and delta-P in p_term;
        integral is None because its kernel uses incremental, not positional, I.
        """
        r = np.atleast_1d(np.asarray(kl_batch, dtype=float))
        total = float(r.sum())
        kl = float(r.max()) if self.signal == "max" and r.size > 1 else total
        if self.signal in self.HELD_SIGNALS:
            # Own feedback adaptation: hold the last eval-mode validation
            # measurement between epochs; keep the existing per-batch PI law.
            # The actual training-batch total remains in the separate log column.
            kl = self._validation_feedback if self._validation_feedback is not None else total
        e = p = None
        clamped = False
        if self.controller is not None and self.state == "hold":
            result = self.controller.step(kl, beta_min=self.beta_min, beta_max=self._beta_max())
            self.beta, self.e_prev = result.beta, result.error
            e, p, clamped = result.error, result.p_delta, result.clamped
            self.factor = self.beta / self.beta_scale
        elif self.form == "controlvae" and self.state == "hold":
            cap = self._beta_max()
            self._batch_recent.append(kl)
            e = self.control_value - float(np.mean(self._batch_recent))
            result = controlvae_step(
                e, kp=self.kp, ki=self.ki_step, integral=self.I,
                previous_output=self._u_prev, beta_min=self.beta_min,
                beta_max=cap, anti_windup=self.anti_windup)
            self.beta, self.I, self._u_prev = result.beta, result.integral, result.unclamped
            p, clamped = result.p_term, result.clamped
            self.factor = self.beta / self.beta_scale
        elif self.form == "nupi" and self.state == "hold":
            nupi_step = _complementary('nupi_controller', 'nupi_step')
            e = self.control_value - kl
            result = nupi_step(
                e, kp=self.kp, ki=self.ki_step, nu=self.nu, multiplier=self.I,
                xi_previous=self._xi, beta_ref=self._beta_max(), beta_min=self.beta_min)
            self.beta, self.I, self._xi = result.beta, result.multiplier, result.xi
            p, clamped = result.p_delta, result.clamped
            self.factor = self.beta / self.beta_scale
        beta_now = self.beta_scale * (self.factor if self.factor is not None else self.schedule_factor(self._epoch))
        self.steps.append((self._epoch, self.state, kl, total, e, p, self.I, beta_now, clamped))

    def _dynamicvae_step(self, y):
        """Delegate the historical epoch rule; keep the old method for saved guards."""
        legacy_epoch_step = _complementary('dynamicvae_controller', 'legacy_epoch_step')
        return legacy_epoch_step(
            y, target=self.control_value, kp=self.kp, ki=self.ki, error_previous=self.e_prev,
            beta=self.beta, beta_min=self.beta_min, max_delta=self.max_du)

    def _stable(self, epoch):
        if epoch - self.trigger_epoch < max(self.settle_epochs, self.window):
            return False
        win = self.history[-self.window:]
        mean_r = float(np.mean([h["max_R"] for h in win]))
        f0, f1 = win[0]["applied_factor"], win[-1]["applied_factor"]
        # the controlled (largest) rate must never drop below collapse_value inside the window,
        # and no further dimension may die in it
        never_below_dead = min(h["max_R"] for h in win) >= self.collapse_value
        no_new_death = win[-1]["n_alive"] >= win[0]["n_alive"]
        return (self.band[0] <= mean_r <= self.band[1] and abs(f1 / f0 - 1) < self.beta_tolerance
                and never_below_dead and no_new_death)

    def should_stop(self):
        return self.stop_on_done and self.state in ("done", "failed")

    @property
    def status(self):
        # status (mean-KL criterion, not Lucas; Lucas endpoint: analysis.collapse.summarize):
        #   held                    hold phase without max R_j < collapse_value
        #   recovered_after_dip     dipped below collapse_value, active again later (can still end dead)
        #   collapse_despite_guard  below collapse_value for the last fail_epochs
        #   early_stopped / done    stopped at the trigger / KL stable in band
        #   no_trigger              control_value never reached
        #   unguarded               triggered with the guard disabled
        #   hold_truncated / not_converged  hold too short (< 150 epochs) / no stable end
        if not self.stop_on_done and self.state == "hold":
            # full-length run: judge the whole hold phase
            hold = [h for h in self.history if h["epoch"] > self.trigger_epoch]
            if not hold:
                return "hold_truncated"
            below = [h["max_R"] < self.collapse_value for h in hold]
            if all(below[-self.fail_epochs:]):
                return "collapse_despite_guard"
            return "recovered_after_dip" if any(below) else "held"
        if self.state == "done":
            return "early_stopped" if self.form == "early_stop" else "done"
        if self.state == "failed":
            return "collapse_despite_guard"
        if self.trigger_epoch is None:
            return "no_trigger"
        if self.state == "triggered_unguarded":
            return "unguarded"
        held = self.history[-1]["epoch"] - self.trigger_epoch
        return "hold_truncated" if held < 150 else "not_converged"
