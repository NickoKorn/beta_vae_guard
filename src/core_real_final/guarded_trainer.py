import json
import math
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from autoencodix.trainers._general_trainer import GeneralTrainer
from torch.utils.data import DataLoader

from .beta_scheduler import BetaScheduler


def kl_per_sample(mu, logvar):
    """KL(q(z|x) || N(0, I)) per sample and latent dimension, float64, shape (samples, dimensions)."""
    mu, logvar = mu.double(), logvar.double()
    posterior = torch.distributions.Normal(mu, (0.5 * logvar).exp())
    prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    return torch.distributions.kl_divergence(posterior, prior)


def lucas_quantile(kl_per_sample, *, delta=0.01):
    """(1 - delta) quantile of the KL per dimension (Q99 for delta = 0.01).

    Dimension j is Lucas-collapsed exactly when this value is below epsilon, so it shows
    how far each axis is from the collapse threshold.
    """
    sample_count = kl_per_sample.shape[0]
    return kl_per_sample.kthvalue(math.ceil((1 - delta) * sample_count), dim=0).values.tolist()


def lucas_collapse(kl_per_sample, *, epsilon=0.01, delta=0.01):
    """Mark axes with KL below epsilon for at least 1 - delta of the samples."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError("delta must be finite and in [0, 1)")
    with torch.no_grad():
        kl = torch.as_tensor(kl_per_sample).detach().double()
        if kl.ndim != 2 or min(kl.shape) < 1 or not torch.isfinite(kl).all() or (kl < 0).any():
            raise ValueError("Expected finite nonnegative KL of shape (samples, dimensions)")
        counts = (kl < epsilon).sum(dim=0)
        fractions = counts.double() / kl.shape[0]
        collapsed = fractions >= 1 - delta
        return {
            "epsilon_nats": float(epsilon),
            "delta": float(delta),
            "samples": kl.shape[0],
            "mean_kl_per_dimension_nats": kl.mean(dim=0).cpu().tolist(),
            "mean_total_kl_nats": kl.sum(dim=1).mean().item(),
            "counts_below_epsilon": counts.cpu().tolist(),
            "fractions_below_epsilon": fractions.cpu().tolist(),
            "collapsed_dimensions": torch.where(collapsed)[0].cpu().tolist(),
            "n_noncollapsed_dimensions": int((~collapsed).sum().item()),
        }


class GuardedTrainer(GeneralTrainer):
    """Feed the beta scheduler using posterior statistics from training batches."""

    def __init__(self, *args, collapse_threshold=0.01, collapse_output, guard_enabled=True,
                 guard_options=None, on_epoch=None, lucas_epsilon=0.01, lucas_delta=0.01,
                 lucas_confirm_epochs=3, handover_kl=None, target_kl=None,
                 collapse_eval_batch_size=2048, lucas_handover=False, handover_k=3, setpoint_k=None,
                 handover_k_step=1, max_restarts=15, control_epochs=1, beta_final_window=20,
                 handover_checkpoint_dir=None, lucas_observe=False, observe_until_end=False,
                 handover_trigger="rate", trigger_epsilon=1.0, trigger_fraction=0.75,
                 monitor_epsilons=None, rate_epsilon=0.01, control_steps=None,
                 single_threshold=True, ring_buffer_size=30, restart_beta_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.collapse_output = Path(collapse_output)
        self.collapse_threshold = collapse_threshold
        self.on_epoch = on_epoch
        # One (epsilon, delta) for every Lucas measurement: validation (_validate_epoch)
        # and training data in eval mode (_eval_collapse).
        if not math.isfinite(lucas_epsilon) or lucas_epsilon <= 0 or not 0 <= lucas_delta < 1:
            raise ValueError("lucas_epsilon must be positive and lucas_delta in [0, 1)")
        self.lucas_epsilon = lucas_epsilon
        self.lucas_delta = lucas_delta
        self.lucas_confirm_epochs = lucas_confirm_epochs
        if type(collapse_eval_batch_size) is not int or collapse_eval_batch_size < 1:
            raise ValueError("collapse_eval_batch_size must be a positive integer")
        self.collapse_eval_batch_size = collapse_eval_batch_size
        options = dict(guard_options or {})
        self.guard_enabled = bool(guard_enabled)
        self.handover_mode = options.get("form", "controlvae")
        self.signal = options.get("signal", "total")
        if self.handover_mode not in ("early_stop", "freeze", "controlvae"):
            raise ValueError("form must be 'early_stop', 'freeze' or 'controlvae'")
        if self.signal not in ("max", "total"):
            raise ValueError("Training-based handover requires signal='max' or signal='total'")
        self.handover_kl = float(options.get("control_value", 0.05) if handover_kl is None else handover_kl)
        self.observation_start_kl = float(options.get("guard_observe_start_value", 0.10))
        self.handover_confirm_epochs = options.get("control_value_epochs", 2)
        if not 0 < self.handover_kl < self.observation_start_kl or not math.isfinite(self.observation_start_kl):
            raise ValueError("Need 0 < handover_kl < observation_start_kl")
        if type(self.handover_confirm_epochs) is not int or self.handover_confirm_epochs < 1:
            raise ValueError("Handover confirmation requires a positive number of epochs")
        self.target_kl = self.handover_kl if target_kl is None else float(target_kl)
        if not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("target_kl must be finite and positive")
        self.handover_epoch = self.stop_epoch = None
        self._observing = False
        self._handover_streak = 0
        self._training_history = []
        # --- Lucas-monitored handover ---
        self.lucas_handover = bool(lucas_handover)
        self.single_threshold = bool(single_threshold)
        if type(ring_buffer_size) is not int or ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be a positive integer")
        if not (math.isfinite(restart_beta_decay) and 0 < restart_beta_decay <= 1):
            raise ValueError("restart_beta_decay must be in (0, 1]")
        self.ring_buffer_size = ring_buffer_size
        self.restart_beta_decay = float(restart_beta_decay)
        self.handover_k = float(handover_k)
        self.setpoint_k = self.handover_k if setpoint_k is None else float(setpoint_k)
        self.handover_k_step = float(handover_k_step)
        self.max_restarts = max_restarts
        self.control_epochs = control_epochs
        self.beta_final_window = beta_final_window
        if self.lucas_handover:
            if not (self.handover_k > 0 and self.setpoint_k > 0 and self.handover_k_step > 0):
                raise ValueError("handover_k, setpoint_k and handover_k_step must be positive")
            for name, value in (("max_restarts", max_restarts), ("control_epochs", control_epochs),
                                ("beta_final_window", beta_final_window)):
                if type(value) is not int or value < (0 if name == "max_restarts" else 1):
                    raise ValueError(f"{name} must be a {'nonnegative' if name == 'max_restarts' else 'positive'} integer")
            if self.signal != "total" or not self.guard_enabled:
                raise ValueError("lucas_handover requires signal='total' and guard_enabled=True")
        if handover_trigger not in ("rate", "lucas"):
            raise ValueError("handover_trigger must be 'rate' or 'lucas'")
        if not (math.isfinite(trigger_epsilon) and trigger_epsilon > 0 and 0 < trigger_fraction <= 1):
            raise ValueError("trigger_epsilon must be positive and trigger_fraction in (0, 1]")
        self.handover_trigger = handover_trigger
        self.trigger_epsilon = float(trigger_epsilon)
        self.trigger_fraction = float(trigger_fraction)
        self.trigger_axes = int(self.trigger_fraction * self._config.latent_dim)
        if self.lucas_handover and handover_trigger == "lucas" and self.trigger_axes < 1:
            raise ValueError("int(trigger_fraction * latent_dim) must be at least 1")
        if not (math.isfinite(rate_epsilon) and rate_epsilon > 0):
            raise ValueError("rate_epsilon must be positive")
        self.rate_epsilon = float(rate_epsilon)
        # A step budget overrides the epoch budget when explicitly supplied.
        if control_steps is not None and (type(control_steps) is not int or control_steps < 1):
            raise ValueError("control_steps must be a positive integer")
        self.control_steps = control_steps
        # extra epsilons whose collapsed axes are logged every epoch (from Q99, no extra cost)
        self.monitor_epsilons = sorted({self.lucas_epsilon, *(float(e) for e in (monitor_epsilons or []))})
        # --- Observation only ---
        self.lucas_observe = bool(lucas_observe)
        self.observe_until_end = bool(observe_until_end)
        if self.lucas_observe and self.lucas_handover:
            raise ValueError("Choose either lucas_observe or lucas_handover")
        self._axis_events = None
        self._full_collapse_streak = 0
        self.full_collapse_first = self.full_collapse_confirmed = None
        self.handover_checkpoint_dir = (Path(handover_checkpoint_dir) if handover_checkpoint_dir is not None
                                        else self.collapse_output / "handover_levels")
        self.phase = "annealing"
        self.restarts = 0
        self.fail_reason = None
        self.skipped_levels = []
        self.gap_handover = self.setpoint_C = self.beta_handover = None
        self._armed_levels = set()
        self._level_states = {}
        self._anneal_ring = []       # ring buffer of recent healthy annealing states (t - 1, t - 2, ...)
        self.beta_scheduler = BetaScheduler(
            total_epochs=self._config.epochs,
            schedule=self._config.anneal_function,
            beta_scale=float(self._config.beta),
            batches_per_epoch=len(self._trainloader),
            **{name: options[name] for name in ("kp", "ki", "gains_per", "beta_min", "beta_max", "anti_windup")
               if name in options},
        )
        self._loss_fn.annealing_scheduler = self.beta_scheduler

    def _train_epoch(self, should_checkpoint, epoch):
        if len(self._trainloader) == 0:
            raise ValueError("Training requires at least one batch")
        kl_sum = None
        sample_count = 0
        batch_betas = []

        @torch.no_grad()
        def feed_guard(_model, _inputs, output):
            nonlocal kl_sum, sample_count
            if not _model.training:
                return
            mu, logvar = output.latent_mean, output.latent_logvar
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            batch_sum = kl.sum(dim=0, dtype=torch.float64)
            kl_sum = batch_sum if kl_sum is None else kl_sum + batch_sum
            sample_count += kl.shape[0]
            if self.signal == "total":
                # ControlVAE sums latent axes, then averages over samples directly from the training batch.
                measured_kl = kl.sum(dim=1).mean().item()
            else:
                measured_kl = kl.mean(dim=0).max().item()
            self.beta_scheduler.step(measured_kl, epoch)
            batch_betas.append(self.beta_scheduler.beta_scale * self.beta_scheduler.get_weight(epoch_current=epoch))

        hook = self._model.register_forward_hook(feed_guard)
        try:
            losses = super()._train_epoch(should_checkpoint=should_checkpoint, epoch=epoch)
        finally:
            hook.remove()
        if sample_count == 0 or kl_sum is None:
            raise RuntimeError("The training forward hook recorded no KL samples")
        training_rates = (kl_sum / sample_count).cpu().tolist()
        strongest = max(range(len(training_rates)), key=training_rates.__getitem__)
        max_rate, total_rate = training_rates[strongest], sum(training_rates)
        measured_kl = total_rate if self.signal == "total" else max_rate
        applied = self.beta_scheduler.get_weight(epoch_current=epoch)
        if self.handover_epoch is None and not (self.lucas_handover or self.lucas_observe):
            self._observing |= measured_kl > self.observation_start_kl
            self._handover_streak = (self._handover_streak + 1
                                     if self._observing and measured_kl <= self.handover_kl else 0)
            if self._handover_streak >= self.handover_confirm_epochs:
                self._hand_over(epoch, applied, measured_kl)
        if self.stop_epoch == epoch:
            state = "done"
        elif self.beta_scheduler.mode != "annealing":
            state = "hold"
        elif self.handover_epoch is not None:
            state = "triggered_unguarded"
        else:
            state = "observing" if self._observing else "ramp"
        self._training_history.append({
            "epoch": epoch,
            "state": state,
            "schedule_factor": self.beta_scheduler.get_weight(epoch_current=epoch, scheduled=True),
            "applied_factor": applied,
            "next_factor": self.beta_scheduler.get_weight(epoch_current=epoch + 1),
            "max_R": max_rate,
            "total_KL": total_rate,
            "controlled_dim": strongest,
            "n_alive": sum(rate > self.collapse_threshold for rate in training_rates),
            "n_dims_mean_above": sum(rate > self.collapse_threshold for rate in training_rates),
            "R_train_nats": total_rate,
            "integral": self.beta_scheduler.integral,
            "train_R_j_nats": training_rates,
            "beta_batch_count": len(batch_betas),
            "beta_batch_last": batch_betas[-1],
            "beta_batch_mean": sum(batch_betas) / len(batch_betas),
            "beta_batch_min": min(batch_betas),
            "beta_batch_max": max(batch_betas),
        })
        return losses

    def _eval_collapse(self, dataset=None):
        """Lucas collapse and eval KL tracking on all training samples in eval mode."""
        dataset = self._trainloader.dataset if dataset is None else dataset
        loader = DataLoader(dataset, batch_size=self.collapse_eval_batch_size,
                            shuffle=False, drop_last=False)
        was_training = self._model.training
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        kl_batches = []
        squared_error = 0.0
        elements = 0
        try:
            self._model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                for _, features, _ in loader:
                    features = features.to(self.device)
                    mu, logvar = self._model.encode(features)
                    reconstructed = self._model.decode(mu)
                    squared_error += (reconstructed.double() - features.double()).square().sum().item()
                    elements += features.numel()
                    kl_batches.append(kl_per_sample(mu, logvar).cpu())
        finally:
            self._model.train(was_training)
        if not kl_batches:
            raise ValueError("Collapse evaluation requires at least one sample")
        kl_samples = torch.cat(kl_batches)
        lucas = lucas_collapse(kl_samples, epsilon=self.lucas_epsilon, delta=self.lucas_delta)
        rates = kl_samples.mean(dim=0).tolist()
        q99 = lucas_quantile(kl_samples, delta=self.lucas_delta)
        return {
            "R_train_eval_nats": sum(rates),
            "mse_train_eval_mu": squared_error / elements,
            "R_j_train_eval_nats": rates,
            "lucas_n_noncollapsed_train_eval": lucas["n_noncollapsed_dimensions"],
            "lucas_share_active_train_eval": [1.0 - f for f in lucas["fractions_below_epsilon"]],
            "lucas_q99_train_eval_nats": q99,
            "lucas_full_collapse_train_eval": lucas["n_noncollapsed_dimensions"] == 0,
            "n_train_eval": kl_samples.shape[0],
            "lucas_collapsed_dims_by_eps": {
                f"{eps:g}": [j for j, q in enumerate(q99) if q < eps] for eps in self.monitor_epsilons
            },
        }

    def _rate_level(self, k):
        """Rate threshold k * d * rate_epsilon in nats."""
        return k * self._config.latent_dim * self.rate_epsilon

    def _control_finished(self, epoch):
        """Hold long enough: an explicit step budget, otherwise control_epochs complete epochs."""
        held_epochs = epoch - self.handover_epoch
        if self.control_steps is not None:
            return held_epochs * len(self._trainloader) >= self.control_steps
        return held_epochs >= self.control_epochs

    def _restart_levels(self):
        """Handover levels of the first attempt and all restarts, in restart order."""
        if self.handover_trigger == "lucas":
            return [self.trigger_axes - i for i in range(self.max_restarts + 1) if self.trigger_axes - i >= 1]
        return [self.handover_k + i * self.handover_k_step for i in range(self.max_restarts + 1)]

    def _collapsed_at(self, observation, epsilon):
        """Axes (epsilon, delta)-collapsed in this epoch: Q99_j < epsilon."""
        return [j for j, q in enumerate(observation["lucas_q99_train_eval_nats"]) if q < epsilon]

    def _level_reached(self, level, observation):
        """True if the handover condition of `level` holds in this epoch."""
        if self.handover_trigger == "lucas":
            return len(self._collapsed_at(observation, self.trigger_epsilon)) >= level
        return observation["R_train_eval_nats"] <= self._rate_level(level)

    def _push_ring_snapshot(self, epoch, observation, alive):
        """Save a healthy annealing epoch into the bounded ring buffer."""
        path = self.handover_checkpoint_dir / f"ring_epoch{epoch}.pt"
        self.save_checkpoint(path)
        self._anneal_ring.append({
            "path": str(path),
            "epoch": epoch,
            "gap": observation["gap"],
            "applied_factor": observation["applied_factor"],
            "R_train": observation["R_train_nats"],
            "R_train_eval": observation["R_train_eval_nats"],
            "alive": alive,
        })
        while len(self._anneal_ring) > self.ring_buffer_size:
            evicted = self._anneal_ring.pop(0)
            Path(evicted["path"]).unlink(missing_ok=True)

    def _select_ring_checkpoint(self, *, require_beta_drop=False):
        """Choose the latest unused healthy checkpoint before the failed attempt (t-1, then t-2, ...)."""
        if not self._anneal_ring:
            return None
        chosen_idx = len(self._anneal_ring) - 1
        chosen = self._anneal_ring[chosen_idx]
        # Remove chosen and any newer entries so each restart steps strictly further back (t-1 -> t-2 -> t-3)
        for skipped in self._anneal_ring[chosen_idx + 1:]:
            self.skipped_levels.append(skipped["epoch"])
            Path(skipped["path"]).unlink(missing_ok=True)
        del self._anneal_ring[chosen_idx:]
        self._level_states[chosen["epoch"]] = chosen
        return chosen

    def _start_control(self, epoch, level, state):
        """Hand over to PI at `level`; lock onto the training KL rate of snapshot t-n."""
        self.gap_handover = state["gap"]
        if self.single_threshold or self.handover_trigger == "lucas":
            setpoint_k = None
            # Match the exact signal fed to the PI controller (training batch KL) so there is no offset
            self.setpoint_C = state["R_train"]
        else:
            setpoint_k = self.setpoint_k + (level - self.handover_k)
            self.setpoint_C = self._rate_level(setpoint_k)
        self.beta_handover = state["applied_factor"]
        self.target_kl = self.setpoint_C
        self.phase = "control"
        self._current_level, self._current_setpoint_k = level, setpoint_k
        self._alive_at_handover = list(state["alive"])
        self._death_epochs = {}
        self._hand_over(epoch, state["applied_factor"], self.setpoint_C)
        if self.handover_mode == "early_stop":
            self.phase = "done"

    def _lucas_observe_epoch_end(self, epoch):
        """Observation only: measure Lucas, log per-axis collapse events, never intervene."""
        observation = self._training_history[-1]
        observation.update(self._eval_collapse())
        r_eval = observation["R_train_eval_nats"]
        observation["gap"] = observation["R_train_nats"] / r_eval if r_eval > 0 else None
        alive = [share > self.lucas_delta for share in observation["lucas_share_active_train_eval"]]
        beta = observation["applied_factor"]
        if self._axis_events is None:
            self._axis_events = [{"axis": j, "first_alive_epoch": None, "first_alive_beta": None,
                                  "collapse_epoch": None, "collapse_beta": None, "revivals": 0}
                                 for j in range(len(alive))]
        for j, is_alive in enumerate(alive):
            event = self._axis_events[j]
            if is_alive and event["first_alive_epoch"] is None:
                event["first_alive_epoch"], event["first_alive_beta"] = epoch, beta
            elif is_alive and event["collapse_epoch"] is not None:
                event["revivals"] += 1
                event["collapse_epoch"] = event["collapse_beta"] = None
            elif not is_alive and event["first_alive_epoch"] is not None and event["collapse_epoch"] is None:
                event["collapse_epoch"], event["collapse_beta"] = epoch, beta
        collapsed = observation["lucas_full_collapse_train_eval"]
        ever_alive = any(event["first_alive_epoch"] is not None for event in self._axis_events)
        self._full_collapse_streak = self._full_collapse_streak + 1 if collapsed and ever_alive else 0
        if self._full_collapse_streak == 1:
            self.full_collapse_first = {"epoch": epoch, "beta": beta}
        if self._full_collapse_streak == self.lucas_confirm_epochs:
            self.full_collapse_confirmed = {"epoch": epoch, "beta": beta}
            if not self.observe_until_end:
                self.phase = "done"
        observation.update({"phase": self.phase, "full_collapse_streak": self._full_collapse_streak})

    def _lucas_observe_summary(self):
        """Summary block of the observation run: beta at each axis collapse and at full collapse."""
        events = self._axis_events or []
        collapsed = sorted((e for e in events if e["collapse_epoch"] is not None), key=lambda e: e["collapse_epoch"])
        return {
            "phase": self.phase,
            "axis_events": events,
            "collapse_order": [e["axis"] for e in collapsed],
            "never_alive_axes": [e["axis"] for e in events if e["first_alive_epoch"] is None],
            "last_axis_collapse": ({"axis": collapsed[-1]["axis"], "epoch": collapsed[-1]["collapse_epoch"],
                                    "beta": collapsed[-1]["collapse_beta"]}
                                   if collapsed and len(collapsed) == sum(e["first_alive_epoch"] is not None for e in events)
                                   else None),
            "full_collapse_first": self.full_collapse_first,
            "full_collapse_confirmed": self.full_collapse_confirmed,
            "confirm_epochs": self.lucas_confirm_epochs,
            "notes": [
                "Plain annealing without handover, PI or early stop; Lucas (eps, delta) on the training data in eval mode after every epoch.",
                f"beta = applied annealing factor of that epoch (times config beta); loss_reduction = {self._config.loss_reduction}.",
                "collapse_epoch = first epoch of the current collapsed stretch; a revival resets it (revivals counted).",
            ],
        }

    def _lucas_epoch_end(self, epoch):
        """Eval measurement and phase logic after a completed epoch."""
        observation = self._training_history[-1]
        observation.update(self._eval_collapse())
        r_eval, collapsed = observation["R_train_eval_nats"], observation["lucas_full_collapse_train_eval"]
        observation["gap"] = observation["R_train_nats"] / r_eval if r_eval > 0 else None
        alive = [share > self.lucas_delta for share in observation["lucas_share_active_train_eval"]]
        attempt = self.restarts
        if self.phase == "control":
            for axis, was_alive in enumerate(self._alive_at_handover):
                if was_alive and not alive[axis] and axis not in self._death_epochs:
                    self._death_epochs[axis] = epoch
            observation.update({
                "n_alive_at_handover": sum(self._alive_at_handover),
                "axes_died_during_hold": sum(self._alive_at_handover) - sum(alive),
                "death_epochs": {str(axis): e for axis, e in sorted(self._death_epochs.items())},
            })

        restart = None
        lookback = False
        levels = self._restart_levels()
        base_level = levels[0]

        if self.single_threshold:
            if self.phase == "annealing":
                if not collapsed:
                    if observation["gap"] is not None:
                        self._push_ring_snapshot(epoch, observation, alive)
                else:
                    chosen = self._select_ring_checkpoint(require_beta_drop=False)
                    if chosen is not None:
                        restart = (chosen["epoch"], chosen)
                        lookback = True
                    else:
                        self.phase, self.fail_reason = "failed", "collapsed_before_handover"
            elif self.phase == "control":
                if collapsed:
                    if self.restarts >= self.max_restarts:
                        self.phase, self.fail_reason = "failed", "max_restarts"
                    else:
                        chosen = self._select_ring_checkpoint(require_beta_drop=True)
                        if chosen is not None:
                            self.restarts += 1
                            restart = (chosen["epoch"], chosen)
                        else:
                            self.phase, self.fail_reason = "failed", "no_level_left"
                elif self._control_finished(epoch):
                    self.phase = "done"
        else:
            if self.phase == "annealing" and collapsed and base_level in self._armed_levels:
                self.phase, self.fail_reason = "failed", "collapsed_before_handover"
            elif self.phase == "annealing":
                for level in levels:
                    if not self._level_reached(level, observation):
                        self._armed_levels.add(level)
                        stale = self._level_states.pop(level, None)
                        if stale is not None:
                            Path(stale["path"]).unlink(missing_ok=True)
                    elif (level in self._armed_levels and level not in self._level_states
                          and not collapsed and observation["gap"] is not None):
                        prefix = "m" if self.handover_trigger == "lucas" else "k"
                        path = self.handover_checkpoint_dir / f"level_{prefix}{level:g}.pt"
                        self.save_checkpoint(path)
                        self._level_states[level] = {
                            "path": str(path), "epoch": epoch, "gap": observation["gap"],
                            "applied_factor": observation["applied_factor"], "R_train": observation["R_train_nats"],
                            "R_train_eval": r_eval, "alive": alive,
                        }
                if base_level in self._level_states:
                    self._start_control(epoch, base_level, self._level_states[base_level])
            elif self.phase == "control":
                if collapsed:
                    while restart is None:
                        self.restarts += 1
                        if self.restarts > self.max_restarts:
                            self.phase, self.fail_reason = "failed", "max_restarts"
                            break
                        if self.restarts >= len(levels):
                            self.phase, self.fail_reason = "failed", "no_level_left"
                            break
                        level = levels[self.restarts]
                        if level in self._level_states:
                            restart = (level, self._level_states[level])
                        else:
                            self.skipped_levels.append(level)
                elif self._control_finished(epoch):
                    self.phase = "done"

        level = getattr(self, "_current_level", base_level)
        observation.update({
            "phase": self.phase,
            "attempt": attempt,
            "restarts": self.restarts,
            "lookback_handover": lookback,
            "trigger": "single_lucas_ring" if self.single_threshold else self.handover_trigger,
            "k": level,
            "R_thr": (None if (self.single_threshold or self.handover_trigger == "lucas")
                      else self._rate_level(level)),
            "n_collapsed_trigger_eps": len(self._collapsed_at(
                observation, self.lucas_epsilon if self.single_threshold else self.trigger_epsilon
            )),
            "setpoint_k": getattr(self, "_current_setpoint_k", None if self.single_threshold else self.setpoint_k),
            "setpoint_C": self.setpoint_C,
            "gap_handover": self.gap_handover,
            "handover_epoch": self.handover_epoch,
            "fail_reason": self.fail_reason,
            "restart_from_level": restart[0] if restart else None,
        })
        return restart

    def _cleanup_level_checkpoints(self):
        """Delete checkpoints that were not used; keep the one of the final attempt."""
        used = getattr(self, "_current_level", None)
        for level, state in self._level_states.items():
            if level != used:
                Path(state["path"]).unlink(missing_ok=True)
                state["deleted"] = True
        for state in self._anneal_ring:
            Path(state["path"]).unlink(missing_ok=True)
        self._anneal_ring.clear()

    def _lucas_handover_summary(self, epochs_recorded):
        """Summary block of the Lucas-monitored handover."""
        control = [entry for entry in self._training_history if entry.get("phase") in ("control", "done")
                   and entry.get("attempt") == self.restarts and self.handover_epoch is not None
                   and entry["epoch"] > self.handover_epoch]
        window = control[-self.beta_final_window:]
        factors = np.array([entry["applied_factor"] for entry in window], dtype=float)
        last = self._training_history[-1] if self._training_history else {}
        if self.single_threshold or self.handover_trigger == "lucas":
            deviation = np.array([entry["R_train_nats"] - self.setpoint_C for entry in window], dtype=float) \
                if self.setpoint_C is not None else np.array([])
        else:
            setpoint_k = getattr(self, "_current_setpoint_k", None)
            deviation = np.array([entry["R_train_eval_nats"] - self._rate_level(setpoint_k) for entry in window],
                                 dtype=float) if setpoint_k is not None else np.array([])
        return {
            "phase": self.phase,
            "fail_reason": self.fail_reason,
            "single_threshold": self.single_threshold,
            "collapse_epsilon": self.lucas_epsilon,
            "rate_epsilon": self.rate_epsilon,
            "control_epochs": self.control_epochs,
            "control_steps": self.control_steps,
            "steps_per_epoch": len(self._trainloader),
            "handover_trigger": "single_lucas_ring" if self.single_threshold else self.handover_trigger,
            "trigger_epsilon": (self.lucas_epsilon if self.single_threshold
                                else (self.trigger_epsilon if self.handover_trigger == "lucas" else None)),
            "trigger_axes": (self._config.latent_dim if self.single_threshold
                             else (self.trigger_axes if self.handover_trigger == "lucas" else None)),
            "levels": [] if self.single_threshold else self._restart_levels(),
            "handover_k": self.handover_k, "setpoint_k": self.setpoint_k, "handover_k_step": self.handover_k_step,
            "k_final": getattr(self, "_current_level", None),
            "setpoint_k_final": getattr(self, "_current_setpoint_k", None),
            "restarts": self.restarts,
            "skipped_levels": self.skipped_levels,
            "level_checkpoints": {f"{k:g}": state for k, state in self._level_states.items()},
            "handover_epoch": self.handover_epoch,
            "beta_handover": self.beta_handover,
            "beta_final_mean": float(factors.mean()) if factors.size else None,
            "beta_final_std": float(factors.std()) if factors.size else None,
            "beta_final_window_epochs": int(factors.size),
            "setpoint_C": self.setpoint_C,
            "gap_handover": self.gap_handover,
            "gap_final": last.get("gap"),
            "R_train_deviation_final_mean": float(deviation.mean()) if deviation.size else None,
            "R_train_deviation_final_std": float(deviation.std()) if deviation.size else None,
            "epochs_total": epochs_recorded,
            "epochs_saved": self.beta_scheduler.total_epochs - epochs_recorded,
            "n_alive_at_handover": sum(getattr(self, "_alive_at_handover", [])) if self.handover_epoch is not None else None,
            "n_alive_final": last.get("lucas_n_noncollapsed_train_eval"),
            "axes_died_during_hold": last.get("axes_died_during_hold"),
            "death_epochs": last.get("death_epochs"),
            "final_lucas_train_eval": {key: last.get(key) for key in (
                "R_train_eval_nats", "R_j_train_eval_nats", "lucas_n_noncollapsed_train_eval",
                "lucas_share_active_train_eval", "lucas_q99_train_eval_nats", "lucas_full_collapse_train_eval")},
            "final_lucas_test": None,
            "notes": [
                "PI-Regler nutzt die direkte Trainings-Batch-KL (1 Forward-Pass pro Schritt wie ControlVAE); Eval-KL wird am Epochenende fuer Lucas-Monitoring getrackt.",
                "beta_handover/beta_final sind beobachtete beta-Werte dieses Trainings, kein Nachweis des maximal moeglichen festen beta.",
                f"loss_reduction = {self._config.loss_reduction}: beta nur innerhalb dieser Skalierung vergleichbar.",
                "Testsatz: der Trainer hat keinen Zugriff; final_lucas_test extern mit _eval_collapse(test_dataset).",
            ],
        }

    def _restart_from(self, level, state):
        """Reload the level checkpoint and hand over there with the snapshot's setpoint."""
        next_epoch = self.load_checkpoint(state["path"])
        self._start_control(next_epoch - 1, level, state)
        return next_epoch

    def _validate_epoch(self, should_checkpoint, epoch):
        """Measure Lucas collapse from the existing validation forward passes."""
        means, logvars = [], []

        def capture_posteriors(_model, _inputs, output):
            means.append(output.latent_mean.detach().cpu())
            logvars.append(output.latent_logvar.detach().cpu())

        hook = self._model.register_forward_hook(capture_posteriors)
        try:
            losses = super()._validate_epoch(should_checkpoint=should_checkpoint, epoch=epoch)
        finally:
            hook.remove()
        if not means:
            raise RuntimeError("The validation forward hook recorded no posteriors")

        mu = torch.cat(means).double()
        kl_samples = kl_per_sample(mu, torch.cat(logvars))
        result = lucas_collapse(kl_samples, epsilon=self.lucas_epsilon, delta=self.lucas_delta)
        self._lucas_zero_streak = (
            getattr(self, "_lucas_zero_streak", 0) + 1
            if result["n_noncollapsed_dimensions"] == 0 else 0
        )
        previous = getattr(self, "_last_lucas", None) or {}
        self._last_lucas = {
            **result,
            "full_collapse_streak": self._lucas_zero_streak,
            "full_collapse_confirmed": self._lucas_zero_streak >= self.lucas_confirm_epochs,
        }
        sample_count, dimension_count = kl_samples.shape
        rates = kl_samples.mean(dim=0).tolist()
        first_below = previous.get("first_below_threshold_epoch", [None] * dimension_count).copy()
        ever_active = previous.get("ever_active_dimensions", [False] * dimension_count).copy()
        for axis, rate in enumerate(rates):
            ever_active[axis] |= rate > self.observation_start_kl
            if ever_active[axis] and rate < self.collapse_threshold and first_below[axis] is None:
                first_below[axis] = epoch

        previous_streaks = previous.get("lucas_dimension_streaks", [0] * dimension_count)
        collapsed = set(result["collapsed_dimensions"])
        dimension_streaks = [count + 1 if axis in collapsed else 0
                             for axis, count in enumerate(previous_streaks)]
        self._last_lucas.update(first_below_threshold_epoch=first_below,
                                ever_active_dimensions=ever_active, lucas_dimension_streaks=dimension_streaks)
        quantiles = {
            f"q{round(100 * q)}": kl_samples.kthvalue(math.ceil(q * sample_count), dim=0).values.tolist()
            for q in (0.50, 0.90, 0.95, 0.99)
        }
        lucas_q99 = lucas_quantile(kl_samples, delta=self.lucas_delta)
        active_shares = [1.0 - fraction for fraction in result["fractions_below_epsilon"]]
        self._validation_record = {
            "split": "valid",
            "samples": sample_count,
            "R_j_nats": rates,
            "v_j": (0.5 * mu.var(dim=0, correction=int(sample_count > 1))).tolist(),
            "threshold_nats": self.collapse_threshold,
            "collapsed_dimensions": [axis for axis, rate in enumerate(rates) if rate < self.collapse_threshold],
            "first_below_threshold_epoch": first_below,
            "ever_active_dimensions": ever_active,
            "lucas": self._last_lucas,
            "lucas_dimension_streaks": dimension_streaks,
            "lucas_full_collapse_streak": self._lucas_zero_streak,
            "lucas_full_collapse_confirmed": self._last_lucas["full_collapse_confirmed"],
            "lucas_q99_nats": lucas_q99,
            "lucas_q99_max": max(lucas_q99),
            "kl_quantiles_nats": quantiles,
            "lucas_share_active": active_shares,
            "lucas_share_active_max": max(active_shares),
            "max_R": max(rates),
            "total_KL": sum(rates),
            "valid_loss": float(losses[0]),
            "recon_loss": float(losses[1]["recon_loss"]),
        }
        return losses

    def _record_epoch(self, epoch, train_parts):
        """Record the training signal and the independent validation diagnostics."""
        observation = self._training_history[-1]
        record = {
            **observation,
            "epoch_number": epoch + 1,
            "split": "train",
            "effective_beta": float(train_parts["effective_beta_factor"]),
            "config_beta": float(self._config.beta),
            "train_recon_loss": float(train_parts["recon_loss"]),
            "train_R_j_nats": observation["train_R_j_nats"],
            "train_max_R": observation["max_R"],
            "train_total_KL": observation["total_KL"],
            "train_controlled_dim": observation["controlled_dim"],
        }
        if self._validset:
            record.update(self._validation_record)
        with (self.collapse_output / "collapse_history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return record

    def _record_ladder_checkpoints(self, record, armed, triggers):
        """Save each threshold at its first downward crossing during annealing."""
        epsilon = self.lucas_epsilon
        dimensions = len(record["train_R_j_nats"])
        steps = (
            ("train_max_k10", "train_max_R", 10 * epsilon),
            ("train_max_k5", "train_max_R", 5 * epsilon),
            ("train_total_k10", "train_total_KL", 10 * dimensions * epsilon),
            ("train_total_k2", "train_total_KL", 2 * dimensions * epsilon),
        )
        reached = []
        for name, signal, threshold in steps:
            if name in triggers:
                continue
            value = record[signal]
            if value > threshold:
                armed.add(name)
            elif name in armed:
                reached.append((name, signal, threshold, value))

        if reached:
            path = Path("states") / f"epoch{record['epoch_number']}.pt"
            self.save_checkpoint(self.collapse_output / path)
            for name, signal, threshold, value in reached:
                triggers[name] = {
                    "step": name, "signal": signal, "threshold": threshold, "value": value,
                    "epoch_number": record["epoch_number"], "beta": record["effective_beta"],
                    "state": str(path),
                }
        index = self.collapse_output / "triggers.json"
        if reached or not index.exists():
            pending = [name for name, _, _ in steps if name not in triggers]
            comparison = {}
            if "train_total_k2" in triggers:
                comparison = {
                    "state": triggers["train_total_k2"]["state"],
                    "handover_kl": 2 * dimensions * epsilon,
                    "early_stop": True,
                    "pi_target_kl": {
                        "hold": 2 * dimensions * epsilon,
                        "reduce_kl": dimensions * epsilon,
                        "increase_kl": 3 * dimensions * epsilon,
                    },
                }
            contents = {"triggers": triggers, "not_triggered": pending, "total_kl_comparison": comparison}
            index.write_text(json.dumps(contents, indent=2) + "\n")
        return bool(reached)

    def _export_posteriors(self, epoch, beta, *, event="checkpoint", attempt=0):
        """Export the posteriors and reconstruction scores used by the analyses."""
        from ..analysis.latent import reconstruction_r2

        path = Path("posteriors") / f"epoch{epoch + 1}_{event}_attempt{attempt}.npz"
        (self.collapse_output / path).parent.mkdir(parents=True, exist_ok=True)
        arrays, scores = {}, {}
        was_training = self._model.training
        devices = [self.device] if self.device.type == "cuda" else []
        self._model.eval()
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model = next(self._model.children())
                for split, dataset in (("train", self._trainset), ("valid", self._validset)):
                    if dataset is None:
                        continue
                    means, logvars = [], []
                    reconstructions = {"mu": [], "sample": [], "mc": []}
                    for batch in dataset.data.split(self._config.batch_size):
                        mu, logvar = model.encode(batch.to(self.device))
                        means.append(mu.cpu().numpy())
                        logvars.append(logvar.cpu().numpy())
                        reconstructions["mu"].append(model.decode(mu).cpu().numpy())
                        draws = [model.decode(model.reparameterize(mu, logvar)) for _ in range(10)]
                        reconstructions["sample"].append(draws[0].cpu().numpy())
                        reconstructions["mc"].append(torch.stack(draws).mean(dim=0).cpu().numpy())
                    arrays[f"{split}_mu"] = np.concatenate(means)
                    arrays[f"{split}_logvar"] = np.concatenate(logvars)
                    arrays[f"{split}_ids"] = np.asarray(dataset.sample_ids).astype(str)
                    inputs = dataset.data.detach().cpu().numpy().reshape(len(dataset), -1)
                    snapshot_mu = torch.from_numpy(arrays[f"{split}_mu"])
                    snapshot_logvar = torch.from_numpy(arrays[f"{split}_logvar"])
                    scores[f"{split}_rate_nats"] = kl_per_sample(snapshot_mu, snapshot_logvar).sum(1).mean().item()
                    for kind, batches in reconstructions.items():
                        decoded = np.concatenate(batches).reshape(len(dataset), -1)
                        if kind == "mu":
                            scores[f"{split}_mse_mu"] = float(np.mean(np.square(decoded.astype(np.float64) - inputs)))
                        for name, value in reconstruction_r2(inputs, decoded).items():
                            key = f"{split}_{name}_{kind}"
                            arrays[key] = value
                            scores[key] = value if math.isfinite(value) else None
            np.savez_compressed(self.collapse_output / path, **arrays)
        finally:
            self._model.train(was_training)
        entry = {"epoch_number": epoch + 1, "beta": beta, "event": event, "attempt": attempt,
                 "path": str(path), **scores}
        with (self.collapse_output / "posterior_index.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def _write_summary(self, history_path):
        """Summarize the epochs written by this run."""
        first_confirmation = None
        epochs_recorded = 0
        last = None
        with history_path.open() as stream:
            for line in stream:
                last = json.loads(line)
                epochs_recorded += 1
                if first_confirmation is None and last.get("lucas_full_collapse_confirmed"):
                    first_confirmation = last["epoch_number"]
        summary = {
            "status": ("early_stopped" if self.stop_epoch is not None else
                       "collapse_despite_guard" if first_confirmation is not None and self.beta_scheduler.beta is not None else
                       "held" if self.beta_scheduler.beta is not None else
                       "unguarded" if self.handover_epoch is not None else "no_trigger"),
            "form": self.handover_mode,
            "signal": self.signal,
            "guard_enabled": self.guard_enabled,
            "stop_on_done": self.handover_mode == "early_stop",
            "epochs_run": len(self._training_history),
            "epochs_recorded": epochs_recorded,
            "total_epochs": self.beta_scheduler.total_epochs,
            "trigger_epoch": self.handover_epoch,
            "stop_epoch": self.stop_epoch,
            "handover_kl": self.handover_kl,
            "target_kl": self.target_kl,
            "guard_observe_start_value": self.observation_start_kl,
            "control_value": self.beta_scheduler.target_kl or self.handover_kl,
            "collapse_value": self.collapse_threshold,
            "config_beta": float(self._config.beta),
            "schedule": self.beta_scheduler.schedule,
            "beta_trigger": (last["effective_beta"] if self.stop_epoch is not None else
                             self.beta_scheduler.beta_trigger),
            "beta_hold": last["effective_beta"] if last and self.handover_epoch is not None else None,
            "kp": self.beta_scheduler.kp,
            "ki": self.beta_scheduler.ki,
            "ki_step": self.beta_scheduler.ki_step,
            "gains_per": self.beta_scheduler.gains_per,
            "beta_max_rule": self.beta_scheduler.beta_max,
            "anti_windup": self.beta_scheduler.anti_windup,
            "feedback_source": "training_batch_mean_KL",
            "lucas": last.get("lucas") if last else None,
            "first_full_collapse_confirmation": first_confirmation,
        }
        if self.lucas_handover:
            summary["lucas_handover"] = self._lucas_handover_summary(epochs_recorded)
        if self.lucas_observe:
            summary["lucas_observe"] = self._lucas_observe_summary()
        (self.collapse_output / "guard_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        triggers_path = self.collapse_output / "triggers.json"
        if triggers_path.exists():
            ladder = json.loads(triggers_path.read_text())
            ladder["confirmation_epoch_number"] = first_confirmation
            ladder["onset_epoch_number"] = (
                first_confirmation - self.lucas_confirm_epochs + 1 if first_confirmation is not None else None
            )
            triggers_path.write_text(json.dumps(ladder, indent=2) + "\n")

    def _hand_over(self, epoch, factor, measured_kl, *, kp=None, ki_epoch=None):
        """Apply the selected action to the current training state."""
        self.handover_epoch = epoch
        if not self.guard_enabled:
            return
        if self.handover_mode == "early_stop":
            self.stop_epoch = epoch
        else:
            self.beta_scheduler.start(
                self.handover_mode, self.beta_scheduler.beta_scale * factor,
                target_kl=self.target_kl, measured_kl=measured_kl, kp=kp, ki_epoch=ki_epoch,
            )

    def train(self, epochs_overwrite=None, *, start_epoch=0, continuation_epochs=None,
              continuation_mode=None, kp=0.002, ki_epoch=None, handover_checkpoint=None,
              ladder=False, export_posteriors=False, continuation_target_kl=None):
        """Train from scratch or continue a restored checkpoint for a fixed budget."""
        if type(start_epoch) is not int or start_epoch < 0:
            raise ValueError("start_epoch must be a nonnegative integer")
        epochs = epochs_overwrite or self._config.epochs
        if epochs < 1:
            raise ValueError("The training budget must be positive")
        if ladder and (start_epoch or self.guard_enabled):
            raise ValueError("Run the threshold ladder from scratch with guard_enabled=False")
        if self.lucas_handover and (start_epoch or ladder):
            raise ValueError("lucas_handover runs from scratch, without ladder or checkpoint continuation")
        if continuation_target_kl is not None and (
            not start_epoch or continuation_mode != "controlvae"
            or not math.isfinite(continuation_target_kl) or continuation_target_kl <= 0
        ):
            raise ValueError("continuation_target_kl must be a positive finite target for a PI continuation")
        history_path = self.collapse_output / "collapse_history.jsonl"
        if history_path.exists():
            raise FileExistsError(f"Choose a separate output directory for this run: {history_path}")
        if start_epoch:
            if not self._training_history or self._training_history[-1]["epoch"] != start_epoch - 1:
                raise ValueError("start_epoch must match the loaded checkpoint")
            if epochs_overwrite is not None:
                raise ValueError("Use continuation_epochs for a checkpoint continuation")
            if type(continuation_epochs) is not int or continuation_epochs < 1:
                raise ValueError("continuation_epochs must be a positive integer")
            if continuation_mode not in ("freeze", "controlvae"):
                raise ValueError("continuation_mode must be 'freeze' or 'controlvae'")
            if continuation_mode == "controlvae" and (
                ki_epoch is None or not math.isfinite(ki_epoch) or ki_epoch <= 0
                or not math.isfinite(kp) or kp < 0
            ):
                raise ValueError("ControlVAE continuation requires a positive ki_epoch and nonnegative kp")

            handover = self._training_history[-1]
            self.handover_mode, self.guard_enabled = continuation_mode, True
            self.stop_epoch = None
            if continuation_target_kl is not None:
                self.target_kl = float(continuation_target_kl)
            measured_kl = handover["total_KL" if self.signal == "total" else "max_R"]
            self._hand_over(start_epoch - 1, handover["applied_factor"], measured_kl, kp=kp, ki_epoch=ki_epoch)
            epochs = start_epoch + continuation_epochs
        elif continuation_epochs is not None or continuation_mode is not None:
            raise ValueError("Load a checkpoint and pass its start_epoch before continuing")

        self.collapse_output.mkdir(parents=True, exist_ok=True)
        armed, triggers = set(), {}
        with self._fabric.autocast():
            epoch = start_epoch
            while epoch < epochs or (self.lucas_handover and self.phase == "control"):
                self._init_buffers()
                should_checkpoint = self._should_checkpoint(epoch) or (
                    continuation_epochs is not None and epoch == epochs - 1
                )
                self._model.train()
                train_loss, train_parts = self._train_epoch(should_checkpoint, epoch)
                self._log_losses(epoch, "train", train_loss, train_parts)
                if self._validset:
                    valid_loss, valid_parts = self._validate_epoch(should_checkpoint, epoch)
                    self._log_losses(epoch, "valid", valid_loss, valid_parts)
                if should_checkpoint:
                    self._store_checkpoint(epoch)
                restart = self._lucas_epoch_end(epoch) if self.lucas_handover else None
                if self.lucas_observe:
                    self._lucas_observe_epoch_end(epoch)
                record = self._record_epoch(epoch, train_parts)
                handover_reached = self.handover_epoch == epoch
                if handover_checkpoint is not None and handover_reached:
                    self.save_checkpoint(handover_checkpoint)
                ladder_reached = ladder and self._record_ladder_checkpoints(record, armed, triggers)
                if export_posteriors:
                    if self.lucas_handover and record["lucas_full_collapse_train_eval"]:
                        event = "pi_collapse" if record["state"] == "hold" else "annealing_collapse"
                    elif self.lucas_handover and self.phase in ("done", "failed"):
                        event = self.phase
                    elif handover_reached or ladder_reached or epoch == epochs - 1 or self.stop_epoch == epoch:
                        event = "checkpoint"
                    else:
                        event = None
                    if event:
                        self._export_posteriors(epoch, record["beta_batch_last"],
                                                event=event, attempt=record.get("attempt", 0))
                if self.on_epoch is not None:
                    self.on_epoch(self, epoch, record)
                if self.stop_epoch == epoch or (
                    (self.lucas_handover or self.lucas_observe) and self.phase in ("done", "failed")
                ):
                    self._finish_early(epoch, already_stored=should_checkpoint)
                    break
                if restart:
                    epoch = self._restart_from(*restart)
                    if export_posteriors:
                        self._export_posteriors(epoch - 1, self.beta_handover * self.beta_scheduler.beta_scale,
                                                event="handover", attempt=self.restarts)
                else:
                    epoch += 1
        if self.lucas_handover:
            if self.phase == "annealing":
                self.fail_reason = "no_handover_within_budget"
            elif self.phase == "control":
                self.fail_reason = "epoch_budget_before_control_epochs"
            self._cleanup_level_checkpoints()
        if continuation_epochs is not None:
            self._config.epochs = epochs
        self._write_summary(history_path)
        self._result.model = next(self._model.children())
        return self._result

    def _finish_early(self, epoch, *, already_stored):
        """Save the last epoch where AUTOENCODIX expects the final result."""
        if not already_stored:
            self._init_buffers()
            self._model.eval()
            with torch.no_grad():
                for split, loader in (("train", self._trainloader), ("valid", self._validloader)):
                    if loader is None:
                        continue
                    for indices, features, sample_ids in loader:
                        output = self._model(self.maskix_hook(features))
                        self._capture_dynamics(output, split, indices, sample_ids)
            self._store_checkpoint(epoch)
        self._config.epochs = epoch + 1

    def save_checkpoint(self, path):
        """Save the state needed to continue after the completed epoch."""
        if not self._training_history:
            raise RuntimeError("A checkpoint requires a completed training epoch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 2,
            "next_epoch": self._training_history[-1]["epoch"] + 1,
            "config": self._config,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "model_training": self._model.training,
            "beta_scheduler": self.beta_scheduler,
            "training_history": self._training_history,
            "handover": {
                "enabled": self.guard_enabled,
                "mode": self.handover_mode,
                "signal": self.signal,
                "handover_kl": self.handover_kl,
                "target_kl": self.target_kl,
                "observation_start_kl": self.observation_start_kl,
                "confirm_epochs": self.handover_confirm_epochs,
                "handover_epoch": self.handover_epoch,
                "stop_epoch": self.stop_epoch,
                "observing": self._observing,
                "streak": self._handover_streak,
            },
            "last_lucas": getattr(self, "_last_lucas", None),
            "lucas_zero_streak": getattr(self, "_lucas_zero_streak", 0),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            },
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "torch_version": str(torch.__version__),
        }
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(state, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path):
        """Restore a trusted checkpoint into a trainer with the same configuration."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        version = state["format_version"]
        if version not in (1, 2):
            raise ValueError("Unsupported checkpoint format")
        current_config = self._config.model_dump() if isinstance(state["config"], dict) else self._config
        if state["config"] != current_config:
            raise ValueError("Trainer configuration differs from the checkpoint")
        rng = state["rng"]
        if rng["cuda"] and len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the checkpoint")
        if self.device.type == "cuda" and not rng["cuda"]:
            raise ValueError("Checkpoint lacks the CUDA RNG state for this trainer")

        if version == 1:
            old_guard = state["guard"]
            handover = {
                "enabled": old_guard.enabled,
                "mode": old_guard.form if old_guard.form in ("early_stop", "freeze", "controlvae") else self.handover_mode,
                "signal": old_guard.signal if old_guard.signal in ("max", "total") else self.signal,
                "handover_kl": state.get("handover_kl", old_guard.control_value),
                "target_kl": state.get("target_kl", self.target_kl),
                "observation_start_kl": old_guard.guard_observe_start_value,
                "confirm_epochs": old_guard.control_value_epochs,
                "handover_epoch": old_guard.trigger_epoch,
                "stop_epoch": old_guard.stop_epoch,
                "observing": old_guard.state in ("observing", "triggered_unguarded", "hold", "done", "failed"),
                "streak": old_guard._below_control_value,
            }
            scheduler = BetaScheduler(
                total_epochs=old_guard.total_epochs, schedule=old_guard.schedule, beta_scale=old_guard.beta_scale,
                batches_per_epoch=len(self._trainloader), kp=old_guard.kp, ki=old_guard.ki,
                gains_per=old_guard.gains_per, beta_min=old_guard.beta_min,
                beta_max=old_guard.beta_max_rule, anti_windup=old_guard.anti_windup,
            )
            if old_guard.state == "hold" and old_guard.form in ("freeze", "controlvae"):
                scheduler.mode = old_guard.form
                scheduler.beta, scheduler.beta_trigger = old_guard.beta, old_guard.beta_trigger
                scheduler.target_kl = old_guard.control_value if old_guard.form == "controlvae" else None
                scheduler.integral, scheduler.previous_output = old_guard.I, old_guard._u_prev
            training_history = old_guard.history
        else:
            handover, scheduler = state["handover"], state["beta_scheduler"]
            training_history = state["training_history"]
        diagnostics = state.get("diagnostics", {})
        self._model.load_state_dict(state["model"], strict=True)
        self._model.train(state["model_training"])
        self._optimizer.load_state_dict(state["optimizer"])
        self.beta_scheduler = scheduler
        self._loss_fn.annealing_scheduler = scheduler
        self._training_history = training_history
        self.guard_enabled, self.handover_mode, self.signal = handover["enabled"], handover["mode"], handover["signal"]
        self.handover_kl, self.target_kl = handover["handover_kl"], handover["target_kl"]
        self.observation_start_kl = handover["observation_start_kl"]
        self.handover_confirm_epochs = handover["confirm_epochs"]
        self.handover_epoch, self.stop_epoch = handover["handover_epoch"], handover["stop_epoch"]
        self._observing, self._handover_streak = handover["observing"], handover["streak"]
        self._last_lucas = state.get("last_lucas", diagnostics.get("_last_lucas"))
        self._lucas_zero_streak = state.get("lucas_zero_streak", diagnostics.get("_lucas_zero_streak", 0))
        torch.use_deterministic_algorithms(state["deterministic"])
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        return state["next_epoch"]