"""Varix trainer with the CollapseGuard (PI lock against total posterior collapse).

The Autoencodix training loop and loss stay unchanged; the trainer only swaps the
annealing object of its loss for a CollapseGuard, feeds it R_j of the validation
split after every epoch and leaves the loop when the guard says stop.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from autoencodix.trainers._general_trainer import GeneralTrainer

from .collapse_guard import CollapseGuard, current_options

def KL_per_dim(mu, logvar, *, return_samples=False):
    """Return (R_j, v_j): mean KL in nats and half sample variance per latent.

    Inputs are tensors of shape (samples, latent dimensions). The diagnostic
    is detached from autograd. v_j uses correction=1, matching torch.var's
    default; it measures variation of posterior means, not posterior variance.
    With return_samples=True, also return the (samples, dimensions) KL matrix.
    """
    with torch.no_grad():
        mu, logvar = mu.double(), logvar.double()
        if mu.ndim != 2 or mu.shape != logvar.shape or mu.shape[0] < 2 or mu.shape[1] < 1:
            raise ValueError("Expected matching (N, d) tensors with N >= 2 and d >= 1")
        if mu.device != logvar.device:
            raise ValueError("mu and logvar must be on the same device")
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise ValueError("Posterior parameters must be finite")
        q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        kl = torch.distributions.kl_divergence(q, p)
        R_j = kl.mean(dim=0)
        v_j = 0.5 * mu.var(dim=0, correction=1)
        if not torch.isfinite(R_j).all() or not torch.isfinite(v_j).all():
            raise ValueError("Nonfinite posterior diagnostics")
        return (R_j, v_j, kl) if return_samples else (R_j, v_j)



def lucas_collapse(kl_per_sample, *, epsilon=0.01, delta=0.01):
    """Per-dimension (epsilon, delta) collapse, Lucas et al. (NeurIPS 2019), Sec. 6.2 (arXiv:1911.02469v1).

    A dimension qualifies when at least 1-delta of the samples have KL < epsilon.
    This diagnostic does not change beta, the loss, or the mean-KL status.
    """
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
        return dict(epsilon_nats=float(epsilon), delta=float(delta), samples=kl.shape[0],
                    counts_below_epsilon=counts.cpu().tolist(),
                    fractions_below_epsilon=fractions.cpu().tolist(),
                    collapsed_dimensions=torch.where(collapsed)[0].cpu().tolist(),
                    n_noncollapsed_dimensions=int((~collapsed).sum().item()))


class GuardedTrainer(GeneralTrainer):
    def __init__(self, *args, collapse_threshold=0.01, collapse_output, guard_enabled=True,
                 guard_options=None, on_epoch=None, lucas_epsilon=0.01, lucas_delta=0.01,
                 lucas_confirm_epochs=3, **kwargs):
        super().__init__(*args, **kwargs)
        if self._validloader is None:
            raise ValueError("Collapse monitoring requires a validation split")
        self.collapse_threshold = collapse_threshold
        self.lucas_epsilon, self.lucas_delta = lucas_epsilon, lucas_delta
        self.lucas_confirm_epochs = int(lucas_confirm_epochs)   # consecutive validations for a confirmed collapse
        self.collapse_output = Path(collapse_output)
        self.guard = CollapseGuard(total_epochs=self._config.epochs, schedule=self._config.anneal_function,
                                   enabled=guard_enabled, collapse_value=collapse_threshold,
                                   beta_scale=float(self._config.beta), **current_options(guard_options))
        # The only intervention: beta of the unchanged VarixLoss now comes from the guard.
        self._loss_fn.annealing_scheduler = self.guard
        self.guard.set_batches_per_epoch(len(self._trainloader))
        # Per-batch measurement: VarixLoss computes the variational loss, then asks
        # get_weight. Wrapping compute_variational_loss hands the guard the total KL of
        # each training batch (sum over dims, mean over samples) right before that.
        original = self._loss_fn.compute_variational_loss

        def measured(*args, **kwargs):
            value = original(*args, **kwargs)
            if self._model.training:
                mu, logvar = kwargs.get("mu"), kwargs.get("logvar")
                with torch.no_grad():
                    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
                    self.guard.step(kl.mean(dim=0).double().cpu().numpy())   # R_j of the batch
            return value

        self._loss_fn.compute_variational_loss = measured
        self._posterior = None
        self.on_epoch = on_epoch
        self._first_below = None

    def _validate_epoch(self, should_checkpoint, epoch):
        means, logvars = [], []

        def capture(module, inputs, output):
            means.append(output.latent_mean.detach().cpu())
            logvars.append(output.latent_logvar.detach().cpu())

        hook = self._model.register_forward_hook(capture)
        try:
            losses = super()._validate_epoch(should_checkpoint, epoch)
        finally:
            hook.remove()
        if not means:
            raise RuntimeError("Collapse monitoring received no validation posteriors")
        self._posterior = (torch.cat(means), torch.cat(logvars), losses)
        return losses

    def _record(self, epoch):
        mu, logvar, (valid_loss, sub_losses) = self._posterior
        R_j, v_j, kl_samples = KL_per_dim(mu, logvar, return_samples=True)
        self._last_lucas = lucas_collapse(kl_samples, epsilon=self.lucas_epsilon, delta=self.lucas_delta)
        if self._first_below is None:
            self._first_below = [None] * len(R_j)
            self._ever_alive = [False] * len(R_j)
        for j, below in enumerate((R_j < self.collapse_threshold).tolist()):
            if R_j[j] > self.guard.guard_observe_start_value:
                self._ever_alive[j] = True    # really active once
            if below and self._ever_alive[j] and self._first_below[j] is None:
                self._first_below[j] = epoch  # only dims that were active before can die
        # Lucas diagnostics on every run (endpoint of the last-axis protocols); only
        # the lucas_max signal feeds the order statistic to the guard.
        rank = math.ceil((1 - self.lucas_delta) * len(kl_samples))
        quantile = kl_samples.kthvalue(rank, dim=0).values
        collapsed = quantile < self.lucas_epsilon
        self._lucas_zero_streak = getattr(self, "_lucas_zero_streak", 0) + 1 if collapsed.all() else 0
        old_streaks = getattr(self, "_lucas_dim_streaks", [0] * len(R_j))
        self._lucas_dim_streaks = [n + 1 if below else 0 for n, below in zip(old_streaks, collapsed.tolist())]
        extra = dict(lucas_q99_nats=quantile.tolist(), lucas_q99_max=float(quantile.max()),
                     lucas_dimension_streaks=list(self._lucas_dim_streaks),
                     lucas_full_collapse_streak=self._lucas_zero_streak,
                     lucas_full_collapse_confirmed=self._lucas_zero_streak >= getattr(self, "lucas_confirm_epochs", 3))
        # patient-based quantities per axis, logged in every run; they feed the lucas_* guard signals
        n = len(kl_samples)
        share_active = (kl_samples >= self.lucas_epsilon).double().mean(dim=0)   # share of patients with KL >= eps
        quantiles = {q: kl_samples.kthvalue(max(1, math.ceil(q * n)), dim=0).values for q in (0.5, 0.9, 0.95, 0.99)}
        extra.update(lucas_share_active=share_active.tolist(), lucas_share_active_max=float(share_active.max()),
                     kl_quantiles_nats={f"q{round(q * 100)}": v.tolist() for q, v in quantiles.items()})
        signal = self.guard.signal
        if signal == "lucas_max":
            guard = self.guard.observe(epoch, R_j.numpy(), measured_signal=float(quantile.max()))
        elif signal == "lucas_quantile":
            q = kl_samples.kthvalue(max(1, math.ceil(self.guard.signal_quantile * n)), dim=0).values
            guard = self.guard.observe(epoch, R_j.numpy(), measured_signal=float(q.max()))
        elif signal == "lucas_share":
            guard = self.guard.observe(epoch, R_j.numpy(), measured_signal=float(share_active.max()))
        else:
            guard = self.guard.observe(epoch, R_j.numpy())
        record = {
            "epoch": epoch, "epoch_number": epoch + 1, "split": "valid", "samples": len(mu),
            "R_j_nats": R_j.tolist(), "v_j": v_j.tolist(), "threshold_nats": self.collapse_threshold,
            "collapsed_dimensions": torch.where(R_j < self.collapse_threshold)[0].tolist(),
            "lucas": self._last_lucas,
            "first_below_threshold_epoch": self._first_below,
            "effective_beta": float(sub_losses["effective_beta_factor"]),
            "config_beta": float(self._config.beta),
            "recon_loss": float(sub_losses.get("recon_loss", float("nan"))),
            "valid_loss": float(valid_loss),
            **guard,
            **extra,
        }
        with (self.collapse_output / "collapse_history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=True) + "\n")
        return record

    def train(self, epochs_overwrite=None, *, start_epoch=0):
        """Train up to an absolute epoch limit; continuation keeps the epoch index.

        ``start_epoch`` is only for a trainer whose complete state was restored.
        The default preserves the existing training behavior.
        """
        epochs = epochs_overwrite or self._config.epochs
        if not isinstance(start_epoch, int) or not 0 <= start_epoch < epochs:
            raise ValueError("Need 0 <= start_epoch < the absolute epoch limit")
        if start_epoch and (not self.guard.history or self.guard.history[-1]["epoch"] != start_epoch - 1):
            raise ValueError("Continuation requires guard history through start_epoch - 1")
        with self._fabric.autocast():
            for epoch in range(start_epoch, epochs):
                self._init_buffers()
                should_checkpoint = self._should_checkpoint(epoch)
                self._model.train()
                epoch_loss, epoch_sub_losses = self._train_epoch(should_checkpoint=should_checkpoint, epoch=epoch)
                self._log_losses(epoch, "train", epoch_loss, epoch_sub_losses)
                valid_loss, valid_sub_losses = self._validate_epoch(should_checkpoint=should_checkpoint, epoch=epoch)
                self._log_losses(epoch, "valid", valid_loss, valid_sub_losses)
                if should_checkpoint:
                    self._store_checkpoint(epoch)
                record = self._record(epoch)
                if self.on_epoch is not None:
                    self.on_epoch(self, epoch, record)   # e.g. LivePlot in a notebook
                if self.guard.should_stop():
                    self._finish_early(epoch, should_checkpoint)
                    break
        self._write_summary()
        self._result.model = next(self._model.children())
        return self._result

    def _finish_early(self, epoch, already_stored):
        """Make an early stop look like a normal last epoch to Autoencodix.

        Autoencodix' evaluator reads train/valid latents at ``config.epochs - 1``.
        If the stop epoch was no checkpoint, train and valid are encoded once more
        (eval mode, no update) and stored; then ``config.epochs`` is set to the
        epochs actually run. The planned length stays in guard_summary.json.
        """
        if not already_stored:
            self._init_buffers()
            self._model.eval()
            with torch.no_grad():
                for split, loader in (("train", self._trainloader), ("valid", self._validloader)):
                    for indices, features, sample_ids in loader:
                        output = self._model(self.maskix_hook(features))
                        self._capture_dynamics(output, split, indices, sample_ids)
            self._store_checkpoint(epoch)
        self._config.epochs = epoch + 1

    def _write_summary(self):
        g, beta = self.guard, float(self._config.beta)
        by_epoch = {h["epoch"]: h for h in g.history}

        def beta_at(epoch):
            return None if epoch is None else beta * by_epoch[epoch]["applied_factor"]

        deaths = {j: e for j, e in enumerate(self._first_below or []) if e is not None}
        summary = dict(
            status=g.status, guard_enabled=g.enabled, epochs_run=len(g.history),
            first_death_epoch=g.first_death_epoch, beta_first_death=beta_at(g.first_death_epoch),
            trigger_epoch=g.trigger_epoch, beta_trigger=beta_at(g.trigger_epoch),
            stop_epoch=g.stop_epoch,
            beta_hold=beta * g.history[-1]["applied_factor"] if g.trigger_epoch is not None and g.enabled else None,
            death_order=[dict(dim=j, epoch=e, beta=beta_at(e)) for j, e in sorted(deaths.items(), key=lambda x: x[1])],
            never_active_dims=[j for j, a in enumerate(getattr(self, "_ever_alive", [])) if not a],
            config_beta=beta, schedule=g.schedule, total_epochs=g.total_epochs,
            guard_observe_start_value=g.guard_observe_start_value, control_value=g.control_value, collapse_value=g.collapse_value, kp=g.kp, ki=g.ki,
            form=g.form, signal=g.signal, stop_on_done=g.stop_on_done,
            update_per=g.update_per, smooth=g._recent.maxlen,
            kl_window=g.kl_window,
            gains_per=g.gains_per, ki_step=g.ki_step, batches_per_epoch=getattr(g, "batches_per_epoch", None),
            beta_max_rule=g.beta_max_rule,
            anti_windup="dynamicvae_algorithm1" if g.controller is not None else g.anti_windup,
            beta_max_trigger=g.beta_trigger, nu=g.nu,
            lucas=getattr(self, "_last_lucas", None),
            feedback_source={"lucas_max": "held_validation_lucas_q99_max",
                             "lucas_quantile": f"held_validation_lucas_q{round(g.signal_quantile * 100)}_max",
                             "lucas_share": "held_validation_max_share_of_patients_with_kl_ge_epsilon",
                             "valid_max": "held_validation_max_Rj",
                             "valid_total": "held_validation_total_KL"}.get(g.signal, "training_batch_mean_KL"),
        )
        (self.collapse_output / "guard_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if g.steps:
            import pandas as pd
            # kl_batch: the controlled value (total KL or max_j R_j), kl_total: sum of R_j
            steps = pd.DataFrame(g.steps, columns=["epoch", "state", "kl_batch", "kl_total", "error", "p_term",
                                                   "integral", "beta", "clamped"])
            steps.insert(0, "step", range(len(steps)))
            # Explicitly distinguish raw logged KL from the controller's feedback
            # when ControlVAE or DynamicVAE uses a measurement filter.
            steps['kl_feedback'] = g.control_value - steps['error']
            steps.to_parquet(self.collapse_output / "steps.parquet", index=False)
