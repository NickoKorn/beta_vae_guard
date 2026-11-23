"""Supply the current beta factor to the native Varix loss."""

import math

from autoencodix.utils._annealer import AnnealingScheduler

from .controlvae import controlvae_step, proportional_term


class BetaScheduler:
    def __init__(self, *, total_epochs, schedule, beta_scale, batches_per_epoch,
                 kp=0.002, ki=0.002, gains_per="step", beta_min=0.0,
                 beta_max="schedule", anti_windup="conditional"):
        if schedule == "no-annealing":
            raise ValueError("Varix bypasses the beta scheduler with no-annealing")
        if gains_per not in ("step", "epoch") or batches_per_epoch < 1:
            raise ValueError("The I gain needs a valid unit and at least one batch per epoch")
        if not all(math.isfinite(value) and value >= 0 for value in (beta_scale, beta_min, kp, ki)):
            raise ValueError("Beta scale, beta minimum and PI gains must be finite and nonnegative")
        if beta_max not in ("schedule", "trigger") and (
            not isinstance(beta_max, (int, float)) or not math.isfinite(beta_max) or beta_max < beta_min
        ):
            raise ValueError("beta_max must be 'schedule', 'trigger' or a finite bound >= beta_min")
        if anti_windup not in ("official", "conditional"):
            raise ValueError("anti_windup must be 'official' or 'conditional'")
        self.total_epochs, self.schedule = total_epochs, schedule
        self.beta_scale, self.beta_min, self.beta_max = beta_scale, beta_min, beta_max
        self.batches_per_epoch, self.gains_per = batches_per_epoch, gains_per
        self.kp, self.ki = kp, ki
        self.ki_step = ki / batches_per_epoch if gains_per == "epoch" else ki
        self.anti_windup = anti_windup
        self.annealing = AnnealingScheduler()
        self.mode = "annealing"
        self.beta = self.beta_trigger = self.target_kl = None
        self.integral = self.previous_output = None

    def get_weight(self, *, epoch_current=None, total_epoch=None, func=None, scheduled=False):
        """Return the factor Varix multiplies by its configured beta."""
        if epoch_current is None:
            return 1.0
        if self.beta is not None and not scheduled:
            return self.beta / self.beta_scale if self.beta_scale else 0.0
        return self.annealing.get_weight(
            epoch_current=epoch_current, total_epoch=self.total_epochs, func=self.schedule,
        )

    def start(self, mode, beta, *, measured_kl=None, target_kl=None, kp=None, ki_epoch=None):
        """Freeze beta or initialize PI at the current beta."""
        if mode not in ("freeze", "controlvae"):
            raise ValueError("mode must be 'freeze' or 'controlvae'")
        if mode == "controlvae":
            if target_kl is None or not math.isfinite(target_kl) or target_kl <= 0:
                raise ValueError("ControlVAE needs a positive finite KL target")
            if measured_kl is None or not math.isfinite(measured_kl):
                raise ValueError("ControlVAE needs a finite KL measurement")
            if kp is not None:
                self.kp = kp
            if ki_epoch is not None:
                self.ki, self.gains_per = ki_epoch, "step"
                self.ki_step = ki_epoch / self.batches_per_epoch
            self.target_kl = target_kl
            error = target_kl - measured_kl
            self.integral = beta - proportional_term(error, self.kp) - self.beta_min
            self.previous_output = beta
        else:
            self.target_kl = self.integral = self.previous_output = None
        self.mode, self.beta, self.beta_trigger = mode, beta, beta

    def step(self, measured_kl, epoch):
        """Update beta from the current training batch when PI is active."""
        if self.mode != "controlvae" or not self.beta_scale:
            return
        if self.beta_max == "schedule":
            cap = self.beta_scale * self.get_weight(epoch_current=epoch, scheduled=True)
        elif self.beta_max == "trigger":
            cap = self.beta_trigger
        else:
            cap = self.beta_max
        result = controlvae_step(
            self.target_kl - measured_kl, kp=self.kp, ki=self.ki_step,
            integral=self.integral, previous_output=self.previous_output,
            beta_min=self.beta_min, beta_max=max(self.beta_min, cap), anti_windup=self.anti_windup,
        )
        self.beta, self.integral, self.previous_output = result.beta, result.integral, result.unclamped
