"""ControlVAE PI update, extracted from CollapseGuard without changing its timing.

Source: Shao et al., ControlVAE (ICML 2020), Eq. 6 / Algorithm 1.
"""
from dataclasses import dataclass
import math


def proportional_term(error, kp):
    return kp / (1.0 + math.exp(min(error, 50.0)))


@dataclass(frozen=True)
class ControlVAEStep:
    beta: float
    integral: float
    unclamped: float
    p_term: float
    clamped: bool


def controlvae_step(error, *, kp, ki, integral, previous_output,
                    beta_min, beta_max, anti_windup):
    p = proportional_term(error, kp)
    inside = beta_min <= previous_output <= beta_max
    back = anti_windup == "conditional" and (
        (previous_output > beta_max and error > 0)
        or (previous_output < beta_min and error < 0))
    if inside or back:
        integral -= ki * error
    output = p + integral + beta_min
    beta = min(max(output, beta_min), beta_max)
    return ControlVAEStep(beta, integral, output, p, beta != output)
