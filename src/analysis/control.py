"""Controller side: beta trajectories, plant estimate, gain derivation, stability (analysis only).

beta trajectory of a controlled run (input: per-epoch arrays):
    settling_time(beta, tol)        epochs until beta stays within tol of its final value
    tracking(signal, target)        relative deviation from the set point (tail median, CV)
    seed_spread(values)             median, CV and range of a quantity over seeds
    steps_to_result(...)            gradient steps from the start of training to a stop epoch
Plant and gains:
    cadence, fit_ramp               first-order plant y(t+1) = c + p y(t) + q beta(t+1) from an
                                    annealing ramp (exploratory; ramps are not independent
                                    excitations, regressors are correlated)
    fit_plant(signals, column, end) the same fit on any signal column up to an epoch
    critical_ki(p, b, kp)           ki for a critically damped closed loop (pole placement)
Local stability of the ControlVAE timing (docs: local/docs/controlvae_theory.md):
    local_matrix, local_conditions, robust_gain_condition, rate_margin
Conditional model results, not certificates for neural-network training.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.collapse_guard import summary_value

def settling_time(beta, tol=0.05, tail=100):
    """Epochs until beta stays within ``tol`` (relative) of the median of its last ``tail`` epochs."""
    beta = np.asarray(beta, float)
    ref = np.median(beta[-tail:])
    outside = np.flatnonzero(np.abs(beta - ref) > tol * max(abs(ref), 1e-12))
    return 0 if outside.size == 0 else int(outside[-1] + 1)


def tracking(signal, target, tail=100):
    """Deviation of the controlled signal from the set point over the last ``tail`` epochs."""
    s = np.asarray(signal, float)[-tail:]
    return dict(rel_error_median=float(np.median(s) / target - 1), cv=float(s.std() / s.mean()) if s.mean() else None,
                below_target_share=float((s < target).mean()))


def seed_spread(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), cv=float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() else None,
                min=float(v.min()), max=float(v.max()))


def steps_to_result(stop_epoch, batches_per_epoch):
    """Gradient steps up to (and including) ``stop_epoch``; compare methods in steps, not epochs."""
    return int(stop_epoch) * int(batches_per_epoch)


def fit_plant(signals, column, end, window=40):
    """y(t+1) = c + p y(t) + q beta(t+1) on the ``window`` epochs up to ``end``; b = -q."""
    w = signals.loc[end - window:end]
    y0, y1, b1 = w[column].to_numpy()[:-1], w[column].to_numpy()[1:], w["beta"].to_numpy()[1:]
    X = np.column_stack([np.ones_like(y0), y0, b1])
    coef, *_ = np.linalg.lstsq(X, y1, rcond=None)
    fit = X @ coef
    r2 = 1 - ((y1 - fit) ** 2).sum() / ((y1 - y1.mean()) ** 2).sum()
    p, b = float(coef[1]), float(-coef[2])
    return dict(p=p, b=b, r2=float(r2), plausible=bool(0 < p < 1 and b > 0))


def critical_ki(p, b, kp=0.002):
    """Pole placement: ki (per epoch) with a double closed-loop pole (critical damping).

    Closed loop z^2 - T z + D, T = 1 + p - b(k + ki), D = p - b k, k = kp/4. T^2 = 4D gives
    ki* = (1 + p - 2 sqrt(D)) / b - k, double pole sqrt(D), 2 % settling ~ ln 0.02 / ln sqrt(D).
    """
    k = kp / 4
    D = p - b * k
    if not (0 < p < 1 and b > 0 and D > 0):
        return dict(ki_epoch=None, settle_epochs=None, stability_limit=None)
    ki = (1 + p - 2 * math.sqrt(D)) / b - k
    return dict(ki_epoch=float(ki), double_pole=math.sqrt(D), settle_epochs=float(math.log(0.02) / math.log(math.sqrt(D))),
                stability_limit=float((2 * (1 + p) - 2 * b * k) / b))

def cadence(a_epoch, batches):
    """Same first-order time constant: p_batch**batches = p_epoch."""
    return np.expm1(np.log1p(a_epoch) / batches)


def fit_ramp(folder):
    """Fit R_next = intercept + p*R + q*beta_next before guard activation.

    Fixed exploratory range [C/2, 4C]. The ramp is not an independent excitation;
    correlated regressors can give optimistic fit scores. No automatic tuning.
    """
    summary = json.loads((folder / 'guard_summary.json').read_text())
    h = pd.read_json(folder / 'collapse_history.jsonl', lines=True)
    c, trigger = summary_value(summary, 'control_value'), summary['trigger_epoch']
    # Select adjacent original epochs, not pairs across gaps in the rate band.
    previous, current = h.iloc[:-1].reset_index(drop=True), h.iloc[1:].reset_index(drop=True)
    use = ((current.epoch <= trigger) & previous.total_KL.between(c/2, 4*c)
           & current.total_KL.between(c/2, 4*c))
    x = np.column_stack([np.ones(use.sum()), previous.total_KL[use], current.effective_beta[use]])
    y = current.total_KL[use].to_numpy()
    intercept, pole, q = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ np.array([intercept, pole, q])
    valid = bool(0 < pole < 1 and q < 0)
    return dict(n_pairs=len(y), pole_epoch=float(pole), q=float(q),
                r2=float(1 - np.var(residual)/np.var(y)),
                design_condition=float(np.linalg.cond(x)),
                a_epoch=float(1/pole-1) if valid else None,
                slope_g=float(q/(1-pole)) if valid else None,
                fit_has_expected_signs=valid, batches=summary['batches_per_epoch'])


def local_matrix(pole, sensitivity, kp, ki):
    """State [R_t-C, I_(t-1)-I*]; r_(t+1)=p*r_t-b*delta_beta_t.

    sensitivity = -dR_equilibrium/dbeta > 0; b=(1-p)*sensitivity.
    Current KL updates I and beta BEFORE the corresponding model update.
    Valid only for kl_window=1, constant C, and inactive saturation/anti-windup.
    """
    if not all(math.isfinite(x) for x in (pole, sensitivity, kp, ki)):
        raise ValueError('parameters must be finite')
    if not (0 <= pole < 1 and sensitivity > 0 and kp >= 0 and ki > 0):
        raise ValueError('need 0<=pole<1, sensitivity>0, kp>=0, ki>0')
    b, k = (1-pole)*sensitivity, kp/4
    return np.array([[pole-b*(k+ki), -b], [ki, 1.]])


def local_conditions(pole, sensitivity, kp, ki):
    """Exact quadratic Jury inequalities for local_matrix; all must be > 0."""
    matrix = local_matrix(pole, sensitivity, kp, ki)
    trace, det = float(np.trace(matrix)), float(np.linalg.det(matrix))
    b = (1-pole)*sensitivity
    margins = (1-trace+det, 1+trace+det, 1-det)
    return dict(jury_1=margins[0], jury_2=margins[1], jury_3=margins[2],
                stable=bool(all(v > 0 for v in margins)),
                gain_limit=2*(1+pole)/b, gain_used=kp/2+ki)


def robust_gain_condition(pole_lower, sensitivity_upper, kp, ki):
    """Sufficient for EACH frozen model in a stated parameter interval.

    p in [pole_lower,1), G in (0,sensitivity_upper]. Does NOT establish stability
    for arbitrary switching between those matrices or for unbounded drift.
    Bounds must be justified externally, not chosen to make the test pass.
    """
    local_matrix(pole_lower, sensitivity_upper, kp, ki)
    limit = 2*(1+pole_lower)/((1-pole_lower)*sensitivity_upper)
    return dict(stable_for_each_fixed_model=bool(kp/2+ki < limit),
                gain_limit=limit, gain_used=kp/2+ki)


def rate_margin(dimensions, collapse_value, target):
    """Pigeonhole bound for the SAME population/split as the controlled rate."""
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError('dimensions must be a positive integer')
    if not all(math.isfinite(x) and x > 0 for x in (collapse_value, target)):
        raise ValueError('collapse_value and target must be positive and finite')
    return dict(minimum_max_rate_at_target=target/dimensions,
                strict_total_rate_floor=dimensions*collapse_value,
                tracking_error_budget=target-dimensions*collapse_value,
                target_excludes_total_collapse=bool(target/dimensions > collapse_value))

