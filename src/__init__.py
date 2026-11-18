"""Varix experiments: core training, analysis and experiment orchestration."""

from importlib import import_module
import sys

# Old saved pipelines refer to these module names. Keep them resolvable without
# duplicate files; every alias points to the canonical implementation module.
for _old, _new in {
    "collapse_guard": "core.collapse_guard",
    "guarded_trainer": "core.guarded_trainer",
    "guard_plots": "analysis.plots",
    "analysis.guard_plots": "analysis.plots",   # renamed 2026-11-18
    "analysis.lucas_collapse": "core.guarded_trainer",   # lucas_collapse lives in the trainer (measured per epoch)
    "core.lucas_collapse": "core.guarded_trainer",
}.items():
    _module = import_module(f"{__name__}.{_new}")
    sys.modules[f"{__name__}.{_old}"] = _module
    globals()[_old] = _module

del _old, _new, _module
