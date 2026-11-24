"""Regression tests for startup, handover, and the reported collapse beta."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core_real_final.beta_results import collect_results
from src.core_real_final.guarded_trainer import GuardedTrainer


@pytest.mark.parametrize(
    "activity, expected_phase, expected_reason, first_active, rollback",
    [
        ([False] * 5, "annealing", None, None, None),
        ([False] * 6, "failed", "never_active_within_start_window", None, None),
        ([False] * 5 + [True, False], "annealing", None, 6, 5),
        ([True, False], "annealing", None, 1, 0),
    ],
)
def test_initial_activity(activity, expected_phase, expected_reason, first_active, rollback):
    trainer = object.__new__(GuardedTrainer)
    trainer.phase = "annealing"
    trainer.fail_reason = trainer.first_active_epoch_number = None
    trainer.restarts = trainer.min_handover_epoch = 0
    trainer.activation_window_epochs = 6
    trainer._anneal_ring, trainer._level_states = [], {}
    trainer._training_history = []
    trainer.handover_epoch = trainer.setpoint_C = trainer.gap_handover = None
    trainer.save_checkpoint = Mock()
    trainer.handover_checkpoint_dir = Path("unused")
    trainer.ring_buffer_size = 30
    for epoch, active in enumerate(activity):
        trainer._training_history.append({"R_train_nats": 0.1, "applied_factor": 0.02})
        trainer._eval_collapse = Mock(return_value={
            "R_train_eval_nats": 0.2 if active else 0.0,
            "lucas_full_collapse_train_eval": not active,
            "R_j_train_eval_nats": [0.2 if active else 0.0],
            "lucas_collapsed_dims_train_eval": [] if active else [0],
        })
        restart = trainer._lucas_epoch_end(epoch)
    assert trainer.phase == expected_phase
    assert trainer.fail_reason == expected_reason
    assert trainer.first_active_epoch_number == first_active
    assert (restart[0] if restart else None) == rollback
    assert trainer.save_checkpoint.call_count == sum(activity)


@pytest.mark.parametrize("handover", [False, True])
def test_beta_collapse_excludes_inactive_startup(tmp_path, handover):
    (tmp_path / "data_manifest.json").write_text(json.dumps({"dataset": "tcga", "group": "test"}))
    (tmp_path / "guard_summary.json").write_text(json.dumps({"lucas_handover": {"activation_window_epochs": 6}}))
    rows = [
        {"epoch_number": 1, "lucas_full_collapse_train_eval": True, "effective_beta": 0.01},
        {"epoch_number": 2, "lucas_full_collapse_train_eval": False},
        {"epoch_number": 3, "lucas_full_collapse_train_eval": handover,
         "lookback_handover": handover, "effective_beta": 0.03},
    ]
    (tmp_path / "collapse_history.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = collect_results(tmp_path)["tcga"]["test"][0]
    assert result["beta_collapse"] == (0.03 if handover else None)
