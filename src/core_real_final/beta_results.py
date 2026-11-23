"""Collect observed collapse and PI beta values from completed runs.

Usage: python -m src.core_real_final.beta_results RESULTS_DIR OUTPUT.json
"""

import argparse
import json
from pathlib import Path


def collect_results(root):
    """Return results grouped by dataset and TCGA group or GUDHI topology."""
    results = {"tcga": {}, "gudhi": {}}
    for summary_path in sorted(Path(root).rglob("guard_summary.json")):
        folder = summary_path.parent
        manifest_path = folder / "data_manifest.json"
        history_path = folder / "collapse_history.jsonl"
        if not manifest_path.is_file() or not history_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        dataset = manifest.get("dataset")
        if dataset not in results:
            continue
        name = manifest.get("group") if dataset == "tcga" else manifest.get("topology")
        if not name:
            continue
        records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        collapse = next((row for row in records if row.get("lucas_full_collapse_train_eval") is True), None)
        protocol = summary.get("lucas_handover") or {}
        success = summary.get("form") == "controlvae" and protocol.get("phase") == "done" and protocol.get("final_lucas_train_eval", {}).get(
            "lucas_full_collapse_train_eval") is False
        attempt = protocol.get("restarts")
        if protocol.get("single_threshold") and summary.get("form") == "controlvae":
            # Older summaries also counted the rejected restart after exhausting the budget.
            attempt = max((row["attempt"] for row in records if row.get("state") == "hold"
                           and row.get("attempt") is not None), default=0)
        control = [row for row in records if row.get("attempt") == attempt
                   and row.get("phase") in ("control", "done")]
        last = control[-1] if control else None
        scale = summary.get("config_beta")
        start_factor = protocol.get("beta_handover")
        run = {
            "run_dir": str(folder),
            "variant": manifest.get("variant", "base"),
            "split_seed": manifest.get("split_seed"),
            "train_seed": manifest.get("train_seed"),
            "beta_collapse": collapse.get("beta_batch_last", collapse.get("effective_beta")) if collapse else None,
            "collapse_epoch": collapse.get("epoch_number") if collapse else None,
            "kl_train_total_at_collapse": collapse.get("R_train_nats") if collapse else None,
            "kl_eval_total_at_collapse": collapse.get("R_train_eval_nats") if collapse else None,
            "lucas_q99_eval_per_dimension_at_collapse": (
                collapse.get("lucas_q99_train_eval_nats") if collapse else None
            ),
            "lucas_sample_share_ge_epsilon_per_dimension_at_collapse": (
                collapse.get("lucas_share_active_train_eval") if collapse else None
            ),
            "lucas_noncollapsed_dimensions_at_collapse": (
                collapse.get("lucas_n_noncollapsed_train_eval") if collapse else None
            ),
            "beta_pi_start": (protocol.get("beta_handover_effective")
                              if protocol.get("beta_handover_effective") is not None else
                              (start_factor * scale if start_factor is not None and scale is not None else None)),
            "beta_stable": last.get("beta_batch_last", last.get("effective_beta")) if success and last else None,
            "kl_train_total_at_stable": last.get("R_train_nats") if success and last else None,
            "kl_eval_total_at_stable": last.get("R_train_eval_nats") if success and last else None,
            "lucas_q99_eval_per_dimension_at_stable": (
                last.get("lucas_q99_train_eval_nats") if success and last else None
            ),
            "lucas_sample_share_ge_epsilon_per_dimension_at_stable": (
                last.get("lucas_share_active_train_eval") if success and last else None
            ),
            "lucas_noncollapsed_dimensions_at_stable": (
                last.get("lucas_n_noncollapsed_train_eval") if success and last else None
            ),
            "lucas_epsilon_nats": protocol.get("collapse_epsilon"),
            "beta_pi_epoch_mean": last.get("beta_batch_mean") if success and last else None,
            "beta_pi_epoch_min": last.get("beta_batch_min") if success and last else None,
            "beta_pi_epoch_max": last.get("beta_batch_max") if success and last else None,
            "restarts": attempt,
            "kl_in_target_band_share": protocol.get("kl_in_target_band_share") if success else None,
            "control_epochs_requested": protocol.get("control_epochs") if protocol.get("control_steps") is None else None,
            "control_steps_requested": protocol.get("control_steps"),
            "status": "stabilized" if success else (protocol.get("fail_reason") or protocol.get("phase") or "unknown"),
            "note": "Beta values are effective loss weights; stability means the recorded control period without full Lucas collapse.",
        }
        results[dataset].setdefault(name, []).append(run)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect_results(args.results_dir), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
