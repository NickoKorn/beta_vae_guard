"""Resumable RTX-5090 experiment launcher for TCGA baselines (freeze/early_stop/controlvae) and GUDHI.

Usage (Linux or Windows, from repo root):
    python -m src.core_real_final.run_5090_experiments OUTPUT_DIR --list
    python -m src.core_real_final.run_5090_experiments OUTPUT_DIR --suite all --workers 2
    python -m src.core_real_final.run_5090_experiments OUTPUT_DIR --suite tcga_baselines --workers 2
    python -m src.core_real_final.run_5090_experiments OUTPUT_DIR --suite gudhi --workers 2
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = Path(__file__).with_name("guarded_varix.json")

TCGA_GROUPS = ("weak_signal_lowvar", "ontix_standard", "all_classes")
GUDHI_TOPOLOGIES = ("circle", "sphere")
SPLITS = (1, 2, 3)
SEEDS = (1, 2, 3)


def build_config(out_dir: Path) -> Path:
    """Write the validated eps=0.1, hold=100, activation_window=6 experiment config."""
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["experiment"]["seeds"] = list(SEEDS)
    cfg["experiment"]["split_seeds"] = list(SPLITS)
    cfg["guard"].update({
        "enabled": True,
        "form": "controlvae",
        "kp": 0.002,
        "ki": 0.06,
        "gains_per": "step",
        "beta_min": 0.0,
        "beta_max": "trigger",
        "anti_windup": "conditional",
        "signal": "total",
    })
    cfg["protocol"]["collapse"]["lucas_epsilon_nats"] = 0.1
    cfg["protocol"]["collapse"]["lucas_delta"] = 0.01
    cfg["protocol"]["lucas_handover"].update({
        "enabled": True,
        "max_restarts": 15,
        "ring_buffer_size": 30,
        "control_steps": None,
        "control_epochs": 100,
        "beta_final_window": 20,
        "activation_window_epochs": 6,
        "min_handover_epoch": 0,
    })
    cfg_path = out_dir / "config_5090_eps01.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def build_jobs(suite: str) -> tuple[list[dict], list[dict]]:
    """Return (parallel_jobs, sequential_jobs). all_classes runs sequentially to avoid RAM spikes."""
    parallel_jobs = []
    sequential_jobs = []

    if suite in ("gudhi", "all"):
        # GUDHI circle & sphere: controlvae (ki=0.60 & ki=0.06 scaling comparison) + freeze + early_stop
        for topo, split, seed in product(GUDHI_TOPOLOGIES, SPLITS, SEEDS):
            for mode, ki, tag in (
                ("controlvae", 0.60, "controlvae-ki0.6"),
                ("controlvae", 0.06, "controlvae-ki0.06"),
                ("freeze", 0.60, "freeze"),
                ("early_stop", 0.60, "early_stop"),
            ):
                parallel_jobs.append({
                    "suite": "gudhi",
                    "dataset": "gudhi",
                    "target": topo,
                    "variant": "base",
                    "mode": mode,
                    "ki": ki,
                    "split": split,
                    "seed": seed,
                    "name": f"gudhi-{topo}-base-{tag}-eps01-split{split}-seed{seed}",
                })

    if suite in ("tcga_baselines", "all"):
        # TCGA fair baseline comparison on base: controlvae (ki=0.06) vs freeze vs early_stop
        for group, split, seed in product(TCGA_GROUPS, SPLITS, SEEDS):
            for mode in ("freeze", "early_stop", "controlvae"):
                job = {
                    "suite": "tcga_baselines",
                    "dataset": "tcga",
                    "target": group,
                    "variant": "base",
                    "mode": mode,
                    "ki": 0.06,
                    "split": split,
                    "seed": seed,
                    "name": f"tcga-{group}-base-{mode}-eps01-split{split}-seed{seed}",
                }
                if group == "all_classes":
                    sequential_jobs.append(job)
                else:
                    parallel_jobs.append(job)

    return parallel_jobs, sequential_jobs


def run_one(job: dict, *, out_dir: Path, cfg_path: Path, env: dict) -> tuple[str, str, float]:
    name = job["name"]
    run_dir = out_dir / "beta_guard_vae" / name
    if (run_dir / "guard_summary.json").is_file():
        return (name, "skipped", 0.0)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    logs_dir = out_dir / "worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    target_flag = "--group" if job["dataset"] == "tcga" else "--topology"
    cmd = [
        sys.executable, "-u", "-m", "src.core_real_final.main", str(out_dir),
        "--config", str(cfg_path),
        "--dataset", job["dataset"],
        target_flag, job["target"],
        "--variant", job["variant"],
        "--mode", job["mode"],
        "--ki", str(job["ki"]),
        "--split-seed", str(job["split"]),
        "--train-seed", str(job["seed"]),
        "--run-name", name,
    ]
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}", flush=True)
    with (logs_dir / f"{name}.log").open("w") as log:
        res = subprocess.run(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "ok" if res.returncode == 0 else f"err({res.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {name} -> {status} ({dt:.1f}s)", flush=True)
    return (name, status, dt)


def write_summary_csv(out_dir: Path) -> Path:
    rows = []
    for sp in sorted((out_dir / "beta_guard_vae").glob("*/guard_summary.json")):
        s = json.loads(sp.read_text())
        m = json.loads((sp.parent / "data_manifest.json").read_text())
        lh = s.get("lucas_handover") or {}
        ft = lh.get("final_lucas_test") or {}
        rows.append({
            "run_name": sp.parent.name,
            "dataset": m.get("dataset"),
            "target": m.get("group") or m.get("topology"),
            "variant": m.get("variant"),
            "mode": s.get("form"),
            "ki": s.get("ki"),
            "split": m.get("split_seed"),
            "seed": m.get("train_seed"),
            "status": s.get("status"),
            "phase": lh.get("phase"),
            "fail_reason": lh.get("fail_reason"),
            "restarts": lh.get("restarts"),
            "first_active_epoch": lh.get("first_active_epoch_number"),
            "handover_epoch": lh.get("handover_epoch"),
            "beta_handover_effective": lh.get("beta_handover_effective"),
            "beta_final_mean": lh.get("beta_final_mean"),
            "setpoint_C": lh.get("setpoint_C"),
            "kl_in_target_band_share": lh.get("kl_in_target_band_share"),
            "n_alive_final": lh.get("n_alive_final"),
            "test_alive": ft.get("lucas_n_noncollapsed_train_eval"),
            "test_full_collapse": ft.get("lucas_full_collapse_train_eval"),
            "epochs_total": lh.get("epochs_total"),
        })
    csv_path = out_dir / "summary_5090.csv"
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Output directory (e.g. /mnt/data/nickok/rtx5090_eps01)")
    parser.add_argument("--suite", choices=("all", "tcga_baselines", "gudhi"), default="all")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers for GUDHI and small TCGA groups (all_classes always runs with 1)")
    parser.add_argument("--list", action="store_true", help="List all planned jobs and exit")
    args = parser.parse_args()

    parallel_jobs, sequential_jobs = build_jobs(args.suite)
    total = len(parallel_jobs) + len(sequential_jobs)
    if args.list:
        for job in parallel_jobs + sequential_jobs:
            print(f"{job['suite']:15s}\t{job['dataset']}\t{job['target']:20s}\t{job['mode']:10s}\tki={job['ki']}\ts{job['split']}/t{job['seed']}\t{job['name']}")
        print(f"Total: {total} runs ({len(parallel_jobs)} parallel, {len(sequential_jobs)} sequential)")
        return

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = build_config(out_dir)
    env = {
        **os.environ,
        "PYTHONHASHSEED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    print(f"Starting suite={args.suite}: {total} runs ({len(parallel_jobs)} @ {args.workers} workers, "
          f"{len(sequential_jobs)} all_classes @ 1 worker)", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(lambda j: run_one(j, out_dir=out_dir, cfg_path=cfg_path, env=env), parallel_jobs))
    for job in sequential_jobs:
        run_one(job, out_dir=out_dir, cfg_path=cfg_path, env=env)

    csv_path = write_summary_csv(out_dir)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
