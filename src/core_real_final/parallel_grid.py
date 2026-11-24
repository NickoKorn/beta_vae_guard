"""Run the TCGA configuration grid with multiple independent workers."""

import argparse
import concurrent.futures
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from filelock import FileLock, Timeout

from .fetch_tcga import fetch_tcga
from .load_config import load_config, training_configs


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config_path = args.config.resolve()
    settings = load_config(config_path)
    if settings.get("dataset") != "tcga":
        parser.error("The parallel grid currently supports dataset='tcga'")
    if not settings.get("guard", {}).get("enabled", True) or settings.get("guard", {}).get("form", "controlvae") != "controlvae" or not settings.get("protocol", {}).get("lucas_handover", {}).get("enabled", False):
        parser.error("The parallel grid requires the configured ControlVAE Lucas handover")
    if not settings.get("protocol", {}).get("export_posteriors", False):
        parser.error("The parallel grid requires export_posteriors=true")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "parallel_grid_config.json"
    lock = FileLock(str(output / "parallel_grid.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        raise SystemExit(f"A parallel grid is already using {output}")
    try:
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != settings:
                raise ValueError(f"Configuration differs from {marker}; choose another output folder")
        else:
            existing = output / "beta_guard_vae"
            if existing.exists() and any(existing.iterdir()):
                raise ValueError(f"Existing runs in {existing} have no matching grid configuration; choose another output folder")
            marker.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        tcga = settings["data"]["tcga"]
        fetch_tcga(ROOT / tcga["root"], tcga["source"])
        experiment = settings["experiment"]
        jobs = []
        logs = output / "worker_logs"
        logs.mkdir(exist_ok=True)
        for (variant, _), group, split, seed in itertools.product(
            training_configs(settings, dataset="tcga"), experiment["groups"],
            experiment["split_seeds"], experiment["seeds"],
        ):
            name = f"tcga-{group}-{variant}-controlvae-lucas-lookback-split{split}-seed{seed}"
            folder = output / "beta_guard_vae" / name
            required = ("guard_summary.json", "data_manifest.json", "auroc.csv", "posterior_index.jsonl")
            if all((folder / item).is_file() for item in required):
                continue
            if folder.exists():
                archive = output / "interrupted_runs"
                archive.mkdir(exist_ok=True)
                preserved = archive / f"{name}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
                folder.rename(preserved)
                print(f"Preserved incomplete run: {preserved}", flush=True)
            jobs.append((variant, group, split, seed, name))

        print(f"Starting {len(jobs)} remaining runs with {args.workers} workers", flush=True)

        def run(job):
            variant, group, split, seed, name = job
            command = [sys.executable, "-u", "-m", "src.core_real_final.main", str(output),
                       "--config", str(marker), "--dataset", "tcga", "--group", group,
                       "--variant", variant, "--split-seed", str(split), "--train-seed", str(seed)]
            print(f"START {name}", flush=True)
            with (logs / f"{name}.log").open("a", encoding="utf-8") as log:
                log.write(f"\nStarting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"END {name}: exit {result.returncode}", flush=True)
            return result.returncode

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            codes = list(pool.map(run, jobs))
        failed = sum(code != 0 for code in codes)
        print(f"Finished {len(codes)} runs; process errors: {failed}", flush=True)
        return 1 if failed else 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
