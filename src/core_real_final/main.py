"""Train one guarded Varix model on TCGA or a GUDHI point cloud."""

import argparse
import json
import os
from itertools import product
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("guarded_varix.json")


def print_lucas(trainer, epoch, record):
    """One line per epoch: phase, beta, rates and the collapsed axes per monitored epsilon."""
    by_eps = record.get("lucas_collapsed_dims_by_eps")
    if by_eps is None:
        return
    collapsed = "  ".join(f"eps={eps}: {len(dims)}/{len(record['R_j_train_eval_nats'])} {dims}"
                          for eps, dims in by_eps.items())
    print(f"[lucas] epoch {epoch:4d} phase={record.get('phase', '-'):9s} beta={record['applied_factor']:.4f} "
          f"R_train={record['R_train_nats']:.3f} R_eval={record['R_train_eval_nats']:.3f}  {collapsed}", flush=True)


def main():
    # Set iteration order in the AUTOENCODIX preprocessing depends on the hash seed: without a
    # fixed PYTHONHASHSEED the split and the selected features change between runs. It only
    # takes effect at interpreter start, so restart this script once with it set.
    if os.environ.get("PYTHONHASHSEED") != "1":
        os.environ["PYTHONHASHSEED"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        if os.name == "nt":
            # Windows: execv does not replace the process, so run a child and pass on its exit code
            sys.exit(subprocess.run(command).returncode)
        os.execv(sys.executable, command)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ssd_path", nargs="?", type=Path, default=Path("/mnt/data"),
                        help="SSD directory (default: /mnt/data)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Experiment configuration (default: guarded_varix.json next to main.py)")
    parser.add_argument("--dataset", choices=("tcga", "gudhi", "both"))
    grid_flags = parser.add_mutually_exclusive_group()
    grid_flags.add_argument("--list-grid", action="store_true", help="Print planned runs and exit")
    grid_flags.add_argument("--grid", action="store_true", help="Train the selected grid sequentially")
    parser.add_argument("--variant", help="Model configuration name from --list-grid")
    parser.add_argument("--group")
    parser.add_argument("--topology", choices=("circle", "sphere"))
    parser.add_argument("--mode", choices=("early_stop", "freeze", "controlvae"))
    parser.add_argument("--signal", choices=("total", "max"))
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--run-name")
    # Lucas-monitored handover (protocol.lucas_handover in the config; flags override it)
    parser.add_argument("--lucas-handover", action=argparse.BooleanOptionalAction, default=None,
                        help="enable the configured Lucas protocol (default: go back one healthy epoch on full collapse)")
    parser.add_argument("--handover-k", type=float, help="handover level k in k*d*eps")
    parser.add_argument("--setpoint-k", type=float, help="PI setpoint level k in k*d*eps (default: handover k)")
    parser.add_argument("--handover-k-step", type=float, help="raise of both k per restart")
    parser.add_argument("--max-restarts", type=int)
    parser.add_argument("--control-epochs", type=int)
    parser.add_argument("--observe-collapse", action="store_true",
                        help="plain annealing without handover/PI/early stop; log epoch and beta of each "
                             "Lucas axis collapse and of the full collapse")
    parser.add_argument("--observe-until-end", action="store_true",
                        help="with --observe-collapse: keep training after the confirmed full collapse")
    parser.add_argument("--trigger", choices=("rate", "lucas"),
                        help="handover trigger: rate (R <= k*d*eps) or lucas (>= int(fraction*d) axes "
                             "collapsed at --trigger-epsilon)")
    parser.add_argument("--trigger-epsilon", type=float, help="epsilon of the lucas trigger (default 1.0)")
    parser.add_argument("--trigger-fraction", type=float, help="share of axes for the lucas trigger (default 0.75)")
    parser.add_argument("--monitor-epsilons", help="comma-separated epsilons whose collapsed axes are "
                        "logged every epoch, e.g. 0.01,0.1,0.5,1")
    parser.add_argument("--collapse-epsilon", type=float,
                        help="Lucas epsilon of the (hard) collapse boundary for restarts and full collapse "
                             "(default: protocol.collapse.lucas_epsilon_nats); rate levels keep that value")
    parser.add_argument("--control-steps", type=int,
                        help="hold duration in training steps (overrides --control-epochs)")
    parser.add_argument("--print-lucas", action="store_true",
                        help="print the collapsed axes per monitored epsilon after every epoch")
    parser.add_argument("--kp", type=float, help="PI proportional gain (overrides guard.kp)")
    parser.add_argument("--ki", type=float, help="PI integral gain per guard.gains_per (overrides guard.ki)")
    parser.add_argument("--beta-max", help="PI upper beta limit: trigger (beta at handover), schedule "
                        "(annealing value) or a number (overrides guard.beta_max)")
    args = parser.parse_args()

    ssd_path = args.ssd_path.expanduser().resolve()
    if not args.list_grid and not ssd_path.is_dir():
        parser.error(f"SSD directory does not exist: {ssd_path}")

    if not __package__:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from src.core.load_config import load_config, training_configs

    settings = load_config(args.config.expanduser())
    dataset = args.dataset or settings.get("dataset", "tcga")
    experiment = settings.get("experiment", {})
    if args.list_grid or args.grid:
        if args.run_name:
            parser.error("--run-name is for single runs")
        if (args.group and dataset != "tcga") or (args.topology and dataset != "gudhi"):
            parser.error("Select one matching dataset when filtering by group or topology")
        split_seeds = [args.split_seed] if args.split_seed is not None else experiment["split_seeds"]
        train_seeds = [args.train_seed] if args.train_seed is not None else experiment["seeds"]
        count = 0
        for source in (("tcga", "gudhi") if dataset == "both" else (dataset,)):
            variants = dict(training_configs(settings, dataset=source))
            if args.variant:
                if args.variant not in variants:
                    parser.error(f"Unknown variant for {source}: {args.variant}")
                variants = {args.variant: variants[args.variant]}
            if source == "tcga":
                option, names = "--group", [args.group] if args.group else experiment["groups"]
            else:
                option, names = "--topology", [args.topology] if args.topology else experiment["topologies"]
            for variant, name, split, seed in product(variants, names, split_seeds, train_seeds):
                count += 1
                if args.list_grid:
                    print(f"{source}\t{name}\t{variant}\tsplit={split}\tseed={seed}")
                else:
                    # Each child uses the same single-run entry point and a fresh Fabric trainer.
                    forwarded = [arg for arg in sys.argv[1:] if arg != "--grid"]
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), *forwarded,
                                    "--dataset", source, option, name, "--variant", variant,
                                    "--split-seed", str(split), "--train-seed", str(seed)], check=True)
        print(f"{count} runs")
        return
    if dataset not in ("tcga", "gudhi"):
        parser.error("This entry point runs one dataset at a time; choose --dataset tcga or gudhi")
    training = {**settings["training"], **settings.get("datasets", {}).get(dataset, {})}
    if args.variant:
        variants = dict(training_configs(settings, dataset=dataset))
        if args.variant not in variants:
            parser.error(f"Unknown variant for {dataset}: {args.variant}")
        training = variants[args.variant]
    training.setdefault("anneal_function", "logistic-mid")
    guard = dict(settings.get("guard", {}))
    for key, value in (("kp", args.kp), ("ki", args.ki)):
        if value is not None:
            guard[key] = value
    if args.beta_max is not None:
        guard["beta_max"] = args.beta_max if args.beta_max in ("trigger", "schedule") else float(args.beta_max)
    protocol = settings.get("protocol", {})
    collapse = protocol.get("collapse", {})
    lucas = dict(protocol.get("lucas_handover", {}))
    lucas.pop("description", None)
    configured_steps = lucas.pop("control_steps", None)
    for key, value in (("enabled", args.lucas_handover), ("handover_k", args.handover_k),
                       ("setpoint_k", args.setpoint_k), ("handover_k_step", args.handover_k_step),
                       ("max_restarts", args.max_restarts), ("control_epochs", args.control_epochs),
                       ("handover_trigger", args.trigger), ("trigger_epsilon", args.trigger_epsilon),
                       ("trigger_fraction", args.trigger_fraction)):
        if value is not None:
            lucas[key] = value
    if args.handover_k is not None and args.setpoint_k is None:
        lucas["setpoint_k"] = args.handover_k   # setpoint follows a changed handover level unless given
    lucas_enabled = bool(lucas.pop("enabled", False)) and not args.observe_collapse
    if args.observe_until_end and not args.observe_collapse:
        parser.error("--observe-until-end requires --observe-collapse")
    mode = args.mode or guard.get("form", "controlvae")
    # The Lucas handover feeds the PI with the training-batch total KL; a validation signal
    # (e.g. guard.signal = valid_total) would let validation data steer training.
    signal = args.signal or ("total" if lucas_enabled or args.observe_collapse else guard.get("signal", "total"))
    if mode not in ("early_stop", "freeze", "controlvae"):
        parser.error("Set guard.form or --mode to early_stop, freeze or controlvae")
    if signal not in ("total", "max"):
        parser.error("Set guard.signal or --signal to total or max for training KL feedback")
    if lucas_enabled and signal != "total":
        parser.error("--lucas-handover requires --signal total")
    if dataset == "tcga":
        group = args.group or (experiment.get("groups") or ["all_classes"])[0]
        topology = None
    else:
        if args.group:
            parser.error("--group applies only to TCGA")
        group = None
        topology = args.topology or (experiment.get("topologies") or ["sphere"])[0]
    seed = training.get("global_seed", 1)
    split_seed = args.split_seed if args.split_seed is not None else (experiment.get("split_seeds") or [seed])[0]
    train_seed = args.train_seed if args.train_seed is not None else (experiment.get("seeds") or [seed])[0]
    handover_kl = float(guard.get("handover_kl", guard.get("control_value", 0.10)))
    target_kl = float(guard.get("target_kl", 0.05))
    guard.update(form=mode, signal=signal)
    guard.setdefault("guard_observe_start_value", 2 * handover_kl)
    data_name = group if dataset == "tcga" else topology
    if lucas_enabled and lucas.get("single_threshold", True):
        lucas_tag = "-lucas-lookback"
    elif lucas_enabled and lucas.get("handover_trigger", "rate") == "lucas":
        lucas_tag = f"-lucastrig-e{lucas.get('trigger_epsilon', 1.0):g}-f{lucas.get('trigger_fraction', 0.75):g}"
    elif lucas_enabled:
        lucas_tag = f"-lucas-k{lucas.get('handover_k', 3):g}-sp{lucas.get('setpoint_k', lucas.get('handover_k', 3)):g}"
    else:
        lucas_tag = ""
    monitor_epsilons = [float(e) for e in args.monitor_epsilons.split(",")] if args.monitor_epsilons else []
    mode_tag = "observe" if args.observe_collapse else mode
    variant_tag = f"-{args.variant}" if args.variant else ""
    run_name = args.run_name or f"{dataset}-{data_name}{variant_tag}-{mode_tag}{lucas_tag}-split{split_seed}-seed{train_seed}"
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    output = ssd_path / "beta_guard_vae" / run_name
    if output.exists():
        parser.error(f"Run directory already exists: {output}")

    from autoencodix import Varix
    from autoencodix.configs.default_config import DataCase
    from autoencodix.configs.varix_config import VarixConfig
    from autoencodix.data.datapackage import DataPackage
    from autoencodix.modeling._varix_architecture import VarixArchitecture
    from autoencodix.utils._losses import VarixLoss
    from src.core_real_final.load_data import gudhi, tcga, variance_order
    from src.core_real_final.guarded_trainer import GuardedTrainer

    if training.get("default_vae_loss", "kl") != "kl":
        parser.error("Guarded training requires default_vae_loss=kl")

    if dataset == "tcga":
        tcga_settings = settings.get("data", {}).get("tcga")
        if not tcga_settings:
            parser.error("The selected config must define data.tcga")
        lowvar_suffix = tcga_settings["lowvar_suffix"]
        group_names = set(tcga_settings["groups"])
        valid_groups = group_names | {name + lowvar_suffix for name in group_names}
        if group not in valid_groups:
            parser.error(f"Unknown TCGA group {group!r}; choose from {sorted(valid_groups)}")
        data_root = ROOT / tcga_settings["root"]
        modalities, annotation = tcga(group, folder=data_root, tcga_config=tcga_settings)
        order = "low" if group.endswith(lowvar_suffix) else "top"
        feature_order_name = order
    else:
        gudhi_settings = settings.get("data", {}).get("gudhi")
        if not gudhi_settings:
            parser.error("The selected config must define data.gudhi")
        data_root = ROOT / gudhi_settings["root"]
        modalities, annotation = gudhi(topology, folder=data_root, gudhi_config=gudhi_settings)
        order = "top"
        feature_order_name = None

    config = VarixConfig(**{**training, "data_case": DataCase.MULTI_BULK, "global_seed": split_seed})
    data_package = DataPackage(
        multi_bulk=modalities,
        annotation={"paired": annotation} if annotation is not None else None,
    )
    if dataset == "tcga":
        from src.core_real_final.tcga_preprocessor import TCGAPreprocessor

        pipeline = Varix(data=data_package, config=config, preprocessor_type=TCGAPreprocessor)
    else:
        pipeline = Varix(data=data_package, config=config)
    with variance_order(order) as filtered:
        pipeline.preprocess()
    if dataset == "tcga" and order == "low" and not filtered:
        raise ValueError("Low-variance groups need a variance filter with k_filter configured")
    datasets = pipeline.preprocessed_data
    config = pipeline.config
    config.global_seed = train_seed

    output.mkdir(parents=True)
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    manifest = {
        "sample_ids": {name: [str(value) for value in getattr(datasets, name).sample_ids]
                       for name in ("train", "valid", "test") if getattr(datasets, name) is not None},
        "feature_ids": [str(value) for value in datasets.train.feature_ids],
        "feature_order": feature_order_name,
        "dataset": dataset,
        "variant": args.variant or "base",
        "group": group,
        "topology": topology,
        "split_unit": "tcga_participant" if dataset == "tcga" else "sample",
        "split_seed": split_seed,
        "train_seed": train_seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trainer = GuardedTrainer(
        trainset=datasets.train, validset=datasets.valid, result=pipeline.result,
        config=config, model_type=VarixArchitecture, loss_type=VarixLoss,
        collapse_output=output, collapse_threshold=float(guard.get("collapse_value", 0.01)),
        guard_enabled=guard.get("enabled", True), guard_options=guard,
        handover_kl=handover_kl, target_kl=target_kl,
        lucas_epsilon=(args.collapse_epsilon if args.collapse_epsilon is not None
                       else float(collapse.get("lucas_epsilon_nats", 0.01))),
        rate_epsilon=float(collapse.get("lucas_epsilon_nats", 0.01)),
        control_steps=args.control_steps if args.control_steps is not None else configured_steps,
        lucas_delta=float(collapse.get("lucas_delta", 0.01)),
        lucas_confirm_epochs=int(collapse.get("lucas_confirm_epochs", 3)),
        lucas_handover=lucas_enabled,
        lucas_observe=args.observe_collapse,
        observe_until_end=args.observe_until_end,
        monitor_epsilons=monitor_epsilons,
        on_epoch=print_lucas if args.print_lucas else None,
        # level checkpoints next to the run on the SSD; unused ones are deleted after training
        handover_checkpoint_dir=output / "handover_levels",
        **lucas,
    )
    # with the Lucas handover the level checkpoints replace the single handover.pt
    trainer.train(handover_checkpoint=None if lucas_enabled else output / "handover.pt",
                  export_posteriors=protocol.get("export_posteriors", False))
    if lucas_enabled:
        # Report only, after training: Lucas on the untouched test split (no influence on training).
        summary_path = output / "guard_summary.json"
        summary = json.loads(summary_path.read_text())
        if datasets.test is not None:
            summary["lucas_handover"]["final_lucas_test"] = trainer._eval_collapse(datasets.test)
            summary["lucas_handover"]["notes"] = [
                note for note in summary["lucas_handover"]["notes"] if not note.startswith("Testsatz")
            ] + ["final_lucas_test: test split, eval mode, final model, measured once after training."]
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if dataset == "tcga" and (output / "posterior_index.jsonl").is_file():
        from src.analysis.downstream import analyze_posteriors

        analyze_posteriors(output, annotation, target=tcga_settings["label"])
    print(output, flush=True)


if __name__ == "__main__":
    main()
