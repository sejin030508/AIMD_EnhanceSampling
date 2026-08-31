from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from confmh.analysis import analyze_baseline_comparison, analyze_run, analyze_umbrella_suite
from confmh.config import load_config, resolve_path
from confmh.diagnostics import diagnose_reversibility
from confmh.doctor import doctor
from confmh.experiment import run_experiment
from confmh.forward import run_forward_baseline
from confmh.pca_cv import fit_reference_pca, project_reference_trajectories
from confmh.sweep import make_umbrella_configs
from confmh.utils import expand_globs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="confmh")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("doctor", help="Check baseline, data, and dependencies")
    check.add_argument("--config", required=True)

    fit = sub.add_parser("fit-reference", help="Fit C-alpha PCA and build the public-MD oracle")
    fit.add_argument("--config", required=True)

    project = sub.add_parser(
        "project-reference", help="Project the reference at a comparison-matched stride"
    )
    project.add_argument("--config", required=True)

    run = sub.add_parser("run", help="Run raw ConfRover or bias-only MH")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--output-dir")

    forward = sub.add_parser(
        "run-forward", help="Run an official-style full-history ConfRover baseline"
    )
    forward.add_argument("--config", required=True)
    forward.add_argument("--seed", type=int)
    forward.add_argument("--output-dir")

    diagnose = sub.add_parser("diagnose-reversibility", help="Estimate projected detailed balance")
    diagnose.add_argument("--config", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one completed run")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--bins", type=int, default=80)

    comparison = sub.add_parser(
        "analyze-comparison", help="Aggregate the matched 100 ns baseline replicates"
    )
    comparison.add_argument("--full-history-glob", required=True)
    comparison.add_argument("--one-step-glob", required=True)
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--bins", type=int, default=20)

    sweep = sub.add_parser("make-umbrella", help="Create five reference-quantile umbrella configs")
    sweep.add_argument("--template", required=True)
    sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--quantiles", nargs="*", type=float)

    suite = sub.add_parser("analyze-umbrella", help="WHAM-combine completed umbrella windows")
    suite.add_argument("--run-glob", required=True)
    suite.add_argument("--output-dir", required=True)
    suite.add_argument("--bins", type=int, default=80)

    level11_validate = sub.add_parser(
        "level11-validate", help="Validate Level 1.1 assets, PC1, units, and CUDA"
    )
    level11_validate.add_argument("--config", required=True)
    level11_validate.add_argument("--output-dir")

    level11_smoke = sub.add_parser(
        "level11-smoke", help="Verify upstream/r=0 equivalence and one guided proposal"
    )
    level11_smoke.add_argument("--config", required=True)
    level11_smoke.add_argument("--output-dir")

    level11_prepare = sub.add_parser(
        "level11-prepare-opes", help="Build the fixed oracle OPES bias from reference CV"
    )
    level11_prepare.add_argument("--config", required=True)

    level11_pilot = sub.add_parser(
        "level11-pilot", help="Run paired unsteered/guided proposal pilot"
    )
    level11_pilot.add_argument("--config", required=True)

    level11_run = sub.add_parser(
        "level11-run", help="Run a resumable matched Level 1.1 chain"
    )
    level11_run.add_argument("--config", required=True)
    level11_run.add_argument("--method", required=True, choices=("unsteered", "guided"))
    level11_run.add_argument("--output-dir")

    level11_analyze = sub.add_parser("level11-analyze", help="Analyze one Level 1.1 run")
    level11_analyze.add_argument("--run-dir", required=True)
    level11_analyze.add_argument("--bins", type=int, default=40)

    level11_compare = sub.add_parser(
        "level11-compare", help="Compare matched unsteered and guided Level 1.1 runs"
    )
    level11_compare.add_argument("--unsteered-dir", required=True)
    level11_compare.add_argument("--guided-dir", required=True)
    level11_compare.add_argument("--output-dir", required=True)

    level11_sweep = sub.add_parser(
        "level11-make-umbrella", help="Generate standard or hard Level 1.1 umbrella configs"
    )
    level11_sweep.add_argument("--template", required=True)
    level11_sweep.add_argument("--output-dir", required=True)
    level11_sweep.add_argument("--hard", action="store_true")

    level11_pilot_gate = sub.add_parser(
        "level11-check-pilot", help="Require a passing pilot before umbrella runs"
    )
    level11_pilot_gate.add_argument("--config", required=True)
    level11_pilot_gate.add_argument("--summary", required=True)

    level11_stage2_gate = sub.add_parser(
        "level11-stage2-gate", help="Require umbrella efficiency/quality gates before OPES"
    )
    level11_stage2_gate.add_argument("--root-dir", required=True)

    discovery_prepare = sub.add_parser(
        "discovery-prepare-bioemu", help="Prepare the preregistered 3+3 BioEmu structure pairs"
    )
    discovery_prepare.add_argument("--bioemu-repo", default="external/bioemu-benchmarks")
    discovery_prepare.add_argument("--output-dir", default="data/discovery/bioemu")
    discovery_prepare.add_argument("--proar-data-dir", default="data/proar_discovery")
    discovery_prepare.add_argument("--include-reverse", action="store_true")

    discovery_configs = sub.add_parser(
        "discovery-make-bioemu-configs", help="Generate raw/blind/aware configs for both models"
    )
    discovery_configs.add_argument("--manifest", required=True)
    discovery_configs.add_argument("--output-dir", default="configs/discovery/generated")
    discovery_configs.add_argument("--project-root", default=".")
    discovery_configs.add_argument("--total-model-calls", type=int, default=128)
    discovery_configs.add_argument("--candidates-per-step", type=int, default=4)

    discovery_doctor_parser = sub.add_parser(
        "discovery-doctor", help="Validate an MH-free discovery config and model assets"
    )
    discovery_doctor_parser.add_argument("--config", required=True)
    discovery_doctor_parser.add_argument("--output-path")

    discovery_run = sub.add_parser(
        "discovery-run", help="Run an MH-free raw or bias-resampled autoregressive rollout"
    )
    discovery_run.add_argument("--config", required=True)
    discovery_run.add_argument("--output-dir")
    discovery_run.add_argument("--total-model-calls", type=int)
    discovery_run.add_argument("--candidates-per-step", type=int)
    discovery_run.add_argument("--centers", nargs="*", type=float)

    discovery_analyze = sub.add_parser(
        "discovery-analyze-alternate", help="Evaluate generated structures against hidden alternate state"
    )
    discovery_analyze.add_argument("--run-dir", required=True)

    discovery_aggregate = sub.add_parser(
        "discovery-aggregate-alternate", help="Aggregate all preregistered alternate-state runs"
    )
    discovery_aggregate.add_argument("--run-glob", required=True)
    discovery_aggregate.add_argument("--output-dir", required=True)

    fastfold_prepare = sub.add_parser(
        "discovery-prepare-fastfold", help="Fit reference-only TICA and metastable-state labels"
    )
    fastfold_prepare.add_argument("--topology", required=True)
    fastfold_prepare.add_argument("--trajectories", nargs="+", required=True)
    fastfold_prepare.add_argument("--output", required=True)
    fastfold_prepare.add_argument("--stride", type=int, default=1)
    fastfold_prepare.add_argument("--lag-frames", type=int, default=10)
    fastfold_prepare.add_argument("--components", type=int, default=3)
    fastfold_prepare.add_argument("--clusters", type=int, default=20)
    fastfold_prepare.add_argument("--grid-bins", type=int, default=30)
    fastfold_prepare.add_argument("--free-energy-threshold", type=float, default=4.0)
    fastfold_prepare.add_argument("--max-frames", type=int)

    fastfold_analyze = sub.add_parser(
        "discovery-analyze-fastfold", help="Compute valid low-energy coverage and TTC"
    )
    fastfold_analyze.add_argument("--run-dir", required=True)
    fastfold_analyze.add_argument("--start-pdb", required=True)
    fastfold_analyze.add_argument("--reference-model", required=True)
    fastfold_analyze.add_argument("--raw-final-coverage", type=float)

    fastfold_status = sub.add_parser(
        "discovery-fastfold-status", help="Report availability of Protein G, WW, and alpha3D data"
    )
    fastfold_status.add_argument("--root", default="data/fastfold")

    fastfold_configs = sub.add_parser(
        "discovery-make-fastfold-configs", help="Generate raw/deployable/oracle configs for both models"
    )
    fastfold_configs.add_argument("--data-root", default="data/fastfold")
    fastfold_configs.add_argument("--output-dir", default="configs/discovery/fastfold_generated")
    fastfold_configs.add_argument("--project-root", default=".")
    fastfold_configs.add_argument("--total-model-calls", type=int, default=160)
    fastfold_configs.add_argument("--candidates-per-step", type=int, default=4)

    minimize_hits = sub.add_parser(
        "discovery-minimize-hits", help="Run restrained OpenMM minimization on valid alternate hits"
    )
    minimize_hits.add_argument("--metrics-csv", required=True)
    minimize_hits.add_argument("--output-dir", required=True)
    minimize_hits.add_argument("--hit-column", default="valid_hit_3a")
    minimize_hits.add_argument("--max-iterations", type=int, default=500)

    return parser


def _apply_run_overrides(cfg, args):
    cfg.setdefault("run", {})
    if getattr(args, "seed", None) is not None:
        cfg["run"]["seed"] = int(args.seed)
    if getattr(args, "output_dir", None) is not None:
        cfg["run"]["output_dir"] = str(args.output_dir)
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor(load_config(args.config))
    if args.command == "fit-reference":
        cfg = load_config(args.config)
        reference = cfg["reference"]
        trajectories = expand_globs(reference["trajectory_globs"], root=resolve_path(cfg, "."))
        if not trajectories:
            raise FileNotFoundError("No reference trajectories matched reference.trajectory_globs")
        output = fit_reference_pca(
            topology_pdb=resolve_path(cfg, reference["topology_pdb"]),
            trajectories=trajectories,
            output_dir=resolve_path(cfg, reference.get("output_dir", "data/reference/6j56_A")),
            protein_selection=str(reference.get("protein_selection", "protein and chainid 0")),
            stride=int(reference.get("stride", 1)),
            burn_in_frames=int(reference.get("burn_in_frames", 0)),
            chunk_size=int(reference.get("chunk_size", 1000)),
            max_frames_per_trajectory=(
                None
                if reference.get("max_frames_per_trajectory") is None
                else int(reference["max_frames_per_trajectory"])
            ),
            n_components=int(reference.get("n_components", 2)),
            n_seed_frames=int(reference.get("n_seed_frames", 20)),
        )
        print(output)
        return 0
    if args.command == "project-reference":
        cfg = load_config(args.config)
        reference = cfg["reference"]
        trajectories = expand_globs(reference["trajectory_globs"], root=resolve_path(cfg, "."))
        if not trajectories:
            raise FileNotFoundError("No reference trajectories matched reference.trajectory_globs")
        output = project_reference_trajectories(
            pca_model=resolve_path(cfg, reference["pca_model"]),
            topology_pdb=resolve_path(cfg, reference["topology_pdb"]),
            trajectories=trajectories,
            output_path=resolve_path(cfg, reference["matched_reference_cv"]),
            protein_selection=str(reference.get("protein_selection", "protein and chainid 0")),
            stride=int(reference.get("comparison_stride", 256)),
            burn_in_frames=int(reference.get("comparison_burn_in_frames", 1)),
            chunk_size=int(reference.get("chunk_size", 1000)),
            max_frames_per_trajectory=(
                None
                if reference.get("max_frames_per_trajectory") is None
                else int(reference["max_frames_per_trajectory"])
            ),
        )
        print(output)
        return 0
    if args.command == "run":
        print(run_experiment(_apply_run_overrides(load_config(args.config), args)))
        return 0
    if args.command == "run-forward":
        print(run_forward_baseline(_apply_run_overrides(load_config(args.config), args)))
        return 0
    if args.command == "diagnose-reversibility":
        print(diagnose_reversibility(load_config(args.config)))
        return 0
    if args.command == "analyze":
        print(analyze_run(args.run_dir, bins=args.bins))
        return 0
    if args.command == "analyze-comparison":
        print(
            analyze_baseline_comparison(
                sorted(glob.glob(args.full_history_glob)),
                sorted(glob.glob(args.one_step_glob)),
                args.output_dir,
                bins=args.bins,
            )
        )
        return 0
    if args.command == "make-umbrella":
        paths = make_umbrella_configs(args.template, args.output_dir, args.quantiles)
        print("\n".join(str(path) for path in paths))
        return 0
    if args.command == "analyze-umbrella":
        run_dirs = sorted(glob.glob(args.run_glob))
        print(analyze_umbrella_suite(run_dirs, args.output_dir, bins=args.bins))
        return 0
    if args.command == "level11-validate":
        from confmh.level11.runner import validate_runtime

        print(validate_runtime(load_config(args.config), output_dir=args.output_dir))
        return 0
    if args.command == "level11-smoke":
        from confmh.level11.runner import validate_model_smoke

        print(validate_model_smoke(load_config(args.config), output_dir=args.output_dir))
        return 0
    if args.command == "level11-prepare-opes":
        from confmh.level11.runner import prepare_frozen_opes

        print(prepare_frozen_opes(load_config(args.config)))
        return 0
    if args.command == "level11-pilot":
        from confmh.level11.runner import run_pilot

        print(run_pilot(load_config(args.config)))
        return 0
    if args.command == "level11-run":
        from confmh.level11.runner import run_chain

        print(
            run_chain(
                load_config(args.config),
                guided=args.method == "guided",
                output_dir=args.output_dir,
            )
        )
        return 0
    if args.command == "level11-analyze":
        from confmh.level11.analysis import analyze_level11_run

        print(analyze_level11_run(args.run_dir, bins=args.bins))
        return 0
    if args.command == "level11-compare":
        from confmh.level11.analysis import compare_level11_runs

        print(
            compare_level11_runs(
                args.unsteered_dir,
                args.guided_dir,
                args.output_dir,
            )
        )
        return 0
    if args.command == "level11-make-umbrella":
        from confmh.level11.sweep import make_level11_umbrella_configs

        paths = make_level11_umbrella_configs(
            args.template,
            args.output_dir,
            hard=args.hard,
        )
        print("\n".join(str(path) for path in paths))
        return 0
    if args.command == "level11-check-pilot":
        from confmh.level11.analysis import check_pilot_gate

        print(check_pilot_gate(args.config, args.summary))
        return 0
    if args.command == "level11-stage2-gate":
        from confmh.level11.analysis import evaluate_stage2_gate

        print(evaluate_stage2_gate(args.root_dir))
        return 0
    if args.command == "discovery-prepare-bioemu":
        from confmh.discovery.bioemu import prepare_bioemu_assets

        print(
            prepare_bioemu_assets(
                bioemu_repo=args.bioemu_repo,
                output_dir=args.output_dir,
                proar_data_dir=args.proar_data_dir,
                include_reverse=args.include_reverse,
            )
        )
        return 0
    if args.command == "discovery-make-bioemu-configs":
        from confmh.discovery.bioemu import make_bioemu_configs

        paths = make_bioemu_configs(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            project_root=args.project_root,
            total_model_calls=args.total_model_calls,
            candidates_per_step=args.candidates_per_step,
        )
        print(f"wrote {len(paths)} configs under {args.output_dir}")
        return 0
    if args.command == "discovery-doctor":
        from confmh.discovery.doctor import discovery_doctor

        print(discovery_doctor(load_config(args.config), args.output_path))
        return 0
    if args.command == "discovery-run":
        from confmh.discovery.runner import run_controlled_rollout

        cfg = load_config(args.config)
        if args.output_dir is not None:
            cfg.setdefault("run", {})["output_dir"] = args.output_dir
        if args.total_model_calls is not None:
            cfg.setdefault("run", {})["total_model_calls"] = args.total_model_calls
        if args.candidates_per_step is not None:
            cfg.setdefault("run", {})["candidates_per_step"] = args.candidates_per_step
        if args.centers is not None:
            cfg.setdefault("controller", {})["centers"] = args.centers
        print(run_controlled_rollout(cfg))
        return 0
    if args.command == "discovery-analyze-alternate":
        from confmh.discovery.analysis import analyze_alternate_run

        print(analyze_alternate_run(args.run_dir))
        return 0
    if args.command == "discovery-aggregate-alternate":
        from confmh.discovery.analysis import aggregate_alternate_runs

        print(aggregate_alternate_runs(sorted(glob.glob(args.run_glob)), args.output_dir))
        return 0
    if args.command == "discovery-prepare-fastfold":
        from confmh.discovery.fastfold import prepare_fastfold_reference

        print(
            prepare_fastfold_reference(
                topology_pdb=args.topology,
                trajectories=args.trajectories,
                output_path=args.output,
                stride=args.stride,
                lag_frames=args.lag_frames,
                n_components=args.components,
                n_clusters=args.clusters,
                grid_bins=args.grid_bins,
                free_energy_threshold_kcal_mol=args.free_energy_threshold,
                max_frames=args.max_frames,
            )
        )
        return 0
    if args.command == "discovery-analyze-fastfold":
        from confmh.discovery.fastfold import analyze_fastfold_run

        print(
            analyze_fastfold_run(
                run_dir=args.run_dir,
                start_pdb=args.start_pdb,
                reference_model=args.reference_model,
                raw_final_coverage=args.raw_final_coverage,
            )
        )
        return 0
    if args.command == "discovery-fastfold-status":
        import json

        from confmh.discovery.fastfold import fastfold_data_status

        print(json.dumps(fastfold_data_status(args.root), indent=2))
        return 0
    if args.command == "discovery-make-fastfold-configs":
        from confmh.discovery.fastfold import make_fastfold_configs

        paths = make_fastfold_configs(
            data_root=args.data_root,
            output_dir=args.output_dir,
            project_root=args.project_root,
            total_model_calls=args.total_model_calls,
            candidates_per_step=args.candidates_per_step,
        )
        print(f"wrote {len(paths)} configs under {args.output_dir}")
        return 0
    if args.command == "discovery-minimize-hits":
        from confmh.discovery.minimize import hit_candidates_from_csv, minimize_candidates

        candidates = hit_candidates_from_csv(args.metrics_csv, args.hit_column)
        print(minimize_candidates(candidates, args.output_dir, max_iterations=args.max_iterations))
        return 0
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
