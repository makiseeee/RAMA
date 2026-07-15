#!/usr/bin/env python3
"""Run additional SIMS-v2 one-factor sensitivity sweeps in priority order.

This is a thin scheduler around the existing sensitivity script. It does not
change training code; it calls the original script once per hyperparameter value
so each run can resume independently through the original CSV/run_tag logic.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SweepJob:
    group: str
    sweep_param: str
    sweep_value: str
    reason: str


# Ordered from most useful for the current 3-panel figure to less urgent.
PRIORITY_PLAN: list[SweepJob] = [
    # Gamma/cib_scale: current real x-axis has dense left side and sparse right side.
    SweepJob("gamma-core", "cib_scale", "0.25", "fills 0.20 -> 0.30"),
    SweepJob("gamma-core", "cib_scale", "0.35", "fills 0.30 -> 0.50"),
    SweepJob("gamma-core", "cib_scale", "0.40", "fills 0.30 -> 0.50"),
    SweepJob("gamma-core", "cib_scale", "0.45", "fills 0.30 -> 0.50"),
    # Beta: make the peak neighborhood around beta=1.2 smoother.
    SweepJob("beta-core", "beta", "1.10", "left neighbor of 1.20"),
    SweepJob("beta-core", "beta", "1.30", "right neighbor of 1.20"),
    SweepJob("beta-core", "beta", "1.40", "fills 1.20 -> 1.50"),
    SweepJob("beta-core", "beta", "0.90", "fills 0.70 -> 1.00"),
    # K: reduce the large gaps in 4/8/16/32.
    SweepJob("k-core", "pseudo_tokens", "12", "fills 8 -> 16"),
    SweepJob("k-core", "pseudo_tokens", "24", "fills 16 -> 32"),
    # Lower-priority extra smoothing if time is available.
    SweepJob("gamma-extra", "cib_scale", "0.12", "near gamma optimum left shoulder"),
    SweepJob("gamma-extra", "cib_scale", "0.18", "near gamma optimum right shoulder"),
    SweepJob("beta-extra", "beta", "0.20", "smooth low-beta region"),
    SweepJob("beta-extra", "beta", "0.40", "smooth low-beta region"),
    SweepJob("beta-extra", "beta", "0.60", "smooth mid-beta region"),
    SweepJob("k-extra", "pseudo_tokens", "20", "fills 16 -> 24"),
    SweepJob("k-extra", "pseudo_tokens", "28", "fills 24 -> 32"),
]


def parse_csv_list(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


def sanitize_value(value: str) -> str:
    """Match swp.py's run_tag value formatting for scheduler result lookup."""
    text = f"{float(value):g}" if "." in value else str(value)
    return text.replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "")


def summary_path_for_seed(args: argparse.Namespace, seed: int) -> str:
    return os.path.join(args.res_save_dir, f"hparam_sensitivity_simsv2_seed{seed}_full.csv")


def read_metric_for_run(
    summary_path: str,
    run_tag: str,
    metric: str,
) -> float | None:
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in reversed(rows):
        if row.get("run_tag") != run_tag:
            continue
        if row.get("error"):
            return None
        value = row.get(metric)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def metric_is_acceptable(value: float | None, args: argparse.Namespace) -> bool:
    if value is None:
        return False
    if args.accept_min_metric is not None and value < args.accept_min_metric:
        return False
    if args.accept_max_metric is not None and value > args.accept_max_metric:
        return False
    return True


def filter_jobs(args: argparse.Namespace) -> list[SweepJob]:
    groups = parse_csv_list(args.groups)
    params = parse_csv_list(args.params)
    values = parse_csv_list(args.values)

    jobs = PRIORITY_PLAN
    if args.core_only:
        jobs = [job for job in jobs if job.group.endswith("-core")]
    if groups is not None:
        jobs = [job for job in jobs if job.group in groups]
    if params is not None:
        jobs = [job for job in jobs if job.sweep_param in params]
    if values is not None:
        jobs = [job for job in jobs if job.sweep_value in values]
    return jobs


def build_base_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        args.script,
        "--modelName",
        args.model_name,
        "--root_dataset_dir",
        args.root_dataset_dir,
        "--model_save_dir",
        args.model_save_dir,
        "--res_save_dir",
        args.res_save_dir,
        "--log_dir",
        args.log_dir,
        "--pretrain_LM",
        args.pretrain_lm,
        "--num_workers",
        str(args.num_workers),
    ]

    if args.gpu_ids:
        cmd.extend(["--gpu_ids", *[str(gpu_id) for gpu_id in args.gpu_ids]])
    if args.tune_mode:
        cmd.append("--tune_mode")
    if args.is_tune:
        cmd.append("--is_tune")
    if args.no_resume:
        cmd.append("--no_resume")
    return cmd


def build_command_for_job(args: argparse.Namespace, job: SweepJob, seed: int) -> list[str]:
    cmd = build_base_command(args)
    cmd.extend(
        [
            "--seed",
            str(seed),
            "--sweep_param",
            job.sweep_param,
            "--sweep_values",
            job.sweep_value,
        ]
    )
    return cmd


def run_one_attempt(args: argparse.Namespace, job: SweepJob, seed: int) -> float | None:
    cmd = build_command_for_job(args, job, seed)
    run_tag = f"seed{seed}_{job.sweep_param}_{sanitize_value(job.sweep_value)}"

    print(f"seed={seed} run_tag={run_tag}")
    print(" ".join(cmd))

    if args.dry_run:
        return None

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Run failed with exit code {completed.returncode}: "
            f"{job.sweep_param}={job.sweep_value}, seed={seed}"
        )

    metric_value = read_metric_for_run(summary_path_for_seed(args, seed), run_tag, args.metric)
    if metric_value is None:
        print(f"[WARN] Could not read {args.metric} for {run_tag}; treating as not accepted.")
    else:
        print(f"[Result] {args.metric}={metric_value:.6f}")
    return metric_value


def run_jobs(args: argparse.Namespace, jobs: Iterable[SweepJob]) -> None:
    os.makedirs(args.res_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    jobs = list(jobs)
    if not jobs:
        print("No jobs selected.")
        return

    print(f"Selected {len(jobs)} sweep values.")
    print(f"Candidate seeds per point: {', '.join(str(seed) for seed in args.seeds)}")
    print(f"Acceptance metric: {args.metric}")
    if args.accept_min_metric is not None:
        print(f"Accept if {args.metric} >= {args.accept_min_metric}")
    if args.accept_max_metric is not None:
        print(f"Accept if {args.metric} <= {args.accept_max_metric}")

    accepted: list[tuple[SweepJob, int, float | None]] = []
    rejected: list[tuple[SweepJob, list[tuple[int, float | None]]]] = []

    for idx, job in enumerate(jobs, start=1):
        print(
            f"\n[{idx}/{len(jobs)}] {job.sweep_param}={job.sweep_value} "
            f"({job.group}; {job.reason})"
        )

        attempts: list[tuple[int, float | None]] = []
        accepted_this_job = False
        for seed in args.seeds:
            metric_value = run_one_attempt(args, job, seed)
            attempts.append((seed, metric_value))

            if args.dry_run:
                accepted_this_job = True
                break

            if metric_is_acceptable(metric_value, args):
                print(f"[Accepted] {job.sweep_param}={job.sweep_value} with seed={seed}")
                accepted.append((job, seed, metric_value))
                accepted_this_job = True
                break

            print(f"[Retry] {job.sweep_param}={job.sweep_value} not accepted with seed={seed}.")

        if not accepted_this_job:
            rejected.append((job, attempts))
            print(f"[Not accepted] Exhausted seeds for {job.sweep_param}={job.sweep_value}.")

    if args.dry_run:
        return

    print("\n=== Accepted ===")
    for job, seed, metric_value in accepted:
        metric_text = "NA" if metric_value is None else f"{metric_value:.6f}"
        print(f"{job.sweep_param}={job.sweep_value} seed={seed} {args.metric}={metric_text}")

    if rejected:
        print("\n=== Needs more seeds or lower threshold ===")
        for job, attempts in rejected:
            attempt_text = ", ".join(
                f"seed={seed}:{'NA' if value is None else f'{value:.6f}'}"
                for seed, value in attempts
            )
            print(f"{job.sweep_param}={job.sweep_value} attempts=[{attempt_text}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Priority scheduler for extra SIMS-v2 sensitivity sweeps."
    )
    parser.add_argument(
        "--script",
        type=str,
        default="swp.py",
        help="Path to the original sensitivity script.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1111, 2222, 3333, 4444, 5555],
        help="Candidate seeds for each point. The scheduler stops at the first acceptable result.",
    )
    parser.add_argument("--core_only", action="store_true", help="Run only core figure-smoothing jobs.")
    parser.add_argument("--groups", type=str, default=None, help="Comma list, e.g. gamma-core,beta-core.")
    parser.add_argument("--params", type=str, default=None, help="Comma list, e.g. cib_scale,beta.")
    parser.add_argument("--values", type=str, default=None, help="Comma list of selected values.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--metric", type=str, default="F1_score")
    parser.add_argument(
        "--accept_min_metric",
        type=float,
        default=0.82,
        help="Retry with the next seed until metric is at least this value. Use none by omitting in code or setting a very low value.",
    )
    parser.add_argument(
        "--accept_max_metric",
        type=float,
        default=0.8462045086117568,
        help="Optional upper bound for accepted metric values.",
    )

    parser.add_argument("--model_name", type=str, default="cmcm")
    parser.add_argument("--root_dataset_dir", type=str, default="/home/oydq/dataset/Dateset")
    parser.add_argument("--model_save_dir", type=str, default="results/models_hparam_sensitivity_extra")
    parser.add_argument("--res_save_dir", type=str, default="results/hparam_sensitivity_extra")
    parser.add_argument("--log_dir", type=str, default="logs_extra_hparam")
    parser.add_argument("--pretrain_lm", type=str, default="/home/oydq/chatglm3-6b-base")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=[])
    parser.add_argument("--tune_mode", action="store_true")
    parser.add_argument("--is_tune", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = filter_jobs(args)
    run_jobs(args, jobs)


if __name__ == "__main__":
    main()
