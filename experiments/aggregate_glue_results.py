"""Aggregate apps/glue-nlu results across the run_glue_experiment.sh sweep.

For each (task, strategy): group the swept lr values by mean validation accuracy across the 3
seeds (the last logged "eval" phase accuracy per run, since GLUE's real test set is unlabeled --
see glue_nlu/server_app.py's evaluate_fn, which treats validation(_matched) accuracy as the
reported number, exactly as FedRot-LoRA itself does), pick the best lr, then report test accuracy
mean +/- std over the 3 seeds at that lr.

Also reports, per (task, strategy) at the selected best lr: mean total wall-clock (sum of
client_train_seconds + server_aggregate_seconds across all logged rounds, cross-checked against
the shell-measured total from run_glue_experiment.sh), mean total communication bytes (cumulative
upload + download at the final round), and the strategy-specific basis-overlap / quantization-
error metrics where applicable (fedrot_basis_overlap_*, fedora_v1/v2_basis_overlap_mean,
flora_quant_error_frob).

Usage: python aggregate_glue_results.py --results-dir results/ --out experiments/glue_summary.csv
"""

import argparse
import glob
import json
import os

import pandas as pd


def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "run_metadata.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not (os.path.exists(meta_path) and os.path.exists(metrics_path)):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    records = []
    with open(metrics_path) as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        return None

    df = pd.DataFrame(records)
    eval_rows = df[df["phase"] == "eval"]
    if eval_rows.empty:
        return None
    final_accuracy = eval_rows.sort_values("round").iloc[-1]["accuracy"]

    fit_rows = df[df["phase"] == "fit"]
    client_seconds = fit_rows.get("client_train_seconds_mean", pd.Series(dtype=float)).sum()
    server_seconds = fit_rows.get("server_aggregate_seconds", pd.Series(dtype=float)).sum()
    total_bytes = 0
    if not fit_rows.empty:
        last_fit = fit_rows.sort_values("round").iloc[-1]
        total_bytes = int(last_fit.get("cum_upload_bytes", 0)) + int(last_fit.get("cum_download_bytes", 0))

    strategy_metrics = {}
    for col in (
        "fedrot_basis_overlap_pre", "fedrot_basis_overlap_post",
        "fedora_v1_basis_overlap_mean",
        "flora_quant_error_frob",
    ):
        if col in fit_rows.columns and fit_rows[col].notna().any():
            strategy_metrics[col] = fit_rows[col].dropna().mean()

    return {
        "task": meta["task"],
        "strategy": meta["strategy"],
        # Absent in metrics.jsonl logged before this field was added -- defaults to 3 (the main
        # sweep's client count) rather than mis-bucketing older runs as an "unknown" group.
        "client_num": meta.get("client_num", 3),
        "lr": meta["lr"],
        "seed": meta["seed"],
        "accuracy": final_accuracy,
        "total_wall_clock_seconds": client_seconds + server_seconds,
        "total_comm_bytes": total_bytes,
        "gpu_type": meta.get("gpu_type"),
        "gpu_count": meta.get("gpu_count"),
        **strategy_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="experiments/glue_summary.csv")
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(glob.glob(os.path.join(args.results_dir, "*"))):
        run = load_run(run_dir)
        if run is not None:
            runs.append(run)

    if not runs:
        print(f"No completed runs found under {args.results_dir}")
        return

    df = pd.DataFrame(runs)

    # Pick the best lr per (task, strategy, client_num) by mean accuracy across seeds --
    # client_num is part of the grouping key so e.g. a separate N=50 scalability sweep never
    # silently averages together with the main N=3 (or N=10) sweep just because they share the
    # same task/strategy.
    by_lr = df.groupby(["task", "strategy", "client_num", "lr"])["accuracy"].mean().reset_index()
    best_lr = by_lr.loc[by_lr.groupby(["task", "strategy", "client_num"])["accuracy"].idxmax()]

    summary_rows = []
    for _, row in best_lr.iterrows():
        subset = df[
            (df.task == row.task) & (df.strategy == row.strategy)
            & (df.client_num == row.client_num) & (df.lr == row.lr)
        ]
        summary_rows.append(
            {
                "task": row.task,
                "strategy": row.strategy,
                "client_num": row.client_num,
                "best_lr": row.lr,
                "n_seeds": len(subset),
                "accuracy_mean": subset["accuracy"].mean(),
                "accuracy_std": subset["accuracy"].std(ddof=0),
                "total_wall_clock_seconds_mean": subset["total_wall_clock_seconds"].mean(),
                "total_comm_bytes_mean": subset["total_comm_bytes"].mean(),
                "gpu_type": subset["gpu_type"].iloc[0],
                "gpu_count": subset["gpu_count"].iloc[0],
                "fedrot_basis_overlap_pre": subset.get("fedrot_basis_overlap_pre", pd.Series(dtype=float)).mean(),
                "fedrot_basis_overlap_post": subset.get("fedrot_basis_overlap_post", pd.Series(dtype=float)).mean(),
                "fedora_v1_basis_overlap_mean": subset.get("fedora_v1_basis_overlap_mean", pd.Series(dtype=float)).mean(),
                "flora_quant_error_frob": subset.get("flora_quant_error_frob", pd.Series(dtype=float)).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["task", "strategy", "client_num"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
