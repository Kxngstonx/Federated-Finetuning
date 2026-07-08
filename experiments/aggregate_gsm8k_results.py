"""Aggregate apps/llm-gsm8k results across the run_gsm8k_experiment.sh sweep.

Same shape as aggregate_humaneval_results.py (lr grid over {5e-4, 1e-3, 5e-3, 2e-2},
checkpoint-cadence learning curve since GSM8K eval only runs at train.save-every-round
boundaries), but for the GSM8K-only training run (N=3, IID) rather than the CodeSearchNet/
HumanEval one. For each (strategy, lr): reports GSM8K exact-match mean +/- std over seeds at the
FINAL checkpoint (num-server-rounds), plus the same wall-clock/comm-bytes/basis-overlap/
quant-error summaries as the other aggregate_*.py scripts.

Usage: python aggregate_gsm8k_results.py --results-dir results/ --out experiments/gsm8k_summary.csv
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

    gsm8k_rows = df[df["phase"] == "gsm8k_eval"]
    if gsm8k_rows.empty:
        return None

    final_gsm8k = gsm8k_rows.sort_values("round").iloc[-1]["gsm8k_exact_match"]

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
        "strategy": meta["strategy"],
        "seed": meta["seed"],
        "lr": meta.get("lr"),
        "gsm8k_exact_match": final_gsm8k,
        "total_wall_clock_seconds": client_seconds + server_seconds,
        "total_comm_bytes": total_bytes,
        "gpu_type": meta.get("gpu_type"),
        "gpu_count": meta.get("gpu_count"),
        **strategy_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="experiments/gsm8k_summary.csv")
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

    summary_rows = []
    for (strategy, lr), subset in df.groupby(["strategy", "lr"]):
        summary_rows.append(
            {
                "strategy": strategy,
                "lr": lr,
                "n_seeds": len(subset),
                "gsm8k_exact_match_mean": subset["gsm8k_exact_match"].mean(),
                "gsm8k_exact_match_std": subset["gsm8k_exact_match"].std(ddof=0),
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

    summary = pd.DataFrame(summary_rows).sort_values(["strategy", "lr"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
