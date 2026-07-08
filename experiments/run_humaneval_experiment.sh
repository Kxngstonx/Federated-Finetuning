#!/bin/bash
# Reproduces the FedRot-LoRA Llama-3-8B CodeSearchNet/HumanEval experiment protocol against
# apps/llm-humaneval: train on CodeSearchNet (federated, 6 clients = 6 languages, non-IID), eval
# HumanEval pass@1 at each checkpoint-save round. Deviates from FedRot-LoRA's own base_rescale.yaml
# (total_round_num=200, save_freq=40) to shorten the sweep: num-server-rounds=100,
# train.save-every-round=20 -- checkpoints land at rounds 20, 40, 60, 80, and 100-final (5 evals
# total).
#
# GSM8K is NOT evaluated here -- cross-checking the official repo's yamls confirmed GSM8K is a
# separate experiment (its own N=3/IID training run on GSM8K's own training data), see
# run_gsm8k_experiment.sh / apps/llm-gsm8k instead.
#
# Only the fedora strategy, single seed. Learning-rate grid search over
# {5e-4, 1e-3, 5e-3, 2e-2} (train.learning-rate-max), replacing the
# apps/llm-humaneval/pyproject.toml lr=0.005 default for the duration of this sweep.
#
# Usage: ./run_humaneval_experiment.sh [results_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/env.sh

RESULTS_DIR="${1:-results}"
STRATEGIES=(fedora)
SEEDS=(1)
LRS=(5e-4 1e-3 5e-3 2e-2)

START_TIME=$(date +%s)

for strategy in "${STRATEGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
      echo "=== HumanEval strategy=${strategy} seed=${seed} lr=${lr} ==="
      # federate.client-num=6, num-server-rounds=100, 30 local steps
      # (train.training-arguments.max-steps), batch size=1, and LoRA r=8/alpha=16 are all fixed
      # per the experiment protocol and already set as apps/llm-humaneval/pyproject.toml defaults
      # -- only the swept dimensions (strategy/seed/lr) are overridden here. HumanEval eval at
      # each checkpoint-save round is triggered automatically inside
      # llm_humaneval/server_app.py's evaluate_fn.
      flwr run apps/llm-humaneval --run-config "
        strategy.aggregation=\"${strategy}\"
        seed=${seed}
        train.learning-rate-max=${lr}
      "
    done
  done
done

END_TIME=$(date +%s)
echo "Total HumanEval sweep wall-clock (shell-measured, cross-check for the per-run client/server sums in metrics.jsonl): $((END_TIME - START_TIME))s"

python experiments/aggregate_humaneval_results.py --results-dir "${RESULTS_DIR}" --out experiments/humaneval_summary.csv
