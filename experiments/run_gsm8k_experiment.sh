#!/bin/bash
# Reproduces the FedRot-LoRA Llama-3-8B GSM8K experiment protocol against apps/llm-gsm8k: train on
# GSM8K's own training set (federated, 3 clients, IID split), eval GSM8K exact-match at each
# checkpoint-save round. Deviates from FedRot-LoRA's own base_rescale.yaml (total_round_num=100,
# save_freq=40) to shorten the sweep: num-server-rounds=50, train.save-every-round=10 --
# checkpoints land at rounds 10, 20, 30, 40, and 50-final (5 evals total).
#
# This is a SEPARATE experiment from apps/llm-humaneval's CodeSearchNet/HumanEval pipeline --
# cross-checking the official repo's yamls (only federatedscope/llm/yamls/base_rescale.yaml has
# `data.type: gsm8k@llm`, with client_num=3/splitter=iid/total_round_num=100, vs every
# CodeSearchNet yaml's client_num=6/splitter=meta/total_round_num=200) confirmed GSM8K is trained
# and evaluated on its own, not off the CodeSearchNet checkpoint.
#
# Only the fedora strategy, single seed. Learning-rate grid search over
# {5e-4, 1e-3, 5e-3, 2e-2} (train.learning-rate-max), replacing the apps/llm-gsm8k/pyproject.toml
# lr=0.005 default for the duration of this sweep.
#
# Usage: ./run_gsm8k_experiment.sh [results_dir]
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
      echo "=== GSM8K strategy=${strategy} seed=${seed} lr=${lr} ==="
      # federate.client-num=3, num-server-rounds=50, 30 local steps
      # (train.training-arguments.max-steps), batch size=1, and LoRA r=8/alpha=16 are all fixed
      # per the experiment protocol and already set as apps/llm-gsm8k/pyproject.toml defaults --
      # only the swept dimensions (strategy/seed/lr) are overridden here. GSM8K eval at each
      # checkpoint-save round is triggered automatically inside llm_gsm8k/server_app.py's
      # evaluate_fn.
      flwr run apps/llm-gsm8k --run-config "
        strategy.aggregation=\"${strategy}\"
        seed=${seed}
        train.learning-rate-max=${lr}
      "
    done
  done
done

END_TIME=$(date +%s)
echo "Total GSM8K sweep wall-clock (shell-measured, cross-check for the per-run client/server sums in metrics.jsonl): $((END_TIME - START_TIME))s"

python experiments/aggregate_gsm8k_results.py --results-dir "${RESULTS_DIR}" --out experiments/gsm8k_summary.csv
