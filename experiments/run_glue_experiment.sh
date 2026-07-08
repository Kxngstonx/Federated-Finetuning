#!/bin/bash
# Reproduces the FedRot-LoRA GLUE/RoBERTa-Large NLU experiment protocol against apps/glue-nlu:
#   5 GLUE tasks x 8 strategies x 4-point lr grid x 3 seeds
#   N=3 clients, LoRA r=4, Dirichlet alpha=0.5, 250 rounds, 20 local steps, batch size 128.
# All 7 registered strategies are compared (fedavg, fedit, fedora, fedsvd, ffalora,
# flora, fedrot), per the confirmed experiment scope.
#
# Usage: ./run_glue_experiment.sh [results_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/env.sh

RESULTS_DIR="${1:-results}"
TASKS=(sst2 qnli mnli qqp rte)
STRATEGIES=(fedavg fedit fedora fedsvd ffalora flora fedrot)
LRS=(5e-4 1e-3 5e-3 2e-2)
SEEDS=(1 2 3)

START_TIME=$(date +%s)

for task in "${TASKS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for lr in "${LRS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "=== GLUE task=${task} strategy=${strategy} lr=${lr} seed=${seed} ==="
        # federate.client-num=3, num-server-rounds=250, dataset.dirichlet-alpha=0.5, LoRA r=4,
        # batch size=128, and 20 local steps (train.training-arguments.max-steps) are all fixed
        # per the experiment protocol and already set as apps/glue-nlu/pyproject.toml defaults --
        # only the swept dimensions (task/strategy/lr/seed) are overridden here.
        flwr run apps/glue-nlu --run-config "
          dataset.task-name=\"${task}\"
          strategy.aggregation=\"${strategy}\"
          train.learning-rate-max=${lr}
          seed=${seed}
        "
      done
    done
  done
done

END_TIME=$(date +%s)
echo "Total GLUE sweep wall-clock (shell-measured, cross-check for the per-run client/server sums in metrics.jsonl): $((END_TIME - START_TIME))s"

python experiments/aggregate_glue_results.py --results-dir "${RESULTS_DIR}" --out experiments/glue_summary.csv
