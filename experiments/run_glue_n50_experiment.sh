#!/bin/bash
# GLUE/RoBERTa-Large scalability run at N=50 clients against apps/glue-nlu -- NOT part of the
# official FedRot-LoRA repo (grep-confirmed: no GLUE or LLM yaml in FedRot-LoRA ever sets
# client_num=50; the only "50" hit anywhere in that repo is an unrelated generic FederatedScope
# NLP demo config, federatedscope/nlp/baseline/fedavg_transformer_on_cola.yaml). This is the
# user's own scalability ablation, run in the same reduced-scope style as run_gsm8k_experiment.sh:
# single strategy (fedora), single seed, lr grid search over {5e-4, 1e-3, 5e-3, 2e-2}. LoRA r=4 and
# Dirichlet alpha=0.5 are NOT swept (already the apps/glue-nlu/pyproject.toml defaults, unchanged).
#
# N=50 needs TWO separate overrides at invocation time, not just one:
#   --run-config federate.client-num=50       -- cosmetic/logging only (see server_app.py's
#                                                 write_run_metadata comment); nothing in the
#                                                 client_fn code path actually reads this key.
#   --federation-config options.num-supernodes=50 -- this is what actually controls how many
#                                                 Flower client processes the simulator spins up
#                                                 (client_fn reads context.node_config["num-
#                                                 partitions"], which Flower derives from this).
# Both must be kept in sync or the logged client_num in run_metadata.json will misreport how many
# clients actually ran.
#
# GPU sharing: apps/glue-nlu/pyproject.toml's default options.backend.client-resources.num-gpus=1.0
# assumes N=3 (<=2 clients need to share 2 GPUs at once). At N=50 that would force 25 sequential
# waves of 2 clients per round. RoBERTa-Large (355M, unquantized) is small enough that many clients
# comfortably fit on one 48GB GPU at once, so this script overrides num-gpus down to 0.1 (10
# clients/GPU, 20 concurrent across 2 GPUs -> ~3 waves/round instead of 25). Adjust if this proves
# too aggressive (OOM) or too conservative (still underutilized) once the first round actually runs.
#
# Usage: ./run_glue_n50_experiment.sh [results_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source experiments/env.sh

RESULTS_DIR="${1:-results}"
TASKS=(sst2 qnli mnli qqp rte)
STRATEGIES=(fedora)
LRS=(5e-4 1e-3 5e-3 2e-2)
SEEDS=(1)
N_CLIENTS=50
NUM_GPUS_PER_CLIENT=0.1

START_TIME=$(date +%s)

for task in "${TASKS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for lr in "${LRS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "=== GLUE-N50 task=${task} strategy=${strategy} lr=${lr} seed=${seed} ==="
        # LoRA r=4, Dirichlet alpha=0.5, 250 rounds, 20 local steps, batch size=128 are all fixed
        # per the main protocol and already set as apps/glue-nlu/pyproject.toml defaults -- only
        # the swept dimensions (task/lr) plus the N=50 client-count overrides are set here.
        flwr run apps/glue-nlu --run-config "
          dataset.task-name=\"${task}\"
          strategy.aggregation=\"${strategy}\"
          train.learning-rate-max=${lr}
          seed=${seed}
          federate.client-num=${N_CLIENTS}
        " --federation-config "
          options.num-supernodes=${N_CLIENTS}
          options.backend.client-resources.num-gpus=${NUM_GPUS_PER_CLIENT}
        "
      done
    done
  done
done

END_TIME=$(date +%s)
echo "Total GLUE-N50 sweep wall-clock (shell-measured, cross-check for the per-run client/server sums in metrics.jsonl): $((END_TIME - START_TIME))s"

python experiments/aggregate_glue_results.py --results-dir "${RESULTS_DIR}" --out experiments/glue_n50_summary.csv
