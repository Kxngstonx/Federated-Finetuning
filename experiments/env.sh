#!/bin/bash
# Redirects everything storage-heavy (HF model/dataset cache, GSM8K/HumanEval raw data, and
# experiment results/checkpoints) onto the shared NVMe mount at /mnt/data1 instead of the root
# disk (which was down to ~8GB free). Sourced by run_glue_experiment.sh / run_nlg_experiment.sh;
# source it manually (`source experiments/env.sh`) before any ad hoc `flwr run` invocation too.

export FEDORA_ROOT="/mnt/data1/hmkang/fedora"

# HuggingFace model + dataset cache (transformers, datasets, flwr_datasets all respect these).
# (TRANSFORMERS_CACHE is deprecated as of transformers>=5 in favor of HF_HOME alone -- omitted to
# avoid the FutureWarning; HF_HOME/hub is used automatically.)
export HF_HOME="${FEDORA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${FEDORA_ROOT}/hf_cache/datasets"

# GSM8K/HumanEval raw eval data downloads (eval_gsm8k.py / eval_humaneval.py --data-root default).
export FEDORA_DATA_ROOT="${FEDORA_ROOT}/data"

# pip's own download/wheel cache -- also redirected off the root disk (torch/transformers/etc.
# wheels add up to several GB).
export PIP_CACHE_DIR="${FEDORA_ROOT}/pip_cache"

# Flower's simulation backend runs on Ray, which defaults its own session/log/object-spill
# directory to /tmp/ray -- on the root disk (8-9GB free), NOT covered by any of the redirects
# above. Ray fills this fast during a real run (raylet warns "over 95% full" almost immediately)
# and object creation fails outright once it's exhausted. Ray honors RAY_TMPDIR for this.
export RAY_TMPDIR="${FEDORA_ROOT}/ray_tmp"

# meta-llama/Meta-Llama-3-8B is gated -- HF_TOKEN needed to download it. Read from ~/.hf_token
# (outside the repo, chmod 600) rather than hardcoding the value here, since this file IS tracked
# by git -- a literal token here would leak into git history the moment it's committed.
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN
  HF_TOKEN=$(cat "$HOME/.hf_token")
fi

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${FEDORA_DATA_ROOT}" "${PIP_CACHE_DIR}" "${RAY_TMPDIR}"

# /usr/local/bin/python3.11 was linked against libffi.so.6, but this host (Ubuntu 20.04/focal)
# only ships libffi.so.7 -- ctypes (and anything depending on it: protobuf, scipy, etc.) fails to
# import without it. libffi.so.7 is ABI-compatible enough in practice (another user on this box
# already worked around the same issue the same way, see /mnt/data1/namil_work/anaconda3/pkgs/
# libffi-3.3-he6710b0_2/lib/libffi.so.6 -- also just a symlink to .so.7.1.0). Point our own
# compat symlink dir at it via LD_LIBRARY_PATH rather than relying on that other user's directory.
mkdir -p "${FEDORA_ROOT}/lib_compat"
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 "${FEDORA_ROOT}/lib_compat/libffi.so.6"
export LD_LIBRARY_PATH="${FEDORA_ROOT}/lib_compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# The project venv (python3.11, since flwr/torch/peft all require Python >=3.10 but this host's
# default `python3`/`pip` is 3.8) lives on NVMe too -- auto-activate it here if it exists, so
# sourcing this file is the one thing every script/shell needs to do.
FEDORA_VENV="${FEDORA_ROOT}/venv"
if [ -f "${FEDORA_VENV}/bin/activate" ]; then
  source "${FEDORA_VENV}/bin/activate"
fi
