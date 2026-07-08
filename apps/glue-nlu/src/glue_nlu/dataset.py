"""GLUE dataset loading + Dirichlet non-IID partitioning, ported from
FedRot-LoRA/federatedscope/glue/dataloader/dataloader.py.

Dirichlet partitioning replaces FedRot-LoRA's own LDASplitter
(core/splitters/generic/lda_splitter.py) with flwr_datasets.partitioner.DirichletPartitioner
(see fedbench_common/partitioners.py) -- same alpha-parametrized non-IID label distribution.

MNLI is hardcoded matched-only (config.data.matched=True in every FedRot-LoRA yaml); the other 4
tasks have no matched/mismatched split. Centralized (non-partitioned) validation is used for the
reported accuracy, mirroring FedRot-LoRA's `federate.make_global_eval: True` -- GLUE's real test
split is unlabeled, so validation _is_ the reported "test accuracy" number here, exactly as
FedRot-LoRA treats it.
"""

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from transformers import AutoTokenizer

# Ported verbatim from FedRot-LoRA/federatedscope/glue/dataloader/dataloader.py's task_to_keys,
# restricted to the 5 tasks in scope for this experiment.
TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
}

_FDS_CACHE: dict = {}  # keyed by (task_name, num_partitions, alpha, seed) -- one per config


def get_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _preprocess_fn(tokenizer, task_name: str, seq_length: int):
    key1, key2 = TASK_TO_KEYS[task_name]

    def _fn(examples):
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(*args, padding="max_length", max_length=seq_length, truncation=True)

    return _fn


def load_data(
    partition_id: int,
    num_partitions: int,
    task_name: str,
    tokenizer: AutoTokenizer,
    seq_length: int,
    alpha: float,
    seed: int,
):
    """Load this client's Dirichlet-partitioned, tokenized GLUE training shard."""
    from fedbench_common.partitioners import build_dirichlet_partitioner

    cache_key = (task_name, num_partitions, alpha, seed)
    if cache_key not in _FDS_CACHE:
        partitioner = build_dirichlet_partitioner(num_partitions, "label", alpha, seed)
        _FDS_CACHE[cache_key] = FederatedDataset(
            dataset="glue", subset=task_name, partitioners={"train": partitioner}
        )
    fds = _FDS_CACHE[cache_key]
    partition = fds.load_partition(partition_id, "train")
    return partition.map(_preprocess_fn(tokenizer, task_name, seq_length), batched=True)


def load_centralized_validation(task_name: str, tokenizer: AutoTokenizer, seq_length: int):
    """Full, un-partitioned validation(_matched) split, used for the server's centralized
    evaluate_fn -- matches FedRot-LoRA's federate.make_global_eval=True protocol."""
    datasets = load_dataset("glue", task_name)
    split = "validation_matched" if task_name == "mnli" else "validation"
    eval_dataset = datasets[split]
    return eval_dataset.map(_preprocess_fn(tokenizer, task_name, seq_length), batched=True)
