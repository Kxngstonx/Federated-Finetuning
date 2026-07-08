"""Non-IID partitioner builders for the GLUE (Dirichlet) and CodeSearchNet (language/category)
experiment pipelines.

GLUE: ported from FedRot-LoRA/federatedscope/core/splitters/generic/lda_splitter.py's use of
`data.splitter: 'lda'`, `data.splitter_args: [{'alpha': 0.5}]` -- but using flwr-datasets' own
DirichletPartitioner rather than hand-rolling the dirichlet_distribution_noniid_slice logic.

CodeSearchNet: ported from FedRot-LoRA's `data.splitter: 'meta'` (category-based split, one
client per programming language) via flwr-datasets' NaturalIdPartitioner.
"""

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, NaturalIdPartitioner


def build_dirichlet_partitioner(
    num_partitions: int,
    label_column: str,
    alpha: float,
    seed: int,
) -> DirichletPartitioner:
    """Dirichlet non-IID partitioner for GLUE tasks (label_column='label').

    min_partition_size=1 (rather than DirichletPartitioner's default of 10) to tolerate small
    GLUE tasks like RTE (~2.5k train examples) split across only 3 clients at alpha=0.5, which
    FedRot-LoRA's own LDASplitter did not guard against either.
    """
    return DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=label_column,
        alpha=alpha,
        min_partition_size=1,
        self_balancing=True,
        shuffle=True,
        seed=seed,
    )


def build_language_partitioner(category_column: str) -> NaturalIdPartitioner:
    """Category-based partitioner for CodeSearchNet: one client per distinct value of
    `category_column` (e.g. programming language: python/javascript/java/ruby/php/go), matching
    FedRot-LoRA's `splitter='meta'` protocol (client_num=6, one language per client)."""
    return NaturalIdPartitioner(partition_by=category_column)


def build_iid_partitioner(num_partitions: int) -> IidPartitioner:
    """IID partitioner for GSM8K training data, matching FedRot-LoRA's `data.splitter: 'iid'`
    (client_num=3) -- confirmed via federatedscope/llm/yamls/base_rescale.yaml, the only yaml in
    the official repo with `data.type: 'gsm8k@llm'`, as the shared base config for every
    strategy's separate GSM8K experiment (distinct from the CodeSearchNet/HumanEval experiment's
    N=6 language-split non-IID setup).

    Unlike build_language_partitioner/build_dirichlet_partitioner, this isn't wired up via a HF
    Hub `flwr_datasets.FederatedDataset` -- GSM8K's own train.jsonl is downloaded directly (same
    raw-GitHub convention as eval_gsm8k.py's download_gsm8k_test) and wrapped in a plain
    datasets.Dataset, so callers assign that Dataset directly to the returned partitioner's
    `.dataset` property before calling `.load_partition(partition_id)` -- flwr_datasets
    partitioners work against any datasets.Dataset, not just Hub-loaded ones.

    NOTE: IidPartitioner.load_partition just does a contiguous `dataset.shard(...)` with no
    shuffling of its own (confirmed against the installed flwr_datasets source) -- callers MUST
    call `.shuffle(seed=...)` on the Dataset themselves before assigning it to `.dataset`, or the
    "IID" partitions will actually be contiguous slices of whatever order train.jsonl was in.
    """
    return IidPartitioner(num_partitions=num_partitions)
