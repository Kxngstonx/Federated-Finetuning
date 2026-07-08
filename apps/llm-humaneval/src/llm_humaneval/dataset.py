"""CodeSearchNet training-set loading + language-based partitioning, ported from
FedRot-LoRA/federatedscope/llm/dataset/code_search_net.py's use of `data.splitter='meta'`
(client_num=6, one client per programming language: python/javascript/java/ruby/php/go).

Uses flwr_datasets.NaturalIdPartitioner (fedbench_common/partitioners.py::build_language_
partitioner) partitioned on the "language" column, rather than FedRot-LoRA's own manual
per-language .jsonl.gz directory layout.

NOTE (flagged in the port plan as a verify-at-implementation-time risk): the HF `code_search_net`
dataset uses a script-based loader that HF has deprecated in newer `datasets` versions; if
`datasets.load_dataset("code_search_net", "all")` no longer resolves in the target environment,
fall back to porting FedRot-LoRA's own zip-download loader
(FedRot-LoRA/federatedscope/llm/dataset/code_search_net.py) instead. Column names below
(func_documentation_string / func_code_string / language) match that dataset's schema as
published; verify against the installed `datasets` version.

Training prompt formatting follows FedRot-LoRA's llmtrainer: plain instruction (docstring) ->
response (code) causal-LM SFT, WITHOUT the pos/neg contrastive relabeling FedRot-LoRA's own
generate_eval_files applies -- that relabeling only feeds FedRot-LoRA's own CSN match/mismatch
eval, which is out of scope here (GSM8K exact-match + HumanEval pass@1 only).
"""

from flwr_datasets import FederatedDataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

CSN_LANGUAGES = ["python", "javascript", "java", "ruby", "php", "go"]

_FDS_CACHE: dict = {}


def formatting_prompts_func(example):
    """Alpaca-style instruction(docstring) -> response(code) formatting, analogous to
    flowertune_llm/dataset.py::formatting_prompts_func but for CodeSearchNet's docstring/code
    fields instead of Alpaca's instruction/response fields."""
    output_texts = []
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["func_documentation_string"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['func_documentation_string'][i]}"
            f"\n### Response: {example['func_code_string'][i]}"
        )
        output_texts.append(text)
    return output_texts


def load_data(partition_id: int, num_partitions: int):
    """Load this client's language-partitioned CodeSearchNet training shard. num_partitions is
    expected to be 6 (one client per CSN_LANGUAGES entry), matching FedRot-LoRA's protocol."""
    from fedbench_common.partitioners import build_language_partitioner

    cache_key = num_partitions
    if cache_key not in _FDS_CACHE:
        partitioner = build_language_partitioner("language")
        _FDS_CACHE[cache_key] = FederatedDataset(
            dataset="code_search_net", subset="all", partitioners={"train": partitioner}
        )
    fds = _FDS_CACHE[cache_key]
    return fds.load_partition(partition_id, "train")


def get_tokenizer_and_data_collator(model_name: str):
    """Same response-template-matching scheme as flowertune_llm/dataset.py, applied to this
    module's formatting_prompts_func's "\\n### Response:" tag instead of Alpaca's."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    response_template_ids = tokenizer.encode("\n### Response:", add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return tokenizer, data_collator
