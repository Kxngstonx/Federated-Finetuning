"""GSM8K training-set loading + IID partitioning, ported from
FedRot-LoRA/federatedscope/llm/dataloader/dataloader.py's `gsm8k` branch of `load_llm_dataset`
(lines 280-295 as of the official repo) and its `data.splitter: 'iid'` (client_num=3) protocol,
confirmed against federatedscope/llm/yamls/base_rescale.yaml -- the only yaml in the official repo
with `data.type: 'gsm8k@llm'`, and the only one with client_num=3/IID (every CodeSearchNet yaml
uses client_num=6/`meta` language-split instead). This is a SEPARATE experiment/training run from
apps/llm-humaneval's CodeSearchNet training -- see that app's docstrings for why they were
previously (incorrectly) combined into one run.

Training data: GSM8K's own train.jsonl (7473 examples), downloaded from the same OpenAI
grade-school-math GitHub source apps/llm-gsm8k/eval_gsm8k.py already uses for the *test* split.
Official preprocessing replaces each answer's "####" final-answer delimiter with "The answer is"
(keeping the chain-of-thought reasoning intact, just rewording the delimiter to match the same
phrasing eval_gsm8k.py's few-shot demos and answer-parsing use) -- ported verbatim below. No
`input` field is set for GSM8K (unlike CodeSearchNet's docstring+language input), so the Alpaca
`prompt_no_input` template applies, which is exactly the template already used by
apps/llm-humaneval/src/llm_humaneval/dataset.py::formatting_prompts_func (no "### Input:" section).
"""

import os

import requests
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/train.jsonl"
)

_DATASET_CACHE: dict = {}


def formatting_prompts_func(example):
    """Alpaca `prompt_no_input`-style instruction(question) -> response(answer) formatting,
    identical template convention to llm_humaneval/dataset.py::formatting_prompts_func."""
    output_texts = []
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["question"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['question'][i]}"
            f"\n### Response: {example['answer'][i]}"
        )
        output_texts.append(text)
    return output_texts


def download_gsm8k_train(data_root: str) -> str:
    fp = os.path.join(data_root, "gsm8k_train.jsonl")
    if not os.path.exists(fp):
        os.makedirs(data_root, exist_ok=True)
        resp = requests.get(GSM8K_TRAIN_URL, timeout=60)
        resp.raise_for_status()
        with open(fp, "wb") as f:
            f.write(resp.content)
    return fp


def _load_gsm8k_train_dataset(data_root: str, seed: int) -> Dataset:
    import json

    fp = download_gsm8k_train(data_root)
    records = []
    with open(fp) as f:
        for line in f:
            record = json.loads(line)
            # FedRot-LoRA's own preprocessing (dataloader.py line 292-293): keep the CoT
            # reasoning, just reword the final-answer delimiter to match the eval prompt phrasing.
            answer = record["answer"].replace("####", "The answer is")
            records.append({"question": record["question"], "answer": answer})
    # IidPartitioner.load_partition does a contiguous dataset.shard with no shuffling of its own
    # (see fedbench_common.partitioners.build_iid_partitioner's docstring) -- shuffle here so the
    # 3 client partitions are genuinely IID rather than contiguous slices of train.jsonl's order.
    return Dataset.from_list(records).shuffle(seed=seed)


def load_data(partition_id: int, num_partitions: int, data_root: str, seed: int):
    """Load this client's IID-partitioned GSM8K training shard. num_partitions is expected to be
    3, matching FedRot-LoRA's GSM8K protocol (federate.client_num=3, data.splitter='iid')."""
    from fedbench_common.partitioners import build_iid_partitioner

    cache_key = (num_partitions, data_root, seed)
    if cache_key not in _DATASET_CACHE:
        partitioner = build_iid_partitioner(num_partitions)
        partitioner.dataset = _load_gsm8k_train_dataset(data_root, seed)
        _DATASET_CACHE[cache_key] = partitioner
    partitioner = _DATASET_CACHE[cache_key]
    return partitioner.load_partition(partition_id)


def get_tokenizer_and_data_collator(model_name: str):
    """Same response-template-matching scheme as llm_humaneval/dataset.py, applied to this
    module's formatting_prompts_func's "\\n### Response:" tag."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    response_template_ids = tokenizer.encode("\n### Response:", add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return tokenizer, data_collator
