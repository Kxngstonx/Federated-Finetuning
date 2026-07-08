"""HumanEval pass@1 evaluation, ported from
FedRot-LoRA/federatedscope/llm/eval/eval_for_code/humaneval.py.

`clean_answer` and the generation config are kept verbatim. FSChatBot is replaced with a plain
transformers.AutoModelForCausalLM + peft.PeftModel checkpoint load; FederatedScope's download_url
is replaced with a plain `requests.get`. Scoring is delegated to the `human-eval` pip package's
`evaluate_functional_correctness`.

SECURITY NOTE (explicitly confirmed acceptable for this environment): human-eval's execution
harness runs arbitrary LLM-generated code in a subprocess, and ships with code execution disabled
by default as a safety gate (see human_eval/execution.py's `check_correctness`, which contains a
commented-out `exec(...)` line the package's own README says must be manually uncommented). This
module's `enable_code_execution()` performs that same one-line patch programmatically against the
installed package on disk. Only call it if you understand and accept that this permits running
untrusted, LLM-generated code (each sample is still subprocess-isolated with a timeout by the
package itself).

Usage: python eval_humaneval.py --base-model meta-llama/Meta-Llama-3-8B --peft-path results/<ts>/peft_<round> \
    --round 200 --metrics-out results/<ts>/metrics.jsonl
"""

import argparse
import gzip
import json
import os

import requests
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

NUM_ANSWERS_PER_QUESTION = 5

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/"
    "463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz"
)


def clean_answer(code: str) -> str:
    """Borrowed from https://github.com/FSoft-AI4Code/CodeCapybara, ported verbatim."""

    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    code = code.replace(" ", "")
    for stop_seq in ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\nassert"]:
        code = code.split(stop_seq)[0]
    return pad_spaces(code, 4)


def enable_code_execution() -> None:
    """Uncomments human_eval/execution.py's guarded `exec(...)` call in the installed package, so
    `evaluate_functional_correctness` can actually run generated code and score pass@k. Idempotent
    -- safe to call even if already enabled. Requires the `human-eval` package to be installed."""
    import human_eval.execution as execution_module

    path = execution_module.__file__
    with open(path) as f:
        source = f.read()

    guarded = "# exec(check_program, exec_globals)"
    enabled = "exec(check_program, exec_globals)"
    if guarded in source:
        with open(path, "w") as f:
            f.write(source.replace(guarded, enabled))
        print(f"human-eval code execution enabled in {path}")
    elif enabled in source:
        print("human-eval code execution already enabled")
    else:
        raise RuntimeError(
            f"Could not find the expected guarded exec(...) line in {path}; the installed "
            "human-eval package's execution.py may differ from the version this port targets -- "
            "inspect it manually before proceeding."
        )


def download_humaneval(data_root: str) -> str:
    fp = os.path.join(data_root, "HumanEval.jsonl.gz")
    if not os.path.exists(fp):
        os.makedirs(data_root, exist_ok=True)
        resp = requests.get(HUMANEVAL_URL, timeout=60)
        resp.raise_for_status()
        with open(fp, "wb") as f:
            f.write(resp.content)
    return fp


def load_humaneval_problems(fp: str):
    data = []
    with gzip.open(fp, "rt") as f:
        for line in f:
            record = json.loads(line)
            data.append({"instruction": record["prompt"], "category": record["task_id"]})
    return data


def load_model(base_model: str, peft_path: str, device: str = "cuda:0"):
    """device_map pinned to a single GPU (not "auto") -- see eval_gsm8k.py::load_model's
    docstring for why: "auto" naively pipeline-splits an 8B model's layers across every visible
    GPU, which for autoregressive generation ping-pongs compute between GPUs one token at a time
    instead of actually running in parallel, and this model fits on one 48GB GPU regardless.

    float16, not bfloat16 -- see eval_gsm8k.py::load_model's docstring for why: this fleet's
    Quadro RTX 8000s are Turing (SM75), which has no bf16 Tensor Core path, so bf16 matmuls fall
    back to an unaccelerated path there. fp16 has full Tensor Core support on Turing."""
    device_map = {"": device} if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map=device_map)
    model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    # Left padding, not the default right padding: for batched decoder-only generation every
    # sequence in a batch must have its prompt end at the same index so a single slice
    # (output_ids[:, inputs["input_ids"].shape[1]:]) recovers each row's completion. Right padding
    # would put trailing pad tokens *after* short prompts, misaligning that shared cutoff.
    tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def generate_samples(
    base_model: str, peft_path: str, data_root: str, samples_out: str, device: str = "cuda:0",
    batch_size: int = 4, shard_index: int = 0, num_shards: int = 1,
) -> None:
    model, tokenizer = load_model(base_model, peft_path, device)
    fp = download_humaneval(data_root)
    list_data_dict = load_humaneval_problems(fp)
    # Interleaved (stride) sharding rather than contiguous chunks: balances each shard's mix of
    # short/long problems more evenly than splitting the file in half. See server_app.py's
    # evaluate_fn, which launches one of these per GPU and concatenates the resulting samples_out
    # files before scoring once (score_pass_at_1 doesn't care which subset of task_ids a samples
    # file covers, since it scores whatever appears in the file against the full problem set).
    list_data_dict = list_data_dict[shard_index::num_shards]

    generation_config = GenerationConfig(
        temperature=0.1, top_k=40, top_p=0.75, do_sample=True,
        num_return_sequences=NUM_ANSWERS_PER_QUESTION,
    )

    with open(samples_out, "w") as f:
        for start in tqdm(range(0, len(list_data_dict), batch_size), desc=f"humaneval[{shard_index}/{num_shards}]"):
            batch = list_data_dict[start:start + batch_size]
            input_texts = [sample["instruction"] for sample in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
            try:
                output_ids = model.generate(
                    **inputs, generation_config=generation_config, max_new_tokens=128
                )
                # generate() expands each input by num_return_sequences copies, consecutively:
                # rows [0:NUM_ANSWERS_PER_QUESTION) belong to batch[0], the next
                # NUM_ANSWERS_PER_QUESTION rows to batch[1], etc.
                completions = tokenizer.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            except torch.cuda.OutOfMemoryError as error:
                print(error)
                completions = ["" for _ in range(len(batch) * NUM_ANSWERS_PER_QUESTION)]

            for i, sample in enumerate(batch):
                sample_completions = completions[
                    i * NUM_ANSWERS_PER_QUESTION:(i + 1) * NUM_ANSWERS_PER_QUESTION
                ]
                for completion in sample_completions:
                    f.write(json.dumps({"task_id": sample["category"], "completion": clean_answer(completion)}) + "\n")


def score_pass_at_1(samples_path: str, problem_file: str) -> float:
    """Delegates to the human-eval package's own scoring (executes generated code -- call
    enable_code_execution() first). Returns pass@1.

    `problem_file` must be an explicit, real path to HumanEval.jsonl.gz -- NOT
    human_eval.data.HUMAN_EVAL, which hardcodes a path relative to wherever the `human_eval`
    package itself is installed (`<pkg_dir>/../data/HumanEval.jsonl.gz`). That relative layout
    only exists in the original GitHub checkout (a sibling data/ dir next to the human_eval/
    package); a `pip install`'d copy only ships the human_eval/ package code, not that sibling
    data/ dir (it was never part of the Python package), so HUMAN_EVAL points at a path that
    doesn't exist post-install -- confirmed: raises FileNotFoundError. Use the same file
    download_humaneval() already fetched for generation instead (see main()/generate_samples).
    """
    from human_eval.evaluation import evaluate_functional_correctness

    results = evaluate_functional_correctness(samples_path, k=[1], problem_file=problem_file)
    return results["pass@1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--peft-path", required=True)
    parser.add_argument("--data-root", default=os.environ.get("FEDORA_DATA_ROOT", "data"))
    parser.add_argument("--samples-out", default="humaneval_samples.jsonl")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--metrics-out", default=None, help="metrics.jsonl to append the result to")
    parser.add_argument("--device", default="cuda:0", help="single GPU to pin the model to (see load_model)")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="problems per generate() call")
    parser.add_argument("--num-shards", type=int, default=1, help="split HumanEval problems across this many parallel generation invocations (see server_app.py's evaluate_fn)")
    parser.add_argument("--shard-index", type=int, default=0, help="this invocation's shard, in [0, num_shards)")
    parser.add_argument(
        "--score-only", action="store_true",
        help="skip generation and score an already-merged --samples-out -- used by "
             "server_app.py after concatenating every shard's generated samples",
    )
    args = parser.parse_args()

    if args.score_only:
        enable_code_execution()
        problem_file = download_humaneval(args.data_root)
        pass_at_1 = score_pass_at_1(args.samples_out, problem_file)
        print(f"HumanEval pass@1: {pass_at_1:.4f}")
        if args.metrics_out:
            from fedbench_common.resultio import MetricsWriter

            MetricsWriter(path=args.metrics_out).write_round(
                args.round or -1, "humaneval_eval", humaneval_pass_at_1=pass_at_1
            )
        return

    generate_samples(
        args.base_model, args.peft_path, args.data_root, args.samples_out, args.device,
        args.eval_batch_size, args.shard_index, args.num_shards,
    )

    if args.num_shards > 1:
        print(f"HumanEval shard {args.shard_index}/{args.num_shards} generation complete -> {args.samples_out}")
        return

    # num_shards == 1: original single-process generate-then-score behavior.
    enable_code_execution()
    # Idempotent: download_humaneval() only re-downloads if the file is missing, so this just
    # returns the path already fetched inside generate_samples() a moment ago.
    problem_file = download_humaneval(args.data_root)
    pass_at_1 = score_pass_at_1(args.samples_out, problem_file)
    print(f"HumanEval pass@1: {pass_at_1:.4f}")

    if args.metrics_out:
        from fedbench_common.resultio import MetricsWriter

        MetricsWriter(path=args.metrics_out).write_round(
            args.round or -1, "humaneval_eval", humaneval_pass_at_1=pass_at_1
        )


if __name__ == "__main__":
    main()
