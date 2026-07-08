"""GSM8K exact-match evaluation, ported from
FedRot-LoRA/federatedscope/llm/eval/eval_for_gsm8k/eval.py.

The pure-Python pieces (ANS_RE, create_demo_text, build_prompt, clean_answer, is_correct) are
kept verbatim -- no FederatedScope dependency there. FSChatBot (FederatedScope's model wrapper)
is replaced with a plain transformers.AutoModelForCausalLM + peft.PeftModel checkpoint load, and
FederatedScope's download_url helper is replaced with a plain `requests.get`.

Usage: python eval_gsm8k.py --base-model meta-llama/Meta-Llama-3-8B --peft-path results/<ts>/peft_<round> \
    --seed 1 --strategy fedrot --round 200 --metrics-out results/<ts>/metrics.jsonl
"""

import argparse
import json
import os
import random
import re
from typing import Tuple

import requests
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
ANSWER_TRIGGER = "The answer is"

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/test.jsonl"
)


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def is_correct(model_answer, answer):
    if "The answer is" in answer:
        answer = answer.replace("The answer is", "####")
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers "
        "plant today?"
    )
    chain.append(
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in "
        "the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they "
        "have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
        "32 + 42 = 74. After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How "
        "many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave "
        "Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How "
        "many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 "
        "more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers were installed each "
        "day, from monday to thursday. How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So "
        "5 * 4 = 20 computers were added. 9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 "
        "more. How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she "
        "has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    index_list = list(range(len(question)))
    random.shuffle(index_list)

    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: " + question[i] + "\nA: " + chain[i] + " " + ANSWER_TRIGGER + " "
                + answer[i] + ".\n\n"
            )
        else:
            demo_text += (
                "Question: " + question[i] + "\nAnswer: " + ANSWER_TRIGGER + " " + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    return demo + "Q: " + input_text + "\n" + "A:"


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    pred = preds[1] if answer_flag else preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    pred = pred[0] if answer_flag else pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred


def download_gsm8k_test(data_root: str) -> str:
    fp = os.path.join(data_root, "gsm8k_test.jsonl")
    if not os.path.exists(fp):
        os.makedirs(data_root, exist_ok=True)
        resp = requests.get(GSM8K_TEST_URL, timeout=60)
        resp.raise_for_status()
        with open(fp, "wb") as f:
            f.write(resp.content)
    return fp


def load_jsonl(fp: str, instruction: str, output: str):
    data = []
    with open(fp) as f:
        for line in f:
            record = json.loads(line)
            data.append({"instruction": record[instruction], "output": record[output]})
    return data


def load_model(base_model: str, peft_path: str, device: str = "cuda:0"):
    """device_map pinned to a single GPU (not "auto"): this 8B model in bf16/4bit fits comfortably
    on one 48GB GPU, and "auto" naively pipeline-splits layers across every visible GPU, which for
    autoregressive generation means computation ping-pongs between GPUs one token at a time
    (only one GPU ever computes while the other sits idle waiting for activations over PCIe) --
    that's what produced the 100%/0%-alternating nvidia-smi pattern and a ~2.4-hour-and-counting
    single GSM8K eval. Pinning to one device also stops this subprocess from silently stealing
    memory/compute on whichever GPU a training client is concurrently using for its other half.

    float16, not bfloat16: this fleet's Quadro RTX 8000s are Turing (SM75), which has no bf16
    Tensor Core path (added in Ampere/SM80) -- bf16 matmuls silently fall back to an unaccelerated
    path there, which was confirmed to blow up a single 1319-example GSM8K eval to ~9.6h. fp16 has
    full Tensor Core support on Turing; generation-only inference (no backward pass) doesn't need
    bf16's wider dynamic range."""
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
def evaluate_gsm8k(
    base_model: str, peft_path: str, data_root: str = "data", device: str = "cuda:0",
    batch_size: int = 4, shard_index: int = 0, num_shards: int = 1,
) -> Tuple[int, int]:
    """Returns (num_correct, num_total) rather than a plain accuracy float so that a caller
    running several shards on several GPUs (see server_app.py's evaluate_fn) can combine them into
    one overall accuracy via a size-weighted sum, instead of a naive mean-of-shard-accuracies that
    would be wrong whenever shards end up uneven in size."""
    model, tokenizer = load_model(base_model, peft_path, device)
    fp = download_gsm8k_test(data_root)
    list_data_dict = load_jsonl(fp, instruction="question", output="answer")
    # Interleaved (stride) sharding rather than contiguous chunks: balances each shard's mix of
    # short/long questions (and thus generation cost) more evenly than splitting the file in half.
    list_data_dict = list_data_dict[shard_index::num_shards]

    answers = []
    for start in tqdm(range(0, len(list_data_dict), batch_size), desc=f"gsm8k[{shard_index}/{num_shards}]"):
        batch = list_data_dict[start:start + batch_size]
        input_texts = [build_prompt(sample["instruction"], N_SHOT, COT_FLAG) for sample in batch]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=256, top_p=0.95, temperature=0.8, do_sample=True
        )
        completions = tokenizer.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for sample, completion in zip(batch, completions):
            model_answer = clean_answer(completion)
            answers.append(is_correct(model_answer, sample["output"]))

    return sum(answers), len(answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--peft-path", required=True)
    parser.add_argument("--data-root", default=os.environ.get("FEDORA_DATA_ROOT", "data"))
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--metrics-out", default=None, help="metrics.jsonl to append the result to")
    parser.add_argument("--device", default="cuda:0", help="single GPU to pin the model to (see load_model)")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="samples per generate() call")
    parser.add_argument("--num-shards", type=int, default=1, help="split the test set across this many parallel invocations (see server_app.py's evaluate_fn)")
    parser.add_argument("--shard-index", type=int, default=0, help="this invocation's shard, in [0, num_shards)")
    parser.add_argument(
        "--shard-out", default=None,
        help="when --num-shards > 1, write {correct, total} JSON here instead of the final "
             "metric -- the caller merges every shard's result and writes metrics-out itself",
    )
    args = parser.parse_args()

    correct, total = evaluate_gsm8k(
        args.base_model, args.peft_path, args.data_root, args.device, args.eval_batch_size,
        args.shard_index, args.num_shards,
    )

    if args.num_shards > 1:
        with open(args.shard_out, "w") as f:
            json.dump({"correct": correct, "total": total}, f)
        print(f"GSM8K shard {args.shard_index}/{args.num_shards}: {correct}/{total} correct")
        return

    acc = correct / total
    print(f"GSM8K exact-match accuracy: {acc:.4f}")

    if args.metrics_out:
        from fedbench_common.resultio import MetricsWriter

        MetricsWriter(path=args.metrics_out).write_round(
            args.round or -1, "gsm8k_eval", gsm8k_exact_match=acc
        )


if __name__ == "__main__":
    main()
