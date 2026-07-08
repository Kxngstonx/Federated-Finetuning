"""llm-humaneval: Llama-3-8B federated fine-tuning Flower app, train on CodeSearchNet (N=6,
non-IID split by programming language), evaluated on HumanEval pass@1. GSM8K moved to its own
sibling app (apps/llm-gsm8k) since the paper trains/evaluates GSM8K as a separate experiment
(N=3, IID split on GSM8K's own training set) rather than off this CodeSearchNet checkpoint."""
