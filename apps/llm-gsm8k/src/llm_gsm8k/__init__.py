"""llm-gsm8k: Llama-3-8B federated fine-tuning Flower app, train on GSM8K's own training set
(N=3, IID split), evaluated on GSM8K exact-match. A separate experiment from apps/llm-humaneval's
CodeSearchNet/HumanEval pipeline (see that app's docstrings for why GSM8K used to be
(incorrectly) evaluated off the CodeSearchNet checkpoint instead of its own dedicated training
run) -- confirmed via FedRot-LoRA/federatedscope/llm/yamls/base_rescale.yaml, the official repo's
only `data.type: gsm8k@llm` yaml (client_num=3, splitter='iid', total_round_num=100)."""
