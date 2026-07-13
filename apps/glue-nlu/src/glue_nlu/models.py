"""RoBERTa-Large + PEFT LoRA sequence-classification model, ported from
FedRot-LoRA/federatedscope/glue/model/{model_builder.py, adapter_builder.py}.

get_parameters/set_parameters are intentionally reused from flowertune_llm.models unchanged --
both already operate generically over get_peft_model_state_dict/set_peft_model_state_dict, with
no assumption about task type (SEQ_CLS vs CAUSAL_LM).
"""

from typing import Optional

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

# label_list per FedRot-LoRA/federatedscope/glue/dataloader/dataloader.py's
# datasets["train"].features["label"].names cardinality for each of the 5 required tasks.
GLUE_NUM_LABELS = {"sst2": 2, "qnli": 2, "mnli": 3, "qqp": 2, "rte": 2}

# RoBERTa attention projection module names (PEFT's own built-in default target-module mapping
# for model_type="roberta" is exactly this pair -- pinned explicitly here for clarity/robustness
# against future PEFT default-mapping changes, matching FedRot-LoRA's LoraConfig call which
# relies on that same PEFT default).
ROBERTA_LORA_TARGET_MODULES = ["query", "value"]


def get_model(model_cfg: DictConfig, task_name: str, aggregation: Optional[str] = None):
    """Build RoBERTa-Large wrapped in a PEFT LoRA adapter for GLUE sequence classification.

    use_dora is forced on when `aggregation` is 'fedora' (this strategy requires DoRA's
    magnitude vector; see flowertune_llm/strategies/fedora.py's ValueError otherwise),
    mirroring flowertune_llm/models.py's own `use_dora=model_cfg.get("use_dora", False)`
    convention but adding this override so the same pyproject.toml config works for every
    strategy in an experiment sweep without per-strategy config edits.
    """
    num_labels = GLUE_NUM_LABELS[task_name]
    # FedRot-LoRA/federatedscope/glue/model/model_builder.py's is_enable_half=True protocol
    # casts RoBERTa-Large to fp16 for GLUE fine-tuning -- torch_dtype at from_pretrained time
    # matches that (equivalent to loading fp32 then .half()).
    base = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name, num_labels=num_labels, torch_dtype=torch.float16
    )

    use_dora = model_cfg.get("use_dora", False) or aggregation == "fedora"

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=model_cfg.lora.peft_lora_dropout,
        task_type=TaskType.SEQ_CLS,
        target_modules=ROBERTA_LORA_TARGET_MODULES,
        use_dora=use_dora,
    )
    return get_peft_model(base, peft_config)
