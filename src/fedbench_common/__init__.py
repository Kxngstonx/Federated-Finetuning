"""Shared library for the GLUE/NLU (apps/glue-nlu) and Llama (apps/llm-humaneval, apps/llm-gsm8k)
experiment pipelines: Dirichlet/language/IID partitioners, measurement instrumentation, and result
I/O. Not itself a Flower app -- imported by all sibling apps alongside flowertune_llm.strategies
and flowertune_llm.peft_layers.
"""
