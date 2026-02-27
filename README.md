# Zen4 Ultra

Zen4 Ultra — frontier-scale language model. Largest dense model in the Zen4 family.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Zen4 Ultra is a 405B dense parameter model delivering frontier-level performance across reasoning, code, math, science, and multilingual tasks. Designed for maximum capability with no routing overhead.

## Key Features

- **405B dense** parameters — no MoE routing, full parameter utilization
- **128K context window** for long-document understanding
- State-of-the-art reasoning and instruction following
- Strong multilingual and cross-domain performance
- Extended thinking and chain-of-thought capabilities

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen4-ultra"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain the implications of Godel's incompleteness theorems for artificial general intelligence."},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=4096)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Serving

For production inference, use vLLM with tensor parallelism:

```bash
vllm serve zenlm/zen4-ultra \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --port 8000
```

## GGUF Quantized

Quantized GGUF models for local inference with llama.cpp:

- [zen4-ultra-Q4_K_M.gguf](https://huggingface.co/zenlm/zen4-ultra/resolve/main/zen4-ultra-Q4_K_M.gguf)
- [zen4-ultra-Q8_0.gguf](https://huggingface.co/zenlm/zen4-ultra/resolve/main/zen4-ultra-Q8_0.gguf)

## Zen4 Family

| Model | Parameters | Focus |
|-------|-----------|-------|
| **Zen4 Ultra** | 405B dense | Frontier general |
| Zen4 Coder Pro | 80B MoE | Professional coding |
| Zen4 Coder | 32B | Code generation |
| Zen4 | 32B | General purpose |
| Zen4 Mini | 8B | Efficient deployment |

## Related

- [zen4-coder-pro](https://github.com/zenlm/zen4-coder-pro) — Professional code generation
- [llama.cpp](https://github.com/zenlm/llama.cpp) — Optimized GGUF inference
- [Zen LM](https://github.com/zenlm) — Full model family

Apache 2.0 · [Zen LM](https://zenlm.org) · [Hanzo AI](https://hanzo.ai)
