# quick-mixtral
A repo for learning code and debug of mixtral, Mixtral-8x7B-v0.1, Mixtral-8x7B-Instruct-v0.1 and new mixtral future

Part of the repository of ["quickllm"](https://github.com/zysNLP/quick-mixtral)，where has lots of models of LLM(large langurage model), thanks for your precious star!

Mixtral当前和未来学习代码和调试的repo。来自["quickllm"](https://github.com/zysNLP/quick-mixtral)的部分内容，感谢您宝贵的star!



### 1. Quickly Start with Mixtral

To study and debug Mixtral-8x7B as soon as possible，please start with the "basic_language_model_moe_transformers.py" in the root directory!

快速学习和调试Mixtral-8x7B，请从根目录下的“basic_language_model_moe_transformers.py”开始!

```python
# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：basic_language_model_moe_transformers.py
    @Author  ：ys
    @Time    ：2023/12/21 18:10
   	The moe part of the Mixtral-8x7b model, the following core code is from the official transformers library source code
    Mixtral-8x7b 模型中的moe部分，以下核心代码来自官方transformers库源代码
"""

import torch
torch.manual_seed(123)

from quickllm.layers.moe_by_transformers import MixtralConfig
from quickllm.layers.moe_by_transformers import MixtralSparseMoeBlock

config = MixtralConfig()
moe = MixtralSparseMoeBlock(config)

hidden_states = torch.randn(4, 71, 4096)
hidden_states, router_logits = moe(hidden_states)

print(hidden_states.shape, router_logits.shape)
```



### 2.Subsequent Update (后续更新)

(1) Quick fine-tuning of Mixtral models，include Mixtral-8x7B-v0.1、Mixtral-8x7B-Instruct-v0.1 and Lora/Qlora of Them.

(2) The new model that follows the Mixtral model in the future, and the related implementation of the higher quality model like Mixtral
