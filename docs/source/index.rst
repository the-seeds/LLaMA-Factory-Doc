Welcome to LLaMA Factory!
=========================

.. figure:: ./assets/logo.png
  :width: 100%
  :align: center
  :alt: logo


LLaMA Factory 是一个简单易用且高效的大型语言模型（Large Language Model）训练与微调平台。通过 LLaMA Factory，可以在无需编写任何代码的前提下，在本地完成上百种预训练模型的微调，框架特性包括：

* 模型种类：LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi 等等。
* 训练算法：（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO 训练、DPO 训练、KTO 训练、ORPO 训练等等。
* 运算精度：16 比特全参数微调、冻结微调、LoRA 微调和基于 AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ 的 2/3/4/5/6/8 比特 QLoRA 微调。
* 优化算法：GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ 和 PiSSA。
* 加速算子：FlashAttention-2 和 Unsloth。
* 推理引擎：Transformers 和 vLLM。
* 实验监控：LlamaBoard、TensorBoard、Wandb、MLflow、SwanLab 等等。

Documentation
------------------

.. toctree::
  :maxdepth: 1
  :caption: 开始使用

  getting_started/installation
  getting_started/data_preparation
  getting_started/webui
  getting_started/sft
  getting_started/merge_lora
  getting_started/inference
  getting_started/eval

.. toctree::
  :maxdepth: 3
  :caption: 高级选项

  advanced/acceleration
  advanced/adapters
  advanced/distributed
  advanced/quantization
  advanced/trainers
  advanced/monitor
  advanced/npu_installation
  advanced/npu_training
  advanced/npu_inference
  advanced/arguments
  advanced/model_support
  advanced/extras
  advanced/best_practice/index

