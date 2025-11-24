.. _SFT 训练:

SFT 训练
==============

命令行
-------------------------

您可以使用以下命令使用 ``examples/train_lora/llama3_lora_sft.yaml`` 中的参数进行微调：

.. code-block:: bash

    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

也可以通过追加参数更新 yaml 文件中的参数:

.. code-block:: bash
  
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
      learning_rate=1e-5 \
      logging_steps=1

.. note::

  LLaMA-Factory 默认使用所有可见的计算设备。根据需求可通过 ``CUDA_VISIBLE_DEVICES`` 或 ``ASCEND_RT_VISIBLE_DEVICES`` 指定计算设备。

.. _sft指令:

``examples/train_lora/llama3_lora_sft.yaml`` 提供了微调时的配置示例。该配置指定了模型参数、微调方法参数、数据集参数以及评估参数等。您需要根据自身需求自行配置。

.. code-block:: yaml

    ### examples/train_lora/llama3_lora_sft.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

    stage: sft
    do_train: true
    finetuning_type: lora
    lora_target: all

    dataset: identity,alpaca_en_demo
    template: llama3
    cutoff_len: 1024
    max_samples: 1000
    overwrite_cache: true
    preprocessing_num_workers: 16

    output_dir: saves/llama3-8b/lora/sft
    logging_steps: 10
    save_steps: 500
    plot_loss: true
    overwrite_output_dir: true

    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-4
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000

    val_size: 0.1
    per_device_eval_batch_size: 1
    eval_strategy: steps
    eval_steps: 500


.. note:: 

    模型 ``model_name_or_path`` 、数据集 ``dataset`` 需要存在且与 ``template`` 相对应。


.. list-table:: 重要训练参数
  :widths: 10 50
  :header-rows: 1

  * - 名称
    - 描述
  * - model_name_or_path
    - 模型名称或路径
  * - stage
    - 训练阶段，可选: rm(reward modeling), pt(pretrain), sft(Supervised Fine-Tuning), PPO, DPO, KTO, ORPO
  * - do_train
    - true用于训练, false用于评估
  * - finetuning_type
    - 微调方式。可选: freeze, lora, full
  * - lora_target
    - 采取LoRA方法的目标模块，默认值为 ``all``。
  * - dataset
    - 使用的数据集，使用","分隔多个数据集
  * - template
    - 数据集模板，请保证数据集模板与模型相对应。
  * - output_dir
    - 输出路径
  * - logging_steps
    - 日志输出步数间隔
  * - save_steps
    - 模型断点保存间隔
  * - overwrite_output_dir
    - 是否允许覆盖输出目录
  * - per_device_train_batch_size
    - 每个设备上训练的批次大小
  * - gradient_accumulation_steps
    - 梯度积累步数
  * - max_grad_norm
    - 梯度裁剪阈值
  * - learning_rate
    - 学习率
  * - lr_scheduler_type
    - 学习率曲线，可选 ``linear``, ``cosine``, ``polynomial``, ``constant`` 等。
  * - num_train_epochs
    - 训练周期数
  * - bf16
    - 是否使用 bf16 格式
  * - warmup_ratio
    - 学习率预热比例
  * - warmup_steps
    - 学习率预热步数
  * - push_to_hub
    - 是否推送模型到 Huggingface

