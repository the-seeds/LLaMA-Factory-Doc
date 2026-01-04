GPT-OSS
==================================================

3步实现 GPT-OSS 的 LoRA 微调
-------------------------------------------------------

1. 安装 LLaMA-Factory 和 transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e .
   pip install -r requirements/metrics.txt --no-build-isolation
   pip install "transformers==4.55.0"

2. 在单张 GPU 上训练 GPT-OSS（要求显存 > 44 GB, 支持多 GPU）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   llamafactory-cli train examples/train_lora/gpt_lora_sft.yaml

3. 合并 LoRA 权重
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    llamafactory-cli export --model_name_or_path openai/gpt-oss-20b \
        --adapter_name_or_path saves/gpt-20b/lora/sft \
        --export_dir gpt_merged

与微调后的模型进行对话
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

   llamafactory-cli chat --model_name_or_path gpt_merged --template gpt --skip_special_tokens False

全量微调脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: bash

    ### model
    model_name_or_path: openai/gpt-oss-20b
    trust_remote_code: true

    ### method
    stage: sft
    do_train: true
    finetuning_type: full
    deepspeed: examples/deepspeed/ds_z3_config.json

    ### dataset
    dataset: identity,alpaca_en_demo
    template: gpt
    cutoff_len: 2048
    max_samples: 1000
    overwrite_cache: true
    preprocessing_num_workers: 16
    dataloader_num_workers: 4

    ### output
    output_dir: saves/gpt-20b/lora/sft
    logging_steps: 10
    save_steps: 500
    plot_loss: true
    overwrite_output_dir: true
    save_only_model: false
    report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

    ### train
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-4
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000
    resume_from_checkpoint: null

    ### eval
    # eval_dataset: alpaca_en_demo
    # val_size: 0.1
    # per_device_eval_batch_size: 1
    # eval_strategy: steps
    # eval_steps: 500


.. image:: ../../assets/advanced/gpt-20b-loss.png
   :alt: 训练损失曲线
   :align: center

使用 Web UI 微调模型：

.. image:: ../../assets/advanced/gpt-20b-webui.png
   :alt: 使用 Web UI 微调 gpt-oss
   :align: center
