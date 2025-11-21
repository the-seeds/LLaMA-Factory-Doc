NPU训练
================

本文档介绍如何在华为昇腾 NPU 上进行 LLaMA-Factory 模型训练，包括设备支持、训练范式、分布式策略以及性能优化等内容。

支持设备
----------------

LLaMA-Factory 当前已适配以下昇腾 NPU 设备：

- **Atlas A2 训练系列**
- **Atlas 800I A2 推理系列**


支持功能
----------------

.. list-table::
   :align: left
   :widths: 20 30 50
   :header-rows: 1

   * - 
     - 功能
     - 支持情况
   * - **训练范式**
     - PT
     - 已支持
   * - 
     - SFT
     - 已支持
   * - 
     - RM
     - 已支持
   * - 
     - DPO
     - 已支持
   * - **参数范式**
     - Full
     - 已支持
   * - 
     - Freeze
     - 已支持
   * - 
     - LoRA
     - 已支持
   * - **模型合并**
     - LoRA权重合并
     - 已支持
   * - **分布式**
     - DDP
     - 已支持
   * - 
     - FSDP
     - 已支持
   * - 
     - DeepSpeed
     - 已支持
   * - **加速**
     - 融合算子
     - 当前仅支持NPU FA融合算子


.. note::
   除特别说明外，NPU 的使用方式与 GPU 保持一致，无需额外配置。以下将重点介绍 NPU 特定的配置和注意事项。

分布式训练
----------------

单机微调
~~~~~~~~~~~~~~~~

本节介绍如何在单机环境下使用 Docker 容器进行 NPU 模型微调。LLaMA-Factory 提供了两种 Docker 部署方式。

**方法一：使用 Docker Compose（推荐）**

.. code-block:: bash

   cd docker/docker-npu/
   docker compose up -d
   docker compose exec llamafactory bash

**方法二：使用 Docker Run**

拉取昇腾 LLaMA-Factory的 `预构建镜像 <https://hub.docker.com/r/hiyouga/llamafactory/tags>`_ ，请根据实际需要选择合适的版本标签，建议选取最新版本以获得更好的功能支持和性能表现：

.. code-block:: bash

   #hub.docker.com
   docker pull hiyouga/llamafactory:<version>-npu-a2
   
   #quay.io
   docker pull quay.io/ascend/llamafactory:<version>-npu-a2
   


启动容器：

.. code-block:: bash

  CONTAINER_NAME=llama_factory_npu
  DOCKER_IMAGE=hiyouga/llamafactory:latest-npu-a2
  docker run -itd \
      --cap-add=SYS_PTRACE \
      --net=host \
      --device=/dev/davinci0 \
      --device=/dev/davinci1 \
      --device=/dev/davinci2 \
      --device=/dev/davinci3 \
      --device=/dev/davinci4 \
      --device=/dev/davinci5 \
      --device=/dev/davinci6 \
      --device=/dev/davinci7 \
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      --shm-size=1200g \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /data:/data \
      --privileged=true \
      --name "$CONTAINER_NAME" \
      "$DOCKER_IMAGE" \
      /bin/bash

进入容器：

.. code-block:: bash

   docker exec -it llama_factory_npu bash

.. note::
   **NPU 设备挂载说明**：
   
   - 通过 ``--device /dev/davinci<N>`` 参数可挂载指定的 NPU 卡，最多可挂载全部 8 卡
   - 昇腾 NPU 设备从 0 开始编号，容器内的设备编号会自动重新映射
   - 例如：将物理机上的 davinci6 和 davinci7 挂载到容器，容器内对应的设备编号将为 0 和 1

进入容器后，LLaMA-Factory 已预装完成，可直接使用 ``llamafactory-cli train`` 命令启动训练。该命令会自动识别容器内所有挂载的 NPU 设备并启用分布式训练：

.. code-block:: bash

   llamafactory-cli train examples/train_full/llama3_full_sft.yaml

部分微调输出如下所示：

.. code-block:: shell

  ...
  [INFO|2025-07-21 06:11:30] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...
  Converting format of dataset (num_proc=16): 100%|████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 3468.44 examples/s]
  ...
  {'loss': 0.9742, 'grad_norm': 30533.959586008496, 'learning_rate': 7.364864864864865e-05, 'epoch': 0.22}
  {'loss': 1.099, 'grad_norm': 57068.40470873529, 'learning_rate': 8.040540540540541e-05, 'epoch': 0.24}
    9%|█████████▊                                                                                                        | 126/1473 [01:15<13:13,  1.70it/s]
  ...
  [INFO|trainer.py:4332] 2025-07-21 06:26:51,940 >>   Batch size = 1
  100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:07<00:00, 13.98it/s]
  ***** eval metrics *****
    epoch                   =        3.0
    eval_loss               =     0.9487
    eval_runtime            = 0:00:07.95
    eval_samples_per_second =     13.826
    eval_steps_per_second   =     13.826
  [INFO|modelcard.py:450] 2025-07-21 06:26:59,899 >> Dropping the following result as it does not have all the necessary fields:
  {'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}

若需指定特定的 NPU 设备进行训练（如仅使用卡 0 和卡 1），可通过 ``ASCEND_RT_VISIBLE_DEVICES`` 环境变量进行配置：

.. code-block:: bash

   ASCEND_RT_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml

.. note::
   **模型权重下载配置**：
   
   - 默认情况下，系统会从 `Hugging Face <https://huggingface.co/models>`_ 下载模型权重
   - 如遇到网络访问问题，可设置环境变量 ``export USE_MODELSCOPE_HUB=1`` 切换至 `ModelScope <https://modelscope.cn/models>`_ 镜像源


多机微调
~~~~~~~~~~~~~~~~

多机微调场景下，建议直接在物理机上部署 LLaMA-Factory 以获得更好的资源利用率。请参考 :doc:`NPU安装及配置 <./npu_installation>` 完成基础环境配置。

.. code-block:: bash
   pip install -e .
   

**环境配置要求**：

1. **指定 NPU 设备**：在每个节点上通过环境变量 ``ASCEND_RT_VISIBLE_DEVICES`` 指定参与训练的 NPU 设备（如 ``export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3``）。若不指定，将默认使用节点上的所有 NPU 设备。

2. **配置通信网卡**：在每个节点上通过环境变量 ``HCCL_SOCKET_IFNAME`` 指定 HCCL 集合通信使用的网卡接口（如 ``export HCCL_SOCKET_IFNAME=eth0``）。可通过 ``ifconfig`` 命令查看可用网卡名称。

**启动训练**：

以双节点训练为例，分别在主节点和从节点执行以下命令：

.. code-block:: bash

   # master node（NODE_RANK=0）
   FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
   llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml
   
   # worker node（NODE_RANK=1）
   FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
   llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml

**环境变量说明**：

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - 环境变量
     - 说明
   * - ``FORCE_TORCHRUN``
     - 强制使用 torchrun 启动分布式训练
   * - ``NNODES``
     - 参与训练的节点总数
   * - ``NODE_RANK``
     - 当前节点的全局排名（主节点为 0，从节点依次递增）
   * - ``MASTER_ADDR``
     - 主节点的 IP 地址
   * - ``MASTER_PORT``
     - 主节点用于分布式通信的端口号

训练范式
----------------

LLaMA-Factory 在 NPU 上支持多种训练范式，使用方式与 GPU 保持一致。以下是各训练范式的启动示例：

**预训练（Pre-Training, PT）**

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml

**监督微调（Supervised Fine-Tuning, SFT）**

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

**奖励模型训练（Reward Modeling, RM）**

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml

**直接偏好优化（Direct Preference Optimization, DPO）**

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml

参数范式
----------------

LLaMA-Factory 在 NPU 上支持多种参数微调策略，使用方式与 GPU 保持一致。以下是各参数范式的启动示例：

**全参数微调（Full Fine-Tuning）**

对模型的所有参数进行更新：

.. code-block:: bash

   llamafactory-cli train examples/train_full/llama3_full_sft.yaml

**冻结微调（Freeze Fine-Tuning）**

冻结部分层的参数，仅更新指定层：

.. code-block:: bash

   llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml

**LoRA 微调**

使用低秩适配器（Low-Rank Adaptation）进行参数高效微调：

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

模型合并
----------------

使用 LoRA 方法训练完成后，会生成相应的适配器权重文件。如需将适配器权重合并到基础模型中，请参考 :doc:`LoRA 模型合并 <../getting_started/merge_lora>` 文档。


分布式策略
----------------

LLaMA-Factory 在 NPU 上支持多种分布式训练策略，包括 DDP、FSDP 和 DeepSpeed，使用方式与 GPU 保持一致。

DeepSpeed
~~~~~~~~~~~~~~~~

LLaMA-Factory 在 ``examples/deepspeed`` 目录下提供了多种 DeepSpeed 配置文件。NPU 当前已支持 ZeRO Stage 1/2/3 及 Offload 功能，可根据训练需求选择相应配置。

在训练配置文件中添加 ``deepspeed`` 参数即可启用 DeepSpeed：

.. code-block:: yaml

   ### model
   model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
   trust_remote_code: true

   ### method
   stage: sft
   do_train: true
   finetuning_type: full
   deepspeed: examples/deepspeed/ds_z3_config.json

FSDP
~~~~~~~~~~~~~~~~

LLaMA-Factory 通过 Accelerate 库支持 FSDP（Fully Sharded Data Parallel）分布式训练。Torch-NPU 已实现与 PyTorch FSDP 接口的兼容，可通过以下命令启动 FSDP 训练：

.. code-block:: bash

   accelerate launch --config_file examples/accelerate/fsdp_config.yaml \
     src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml

若需指定特定 NPU 设备，可通过 ``ASCEND_RT_VISIBLE_DEVICES`` 环境变量进行配置：

.. code-block:: bash

   ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch \
     --config_file examples/accelerate/fsdp_config.yaml \
     src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml

LLaMA-Factory 提供了多种 FSDP 配置文件供选择：

- ``examples/accelerate/fsdp_config.yaml``：标准 FSDP 配置
- ``examples/accelerate/fsdp_config_offload.yaml``：支持 CPU Offload 的 FSDP 配置


性能优化
----------------

本节介绍 NPU 训练场景下已验证的性能优化方法。更多优化特性将在验证后持续更新。

融合算子
~~~~~~~~~~~~~~~~

LLaMA-Factory 已支持昇腾 NPU 的 FA 融合算子，可显著提升训练性能。在训练配置文件中设置 ``flash_attn: fa2`` 即可启用：

.. code-block:: yaml

   ### model
   model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
   trust_remote_code: true

   ### method
   stage: sft
   do_train: true
   finetuning_type: full
   deepspeed: examples/deepspeed/ds_z3_config.json
   flash_attn: fa2

.. note::
   融合算子优化仅在训练阶段生效，推理阶段不启用。系统会自动检测并替换相应的算子实现。

算子下发优化
~~~~~~~~~~~~~~~~

昇腾提供了 ``TASK_QUEUE_ENABLE`` 环境变量用于优化算子下发性能，可通过流水线并行机制降低算子下发延迟：

.. code-block:: bash

   export TASK_QUEUE_ENABLE=2

**配置选项说明**：

- **Level 0（TASK_QUEUE_ENABLE=0）**：关闭 task_queue 算子下发队列优化
  
- **Level 1（TASK_QUEUE_ENABLE=1，默认值）**：启用基础的 task_queue 优化，将算子下发任务分为两级流水线：
  
  - 一级流水：处理常规算子调用
  - 二级流水：处理 ACLNN 算子调用
  - 两级流水通过队列并行执行，部分掩盖下发延迟
  
- **Level 2（TASK_QUEUE_ENABLE=2，推荐）**：在 Level 1 基础上进一步优化任务负载均衡：
  
  - 将 workspace 相关任务迁移至二级流水
  - 提供更好的延迟掩盖效果和性能收益
  - 建议在训练场景中使用该配置以获得最佳性能

