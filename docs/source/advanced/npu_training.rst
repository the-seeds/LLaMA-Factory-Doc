NPU训练
================

本文档介绍如何在华为昇腾 NPU 上进行 LLaMA-Factory 模型训练。

支持设备
--------

LLaMA-Factory 当前已适配以下昇腾 NPU 设备：

- **Atlas A2 训练系列**
- **Atlas A3 训练系列**

支持功能
--------

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
     - FSDP2
     - 已支持
   * - 
     - DeepSpeed
     - 已支持
   * - **加速**
     - 融合算子
     - 当前已支持NpuFusedRMSNorm，NpuFusedSwiGlu，NpuFusedRoPE，NpuFusedMoE

.. note::
   NPU 的大部分使用方式与 GPU 保持一致。关于通用的安装步骤，请参考 :doc:`NPU 安装及配置 <npu_installation>`；关于通用的分布式训练（如 FSDP、FSDP2，DeepSpeed）配置，请参考 :doc:`分布式训练 <distributed>`。

快速开始
--------

为了快速上手，建议直接使用 LLaMA-Factory 提供的 Docker 镜像。

1. **启动容器** (请根据实际情况修改 ``device`` 映射)：

   .. code-block:: bash

     docker run -itd \
         --net=host \
         --device=/dev/davinci0 \
         --device=/dev/davinci1 \
         --device=/dev/davinci_manager \
         --device=/dev/devmm_svm \
         --device=/dev/hisi_hdc \
         --shm-size=1200g \
         -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
         --name llama_factory_npu \
         hiyouga/llamafactory:latest-npu-a2 \
         /bin/bash

2. **配置环境变量**：

   进入容器后，**务必** 先加载 Ascend 环境配置，否则无法识别 NPU 设备：

   .. code-block:: bash

      source /usr/local/Ascend/ascend-toolkit/set_env.sh


3. **开始训练**：

   .. code-block:: bash

      llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

分布式训练
----------

NPU 的分布式训练配置与 :doc:`分布式训练 <distributed>` 文档描述的基本一致。本节主要介绍 NPU 环境下的特定配置，包括设备指定和多机通信设置。

关键环境变量
~~~~~~~~~~~~

在启动训练前，请注意以下环境变量的设置：

*   **ASCEND_RT_VISIBLE_DEVICES** (单机/多机均需关注)

    用于指定参与训练的 NPU 设备。

    *   **默认行为**：如果不设置此变量，程序将尝试使用当前节点上的**所有** NPU 设备。
    *   **指定设备**：如果需要限定特定的 NPU 卡（例如仅使用卡 0 和卡 1），则**必须**显式设置此变量：

        .. code-block:: bash

            export ASCEND_RT_VISIBLE_DEVICES=0,1

*   **HCCL_SOCKET_IFNAME** (仅多机训练必需)

    指定 HCCL 集合通信使用的网卡接口名称。

    *   **获取方式**：在终端运行 ``ifconfig`` 命令查看网卡列表，选择用于通信的网卡名称（如 ``eth0``, ``enp1s0`` 等）。
    *   **设置示例**：

        .. code-block:: bash

           export HCCL_SOCKET_IFNAME=eth0

单机训练
~~~~~~~~

单机训练（单卡或多卡）的启动方式与标准流程一致。

**单机多卡示例**：

.. code-block:: bash

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

多机训练
~~~~~~~~

在 NPU 环境下，推荐使用 ``accelerate launch`` 配合 FSDP 1/2 进行多机训练，这种方式在 NPU 上通信和计算效率更优。

.. note::
   其他启动方式（如 ``torchrun/deepspeed``）及更多详细配置请参考 :doc:`分布式训练 <./distributed>` 文档。

**1. 准备 Accelerate 配置文件**

创建或修改 ``examples/accelerate/fsdp_config.yaml``，关键参数如下（请根据实际节点数和 IP 修改）：

.. code-block:: yaml

    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
      fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
      fsdp_backward_prefetch: BACKWARD_PRE
      fsdp_forward_prefetch: false
      fsdp_cpu_ram_efficient_loading: true
      fsdp_offload_params: false
      fsdp_sharding_strategy: FULL_SHARD
      fsdp_state_dict_type: FULL_STATE_DICT
      fsdp_sync_module_states: true
      fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    main_process_ip: 192.168.0.1
    main_process_port: 29500
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    use_cpu: false

.. note::

    关键多机参数说明：

    - num_machines: 节点总数
    - num_processes: 总进程数（总卡数） = num_machines * 每台机器卡数
    - main_process_ip: 主节点 IP 地址（所有节点需保持一致）
    - main_process_port: 主节点端口（所有节点需保持一致）
    - machine_rank: 当前节点编号（主节点为0，从节点依次递增）

**2. 启动训练**

在所有节点上执行相同的启动命令（确保 ``machine_rank`` 在 yaml 中已正确配置）：

.. code-block:: bash

   export HCCL_SOCKET_IFNAME=eth0

   accelerate launch --config_file examples/accelerate/fsdp_config_multiple_nodes.yaml \
       src/train.py examples/train_lora/llama3_lora_sft.yaml

训练方式
--------

以下是常见训练场景的启动命令参考，具体参数配置文件请根据实际需求调整。

预训练 (PT)
~~~~~~~~~~~

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml

监督微调 (SFT)
~~~~~~~~~~~~~~

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

奖励模型 (RM)
~~~~~~~~~~~~~

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml

DPO 训练
~~~~~~~~

.. code-block:: bash

   llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml

全参数微调 (Full)
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   llamafactory-cli train examples/train_full/llama3_full_sft.yaml

性能优化
--------

融合算子
~~~~~~~~~~~~~~~~~~~~~~~~~~

LLaMA-Factory 支持FA，NpuFusedRMSNorm，NpuFusedSwiGlu，NpuFusedRoPE和NpuFusedMoE融合算子。

可在训练脚本中配置如下参数，模型加载后替换对应模型结构，使能NpuFusedRMSNorm，NpuFusedSwiGlu，NpuFusedRoPE和NpuFusedMoE融合算子，提升训练效率。该接口使能后，代码内部自动识别是否满足模型结构替换的要求，满足的情况对应模型结构会被替换为融合算子形式。

.. code-block:: yaml

   use_v1_kernels: true


同时LLaMA-Factory 支持昇腾 NPU 的 FA 融合算子，代码内部自动识别是否满足模型结构替换的要求，满足的情况对应模型结构会被替换为融合算子形式。在训练配置文件中设置如下参数即可使能：

.. code-block:: yaml

   flash_attn: fa2

当前融合算子对模型的支持程度受限，该功能正在持续迭代开发中，以提升泛化性和适用性。

.. list-table::
   :align: left
   :widths: 20 60
   :header-rows: 1

   * - 融合算子
     - 支持模型系列
   * - FA
     - Qwen3, Qwen3-MOE, Qwen3-VL, Qwen3-VL-MOE
   * - NpuFusedRMSNorm
     - Qwen3, Qwen3-MOE, Qwen3-VL, Qwen3-VL-MOE
   * - NpuFusedSwiGlu
     - Qwen3, Qwen3-MOE, Qwen3-VL, Qwen3-VL-MOE
   * - NpuFusedRoPE
     - Qwen3, Qwen3-MOE, Qwen3-VL, Qwen3-VL-MOE
   * - NpuFusedMoE
     - Qwen3-MOE，Qwen3-VL-MOE


算子下发优化
~~~~~~~~~~~~

通过设置 ``TASK_QUEUE_ENABLE`` 环境变量优化算子下发性能（推荐 Level 2）：

.. code-block:: bash

   export TASK_QUEUE_ENABLE=2
