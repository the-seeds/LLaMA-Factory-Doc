XPU训练
================

本文档介绍如何在 Intel GPU 上进行 LlamaFactory 模型训练。

支持设备
--------

已验证设备：

- **Intel Arc Pro B60**

其他 Intel GPU 请参考 :doc:`XPU 安装及配置 <xpu_installation>` 中的说明。

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
   * -
     - KTO
     - 已支持
   * - **参数范式**
     - Full
     - 已支持
   * -
     - Freeze
     - 已支持
   * -
     - LoRA（含 rsLoRA、DoRA、PiSSA、LoRA+）
     - 已支持
   * -
     - OFT
     - 已支持
   * - **量化**
     - QLoRA（bitsandbytes 4/8-bit）
     - 已支持
   * -
     - GPTQ、AWQ、quanto、FP8
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
     - DeepSpeed ZeRO-2/3
     - 已支持
   * - **多模态**
     - 图文模型 LoRA 微调
     - 已支持
   * - **加速**
     - Liger Kernel
     - 已支持
   * -
     - FlashAttention-2
     - 适配中，当前使用 SDPA
   * - **推理**
     - HuggingFace 引擎、API、WebUI
     - 已支持
   * -
     - vLLM
     - 适配中

.. note::
   XPU 的大部分使用方式与 GPU 保持一致。关于安装步骤，请参考 :doc:`XPU 安装及配置 <xpu_installation>`；关于通用的分布式训练（如 FSDP、FSDP2、DeepSpeed）配置，请参考 :doc:`分布式训练 <../../advanced/distributed>`。

快速开始
--------

1. **启动容器**，或按照 :ref:`xpu_install_form_pip` 安装环境：

   .. code-block:: bash

      cd docker/docker-xpu
      docker compose up -d
      docker compose exec llamafactory bash

2. **开始训练**：

   .. code-block:: bash

      llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

分布式训练
----------

XPU 的分布式训练配置与 :doc:`分布式训练 <../../advanced/distributed>` 文档描述的基本一致，卡间通信使用 oneCCL（``xccl`` 后端）。本节主要介绍 XPU 环境下的特定配置。

.. _xpu_env_vars:

关键环境变量
~~~~~~~~~~~~

*   **OPTIM_TORCH** (多卡训练必需)

    在物理机上进行多卡训练时，**必须** 设置 ``OPTIM_TORCH=0``。默认配置会开启 PyTorch 的 ``expandable_segments`` 显存分配方式，该方式在 XPU 上会导致多卡 DDP 训练出现段错误（``SIGSEGV``）。Docker 镜像中已默认设置该变量。

    .. code-block:: bash

        export OPTIM_TORCH=0

*   **ZE_AFFINITY_MASK** (单机/多机均需关注)

    用于指定参与训练的 GPU 设备，作用与 ``CUDA_VISIBLE_DEVICES`` 相同。

    *   **默认行为**：如果不设置此变量，程序将使用当前节点上的**所有** GPU。
    *   **指定设备**：例如仅使用卡 0 和卡 1：

        .. code-block:: bash

            export ZE_AFFINITY_MASK=0,1

单机训练
~~~~~~~~

当可见的 GPU 多于一张时，``llamafactory-cli`` 会自动使用 ``torchrun`` 启动多卡训练。

**单机多卡示例**：

.. code-block:: bash

    ZE_AFFINITY_MASK=0,1,2,3 OPTIM_TORCH=0 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

**DeepSpeed 示例**：

.. code-block:: bash

    ZE_AFFINITY_MASK=0,1 OPTIM_TORCH=0 llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml

**FSDP2 示例**：

.. code-block:: bash

    ZE_AFFINITY_MASK=0,1 accelerate launch \
        --config_file examples/accelerate/fsdp2_config.yaml \
        src/train.py examples/train_lora/qwen3_lora_sft.yaml

训练方式
--------

常见训练场景的启动命令与 GPU 相同，例如：

.. code-block:: bash

    # PT
    llamafactory-cli train examples/train_lora/qwen3_lora_pretrain.yaml

    # RM
    llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml

    # DPO
    llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml

    # KTO
    llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml

    # Full
    llamafactory-cli train examples/train_full/qwen3_full_sft.yaml

.. _xpu_faq:

常见问题
--------

**torch.xpu.device_count() 返回 0**

- 检查当前用户是否在 ``render`` 和 ``video`` 用户组中。
- 使用 Docker 时，检查镜像中的计算运行时版本是否与宿主机内核驱动兼容。
- 检查是否误装了非 XPU 版本的 PyTorch。可以重新执行 ``pip install --force-reinstall -r requirements/xpu.txt --index-url https://download.pytorch.org/whl/xpu``。

**多卡训练出现段错误（SIGSEGV）**

请设置 ``OPTIM_TORCH=0``，详见 :ref:`关键环境变量 <xpu_env_vars>`。

**DeepSpeed 报错 ImportError: cannot import name 'DpcppBuildExtension'**

请安装 ``deepspeed>=0.18.7``。

**安装 Liger Kernel 等依赖后，Triton 无法在 XPU 上运行**

部分依赖（如 ``liger-kernel``）会安装 CUDA 版本的 ``triton`` 包，覆盖 XPU 版本的 ``triton-xpu``。请使用 ``pip install --no-deps`` 安装这些依赖，并通过 ``pip list | grep triton`` 确认只安装了 ``triton-xpu``。

**torch.compile 报错 FileNotFoundError: 'ocloc'**

请安装与计算运行时版本一致的 ``intel-ocloc``。
