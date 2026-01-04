.. _分布训练:

分布训练
==================

LLaMA-Factory 支持单机多卡和多机多卡分布式训练。同时也支持 :ref:`DDP <NativeDDP>`、:ref:`DeepSpeed <deepspeed_ref>` 和 :ref:`FSDP <fsdp_ref>` 三种分布式引擎。


`DDP <https://pytorch.org/docs/stable/notes/ddp.html>`__ (DistributedDataParallel) 通过实现模型并行和数据并行实现训练加速。
使用 DDP 的程序需要生成多个进程并且为每个进程创建一个 DDP 实例，他们之间通过 ``torch.distributed`` 库同步。

`DeepSpeed <https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/>`__ 是微软开发的分布式训练引擎，并提供ZeRO（Zero Redundancy Optimizer）、offload、Sparse Attention、1 bit Adam、流水线并行等优化技术。
您可以根据任务需求与设备选择使用。

`FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ 通过全切片数据并行技术（Fully Sharded Data Parallel）来处理更多更大的模型。在 DDP 中，每张 GPU 都各自保留了一份完整的模型参数和优化器参数。而 FSDP 切分了模型参数、梯度与优化器参数，使得每张 GPU 只保留这些参数的一部分。
除了并行技术之外，FSDP 还支持将模型参数卸载至CPU，从而进一步降低显存需求。

`FSDP2 <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ 在集成FSDP1基础功能的前提下，摒弃了 FSDP1 将参数压扁拼接的做法，转而基于 DTensor 实现逐参数切分，这一架构升级在完整保留模型原始结构的同时，显著提升了计算与通信的重叠效率，进而增强训练性能。


.. list-table::
    :widths: 30 30 30 30 30
    :header-rows: 1

    * - 引擎
      - 数据切分
      - 模型切分
      - 优化器切分
      - 参数卸载
    * - DDP
      - 支持
      - 不支持
      - 不支持
      - 不支持
    * - DeepSpeed
      - 支持
      - 支持
      - 支持
      - 支持
    * - FSDP
      - 支持
      - 支持
      - 支持
      - 支持
    * - FSDP2
      - 支持
      - 支持
      - 支持
      - 支持


.. 单机多卡
.. ------------------------

.. * :ref:`NativeDDP单机多卡 <torchrun单机多卡>`

.. * :ref:`fsdp单机多卡 <fsdp单机多卡>`

.. * :ref:`deepspeed单机多卡 <deepspeed单机多卡>`


.. 多机多卡
.. -----------------------------
.. * :ref:`NativeDDP多机多卡 <torchrun多机多卡>`
.. * :ref:`fsdp多机多卡 <fsdp多机多卡>`
.. * :ref:`deepspeed多机多卡 <deepspeed多机多卡>`



.. _NativeDDP:

NativeDDP
~~~~~~~~~~~~~~~~~~~~~~~~~

NativeDDP 是 PyTorch 提供的一种分布式训练方式，您可以通过以下命令启动训练：

.. _torchrun:

.. torchrun
.. ~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torchrun单机多卡:

单机多卡
+++++++++++++++++++

llamafactory-cli
***************************

您可以使用 llamafactory-cli 启动 NativeDDP 引擎。

.. code-block:: bash

    FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml

如果 ``CUDA_VISIBLE_DEVICES`` 没有指定，则默认使用所有GPU。如果需要指定GPU，例如第0、1个GPU，可以使用：

.. code-block:: bash

    FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/config1.yaml



torchrun
*******************************
您也可以使用 ``torchrun`` 指令启动 NativeDDP 引擎进行单机多卡训练。下面提供一个示例：

.. code-block:: bash

    torchrun  --standalone --nnodes=1 --nproc-per-node=8  src/train.py \
    --stage sft \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct  \
    --do_train \
    --dataset alpaca_en_demo \
    --template llama3 \
    --finetuning_type lora \
    --output_dir  saves/llama3-8b/lora/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16



accelerate
***************************
您还可以使用 ``accelerate`` 指令启动进行单机多卡训练。

首先运行以下命令，根据需求回答一系列问题后生成配置文件：


.. code-block:: bash

    accelerate config



下面提供一个示例配置文件：

.. code-block:: yaml

    # accelerate_singleNode_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: true
    distributed_type: MULTI_GPU
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    gpu_ids: all
    machine_rank: 0
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false


您可以通过运行以下指令开始训练:

.. code-block:: bash

    accelerate launch \
    --config_file accelerate_singleNode_config.yaml \
    src/train.py training_config.yaml

.. _torchrun多机多卡:

多机多卡
++++++++++++++++++++

llamafactory-cli
*******************

.. code-block:: bash

    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
    
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml


.. list-table::
    :widths: 30 70  
    :header-rows: 1

    * - 变量名
      - 介绍
    * - FORCE_TORCHRUN
      - 是否强制使用torchrun
    * - NNODES
      - 节点数量
    * - NODE_RANK
      - 各个节点的rank。
    * - MASTER_ADDR
      - 主节点的地址。
    * - MASTER_PORT
      - 主节点的端口。

torchrun
******************************

您也可以使用 ``torchrun`` 指令启动 NativeDDP 引擎进行多机多卡训练。

.. code-block:: bash
    
    torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=0  \
    --master_addr=192.168.0.1  train.py
    torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=1  \
    --master_addr=192.168.0.1  train.py

accelerate
***************************
您还可以使用 ``accelerate`` 指令启动进行多机多卡训练。

首先运行以下命令，根据需求回答一系列问题后生成配置文件：

.. code-block:: bash

    accelerate config

下面提供一个示例配置文件：

.. code-block:: yaml

    # accelerate_multiNode_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: true
    distributed_type: MULTI_GPU
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    gpu_ids: all
    machine_rank: 0
    main_process_ip: '192.168.0.1'
    main_process_port: 29500
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false


您可以通过运行以下指令开始训练:

.. code-block:: bash

    accelerate launch \
    --config_file accelerate_multiNode_config.yaml \
    train.py llm_config.yaml




.. _deepspeed_ref:


DeepSpeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DeepSpeed 是由微软开发的一个开源深度学习优化库，旨在提高大模型训练的效率和速度。在使用 DeepSpeed 之前，您需要先估计训练任务的显存大小，再根据任务需求与资源情况选择合适的 ZeRO 阶段。

* ZeRO-1: 仅划分优化器参数，每个GPU各有一份完整的模型参数与梯度。
* ZeRO-2: 划分优化器参数与梯度，每个GPU各有一份完整的模型参数。
* ZeRO-3: 划分优化器参数、梯度与模型参数。

.. 关于 :ref:`显存估计`

简单来说：从 ZeRO-1 到 ZeRO-3，阶段数越高，显存需求越小，但是训练速度也依次变慢。此外，设置 ``offload_param=cpu`` 参数会大幅减小显存需求，但会极大地使训练速度减慢。因此，如果您有足够的显存，
应当使用 ZeRO-1，并且确保 ``offload_param=none``。

LLaMA-Factory提供了使用不同阶段的 DeepSpeed 配置文件的示例。包括：

* :ref:`ZeRO-0` (不开启)
* :ref:`ZeRO-2`
* :ref:`ZeRO-2+offload <zero2O>`
* :ref:`ZeRO-3`
* :ref:`ZeRO-3+offload <zero3O>`
* :ref:`ZeRO-1/2+AutoTP <zero1_2_autotp>`

.. note::
    `https://huggingface.co/docs/transformers/deepspeed <https://huggingface.co/docs/transformers/deepspeed/>`_ 提供了更为详细的介绍。



.. _deepspeed单机多卡:

单机多卡
++++++++++++++++++++++

llamafactory-cli
*********************

您可以使用 llamafactory-cli 启动 DeepSpeed 引擎进行单机多卡训练。

.. code-block:: bash

    FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml

为了启动 DeepSpeed 引擎，配置文件中 ``deepspeed`` 参数指定了 DeepSpeed 配置文件的路径:

.. code-block:: yaml 

    ...
    deepspeed: examples/deepspeed/ds_z3_config.json
    ...


deepspeed
**************************

您也可以使用 ``deepspeed`` 指令启动 DeepSpeed 引擎进行单机多卡训练。

.. code-block:: bash

    deepspeed --include localhost:1 your_program.py <normal cl args> --deepspeed ds_config.json

下面是一个例子：

.. code-block:: bash

    deepspeed --num_gpus 8 src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct  \
    --do_train \
    --dataset alpaca_en \
    --template llama3 \
    --finetuning_type full \
    --output_dir  saves/llama3-8b/lora/full \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16


.. note:: 

    使用 ``deepspeed`` 指令启动 DeepSpeed 引擎时您无法使用 ``CUDA_VISIBLE_DEVICES`` 指定GPU。而需要：

    .. code-block:: bash

        deepspeed --include localhost:1 your_program.py <normal cl args> --deepspeed ds_config.json
    
    ``--include localhost:1`` 表示只是用本节点的gpu1。

.. _deepspeed多机多卡:

多机多卡
+++++++++++++++++++++


LLaMA-Factory 支持使用 DeepSpeed 的多机多卡训练，您可以通过以下命令启动：

.. code-block:: bash

    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml


deepspeed
******************************

您也可以使用 ``deepspeed`` 指令来启动多机多卡训练。

.. code-block:: bash

    deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
    your_program.py <normal cl args> --deepspeed ds_config.json





.. note::

    * 关于hostfile:
        hostfile的每一行指定一个节点，每行的格式为 ``<hostname> slots=<num_slots>`` ，
        其中 ``<hostname>`` 是节点的主机名， ``<num_slots>`` 是该节点上的GPU数量。下面是一个例子：
        .. code-block:: 

            worker-1 slots=4
            worker-2 slots=4

        请在 `https://www.deepspeed.ai/getting-started/ <https://www.deepspeed.ai/getting-started/>`_ 了解更多。
    
    * 如果没有指定 ``hostfile`` 变量, DeepSpeed 会搜索 ``/job/hostfile`` 文件。如果仍未找到，那么 DeepSpeed 会使用本机上所有可用的GPU。

accelerate
******************

您还可以使用 ``accelerate`` 指令启动 DeepSpeed 引擎。
首先通过以下命令生成 DeepSpeed 配置文件：

.. code-block:: bash

    accelerate config

下面提供一个配置文件示例：

.. code-block:: yaml

    # deepspeed_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
        deepspeed_multinode_launcher: standard
        gradient_accumulation_steps: 8
        offload_optimizer_device: none
        offload_param_device: none
        zero3_init_flag: false
        zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_process_ip: '192.168.0.1'
    main_process_port: 29500
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

随后，您可以使用以下命令启动训练：

.. code-block:: bash

    accelerate launch \
    --config_file deepspeed_config.yaml \
    train.py llm_config.yaml



DeepSpeed 配置文件
++++++++++++++++++++++

.. _ZeRO-0:

ZeRO-0
*************************

.. code-block:: yaml

    ### ds_z0_config.json
    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": true,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 0,
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": true,
            "round_robin_gradients": true
        }
    }



.. _ZeRO-2:


ZeRO-2
**************************

只需在 ZeRO-0 的基础上修改 ``zero_optimization`` 中的 ``stage`` 参数即可。

.. code-block:: yaml

    ### ds_z2_config.json
    {
        ...
        "zero_optimization": {
            "stage": 2,
        ...
        }
    }



.. _zero2O:


ZeRO-2+offload
*************************


只需在 ZeRO-0 的基础上在 ``zero_optimization`` 中添加 ``offload_optimizer`` 参数即可。


.. code-block:: yaml

    ### ds_z2_offload_config.json
    {
        ...
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
            },
        ...
        }
    }



.. _ZeRO-3:

ZeRO-3
****************************

只需在 ZeRO-0 的基础上修改 ``zero_optimization`` 中的参数。

.. code-block:: yaml

    ### ds_z3_config.json
    {
        ...
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": true
        }
    }


.. _zero3O:

ZeRO-3+offload
*****************************

只需在 ZeRO-3 的基础上添加 ``zero_optimization`` 中的 ``offload_optimizer`` 和 ``offload_param`` 参数即可。

.. code-block:: yaml


    ### ds_z3_offload_config.json
    {
        ...
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
            },
            "offload_param": {
            "device": "cpu",
            "pin_memory": true
            },
        ...
        }
    }


.. _zero1_2_autotp:

ZeRO-1/2+AutoTP
*************************

TP是一种重要的大模型训练内存优化技术。以往在 Hugging Face Trainer 中，模型扩展主要依赖 ZeRO/FSDP 的分片数据并行，其中 ZeRO3 虽然节省显存，但通信开销较大；ZeRO1/2 通信开销较低，却在超大模型场景下受限于显存容量。

为此，DeepSpeed 引入了 **原生自动张量并行（AutoTP）训练**，并正式支持 Hugging Face Transformers。该能力可与 ZeRO 结合，在训练阶段实现更优的性能与内存平衡，带来以下优势：

* 在比 FSDP / ZeRO3 更低通信开销的情况下实现大模型扩展（如 AutoTP + ZeRO1 达到接近 ZeRO3 的显存节省效果）
* 支持更大的 batch size，提高训练吞吐
* 支持更长上下文长度，拓展应用场景

目前 AutoTP 训练已支持 **ZeRO1 和 ZeRO2**。该功能已在 **DeepSpeed ≥ 0.16.4** 中提供。

具体的参数配置如下：

.. code-block:: yaml

    ### ds2_autotp.json
    {
        "ZeRO_optimization": {
            "stage": 2,
            ...
        },
        "tensor_parallel":{
            "autotp_size": 4
        },
    }

当前模型支持受限，具体支持列表请查看 `AutoTP支持模型列表 <https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/#supported-models>`__ 。


.. note:: 

    `https://www.deepspeed.ai/docs/config-json/ <https://www.deepspeed.ai/docs/config-json/>`_ 提供了关于deepspeed配置文件的更详细的介绍。




.. _fsdp_ref:

FSDP
~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch 的全切片数据并行技术 `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_ （Fully Sharded Data Parallel）能让我们处理更多更大的模型。LLaMA-Factory支持使用 FSDP 引擎进行分布式训练。

FSDP 的参数 ``ShardingStrategy`` 的不同取值决定了模型的划分方式：

* ``FULL_SHARD``: 将模型参数、梯度和优化器状态都切分到不同的GPU上，类似ZeRO-3。
* ``SHARD_GRAD_OP``: 将梯度、优化器状态切分到不同的GPU上，每个GPU仍各自保留一份完整的模型参数。类似ZeRO-2。 
* ``NO_SHARD``: 不切分任何参数。类似ZeRO-0。


.. _fsdp单机多卡:

单机多卡
++++++++++++++++++++++

llamafactory-cli
******************

您只需根据需要修改 ``examples/accelerate/fsdp_config.yaml`` 以及 ``examples/extras/fsdp_qlora/llama3_lora_sft.yaml`` ，文件然后运行以下命令即可启动 FSDP+QLoRA 微调：

.. code-block:: bash

    bash examples/extras/fsdp_qlora/train.sh



accelerate
******************


此外，您也可以使用 accelerate 启动 FSDP 引擎， **节点数与 GPU 数可以通过 num_machines 和  num_processes 指定**。对此，Huggingface 提供了便捷的配置功能。
只需运行：

.. code-block:: bash

    accelerate config


根据提示回答一系列问题后，我们就可以生成 FSDP 所需的配置文件。

当然您也可以根据需求自行配置 ``fsdp_config.yaml`` 。

.. code-block:: yaml

    ### /examples/accelerate/fsdp_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
        fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
        fsdp_backward_prefetch: BACKWARD_PRE
        fsdp_forward_prefetch: false
        fsdp_cpu_ram_efficient_loading: true
        fsdp_offload_params: true # offload may affect training speed
        fsdp_sharding_strategy: FULL_SHARD
        fsdp_state_dict_type: FULL_STATE_DICT
        fsdp_sync_module_states: true
        fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: fp16 # or bf16
    num_machines: 1 # the number of nodes
    num_processes: 2 # the number of GPUs in all nodes
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

.. note:: 
    * 请确保 ``num_processes`` 和实际使用的总GPU数量一致 


随后，您可以使用以下命令启动训练：

.. code-block:: bash

    accelerate launch \
    --config_file fsdp_config.yaml \
    src/train.py llm_config.yaml

.. warning:: 

    不要在 FSDP+QLoRA 中使用 GPTQ/AWQ 模型


.. _fsdp多机多卡:

多机多卡
++++++++++++++++++++++

accelerate
******************

您可以通过 `accelerate config` 根据提示回答一系列问题后，生成 多机 FSDP 所需的配置文件。

当然您也可以根据需求自行配置 fsdp_config.yaml 。

.. code-block:: yaml

    #examples/accelerate/fsdp_config_multiple_nodes.yaml
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
    mixed_precision: bf16  # or fp16
    main_process_ip: 192.168.0.1
    main_process_port: 29500
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

这份yaml文件里，您主要需要注意配置的参数是以下四个：

- num_machines: 节点（机器）的数量。
- num_processes: 所有节点上的 GPU 总数，即 num_machines * num_processes_per_machine。
- main_process_ip: 主进程所在节点的 IP 地址；请确保所有节点使用相同的 IP。
- main_process_port: 主进程的端口号；请确保所有节点使用相同的端口。
- machine_rank: 当前节点（机器）的编号，从 0 开始；并且主节点（main_process_ip）的 machine_rank 必须为 0。

当配置完成后，在所有机器上运行如下命令即可启动FSDP的多机多卡训练：

.. code-block:: shell

    accelerate launch \
    --config_file fsdp_config_multiple_nodes.yaml \
    train.py llm_config.yaml


.. _fsdp2_ref:

FSDP2
~~~~~~~~~~~~~~~~~~~~~~~~~

当前LLamafactory基于Accelerate集成使用FSDP2的分布式训练，与FSDP的差异主要在于参数配置的不同。您可以采用FSDP相同的方式运行单机多卡和多机多卡。我们也为您提供了通用的入参配置：

.. code-block:: shell

    #examples/accelerate/fsdp2_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
      fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
      fsdp_cpu_ram_efficient_loading: true
      fsdp_offload_params: false
      fsdp_reshard_after_forward: true
      fsdp_state_dict_type: FULL_STATE_DICT
      fsdp_version: 2
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16  # or fp16
    num_machines: 1  # the number of nodes
    num_processes: 2  # the number of GPUs in all nodes
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

您可以通过以下命令快速拉起训练脚本：

.. code-block:: shell

    accelerate launch \
    --config_file fsdp2_config.yaml \
    train.py llm_config.yaml


更多的入参配置差异，您可以参考 `Accelerate FSDP1 vs FSDP2 <chttps://huggingface.co/docs/accelerate/main/en/concept_guides/fsdp1_vs_fsdp2>`__ 。


.. _显存估计:

.. 显存估计
.. +++++++++++++++++


.. 
.. TODO