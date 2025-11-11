NPU 训练
###############

支持设备
============

LLaMA-Factory当前支持以下设备，您可以通过`npu-smi info`查看具体信息。

- Atlas A2训练系列（Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2）

- Atlas 800I A2推理系列（Atlas 800I A2）


支持功能
============

.. raw:: html

   <table class="docutils" style="border-collapse: collapse; border: 1px solid #ccc;">
     <thead>
       <tr>
         <th style="border: 1px solid #ccc; padding: 8px; text-align: left;"></th>
         <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">功能</th>
         <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">限制条件</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td rowspan="4" style="border: 1px solid #ccc; padding: 8px; vertical-align: middle;"><strong>训练范式</strong></td>
         <td style="border: 1px solid #ccc; padding: 8px;">PT</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">SFT</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">RM</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">DPO</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td rowspan="3" style="border: 1px solid #ccc; padding: 8px; vertical-align: middle;"><strong>参数范式</strong></td>
         <td style="border: 1px solid #ccc; padding: 8px;">Full</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">Freeze</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">LoRA</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;"><strong>模型合并</strong></td>
         <td style="border: 1px solid #ccc; padding: 8px;">LoRA权重合并</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td rowspan="3" style="border: 1px solid #ccc; padding: 8px; vertical-align: middle;"><strong>分布式</strong></td>
         <td style="border: 1px solid #ccc; padding: 8px;">DDP</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">FSDP</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;">DeepSpeed</td>
         <td style="border: 1px solid #ccc; padding: 8px;"></td>
       </tr>
       <tr>
         <td style="border: 1px solid #ccc; padding: 8px;"><strong>加速</strong></td>
         <td style="border: 1px solid #ccc; padding: 8px;">融合算子</td>
         <td style="border: 1px solid #ccc; padding: 8px;">当前仅支持NPU FA融合算子</td>
       </tr>
     </tbody>
   </table>


单机微调
============

以 ``davinci0`` 单卡为例，下载并使用ascend llamafactory镜像。

首先在环境当前目录下执行如下命令，进入容器。

.. code-block::

  docker pull quay.io/ascend/llamafactory:0.9.4-npu-a2

  docker run -it \
    -v $PWD/hf_cache:/root/.cache/huggingface \
    -v $PWD/ms_cache:/root/.cache/modelscope \
    -v $PWD/data:/app/data \
    -v $PWD/output:/app/output \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $PWD/Test:/home/Test/ \
    -p 7860:7860 \
    -p 8000:8000 \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --shm-size 16G \
    --name llamafactory   quay.io/ascend/llamafactory:0.9.4-npu-a2 bash

如果在单机上使用多卡微调时，可使用 ``--device /dev/davinci1, --device /dev/davinci2, ...`` 来增加 NPU 卡。

.. note::

    昇腾 NPU 卡从 0 开始编号，docker 容器内也是如此；

    如映射物理机上的 davinci6，davinci7 NPU 卡到容器内使用，其对应的卡号分别为 0，1


进入docker后安装相关依赖、设置环境变量、配置 LoRA 微调参数文件(qwen1_5_lora_sft_ds.yaml)

.. note::

  deepspeed用于训练优化

  modelscope用于模型下载

  ASCEND_RT_VISIBLE_DEVICES=0指定使用容器内卡号

  USE_MODELSCOPE_HUB=1使用modelscope  

.. code-block::
  
  pip install -e ".[deepspeed,modelscope]" -i https://pypi.tuna.tsinghua.edu.cn/simple

  export ASCEND_RT_VISIBLE_DEVICES=0
  export USE_MODELSCOPE_HUB=1

在 LLAMA-Factory 目录下，创建如下 qwen1_5_lora_sft_ds.yaml：

.. code-block::

    model_name_or_path: qwen/Qwen1.5-7B

    ### method
    stage: sft
    do_train: true
    finetuning_type: lora
    lora_target: q_proj,v_proj

    ### ddp
    ddp_timeout: 180000000
    deepspeed: examples/deepspeed/ds_z0_config.json

    ### dataset
    dataset: identity,alpaca_en_demo
    template: qwen
    cutoff_len: 1024
    max_samples: 1000
    overwrite_cache: true
    preprocessing_num_workers: 16

    ### output
    output_dir: saves/Qwen1.5-7B/lora/sft
    logging_steps: 10
    save_steps: 500
    plot_loss: true
    overwrite_output_dir: true

    ### train
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 2
    learning_rate: 0.0001
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    fp16: true

    ### eval
    val_size: 0.1
    per_device_eval_batch_size: 1
    eval_strategy: steps
    eval_steps: 500

使用 torchrun 启动 LoRA 微调，如正常输出模型加载、损失 loss 等日志，即说明成功微调。

.. code-block:: shell

  torchrun --nproc_per_node 1 \
      --nnodes 1 \
      --node_rank 0 \
      --master_addr 127.0.0.1 \
      --master_port 7007 \
      src/train.py qwen1_5_lora_sft_ds.yaml

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


经 LoRA 微调后，通过 ``llamafactory-cli chat`` 使用微调后的模型进行交互对话，使用 Ctrl+C 或输入 exit 退出该问答聊天。

.. code-block:: shell

  llamafactory-cli chat --model_name_or_path qwen/Qwen1.5-7B \
              --adapter_name_or_path saves/Qwen1.5-7B/lora/sft \
              --template qwen \
              --finetuning_type lora

部分交互对话输出如下所示:

.. code-block:: shell

  ...
  [INFO|configuration_utils.py:1135] 2025-07-21 06:31:19,166 >> Generate config GenerationConfig {
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "max_new_tokens": 2048
  }

  [INFO|2025-07-21 06:31:19] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
  [INFO|2025-07-21 06:31:19] llamafactory.model.adapter:143 >> Merged 1 adapter(s).
  [INFO|2025-07-21 06:31:19] llamafactory.model.adapter:143 >> Loaded adapter(s): saves/Qwen1.5-7B/lora/sft
  [INFO|2025-07-21 06:31:19] llamafactory.model.loader:143 >> all params: 7,721,324,544
  Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.

  User: 帮我定制一个减肥计划
  Assistant: 当然可以，以下是一个简单的减肥计划，供您参考：

  1. 制定一个目标：首先要明确自己的减肥目标，是想在一个月内减掉5斤还是10斤，或者更久。制定目标后，可以更清晰地制定减肥计划。

  2. 控制饮食：减肥最重要的就是控制饮食，少吃高热量的食物，增加蔬菜、水果和蛋白质的摄入。可以适当减少碳水化合物的摄入，但不要完全戒掉，以免影响身体的能量。

  3. 增加运动量：除了控制饮食，增加运动量也是减肥的关键。可以每天进行30分钟的有氧运动，比如快走、跑步、游泳等，也可以增加力量训练来增强肌肉。

  4. 控制饮酒：酒精是高热量的饮料，容易导致体重增加。所以要控制饮酒量，尽量少喝或者不喝。

  5. 规律作息：保持规律的作息可以有助于身体代谢的正常运转，也可以提高身体的免疫力。

  6. 每天记录体重和饮食：每天记录体重和饮食可以帮助您更好地掌握自己的减肥进度，及时调整计划。

  以上是一个简单的减肥计划，但请注意，减肥需要持之以恒，不能急于求成，建议您在制定计划前咨询专业医生或营养师的建议。


多机微调
============

多机微调时，不建议使用容器部署方式（单机都不够用的情况下，起多个容器资源更加紧张），请直接在每个节点安装 llamafactory（请参考 :doc:`NPU <./npu_installation>` 中的安装步骤），同时仍需要安装 DeepSpeed 和 ModelScope：

.. code-block::

  pip install -e ".[deepspeed,modelscope]" -i https://pypi.tuna.tsinghua.edu.cn/simple


安装成功后，请在每个节点上使用 ``export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`` 显式指定所需的 NPU 卡号，不指定时默认使用当前节点的所有 NPU 卡。

然后，必须在每个节点上使用 ``export HCCL_SOCKET_IFNAME=eth0`` 来指定当前节点的 HCCL 通信网卡（请使用目标网卡名替换 ``eth0``）。

以两机环境为例，分别在主、从节点（机器）上执行如下两条命令即可启动多机训练：

.. code-block:: shell

    # 在主节点执行如下命令，设置 rank_id = 0
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml
    
    # 在从节点执行如下命令，设置 rank_id = 1
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml

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