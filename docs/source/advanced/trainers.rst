训练方法
================


Pre-training
--------------------

大语言模型通过在一个大型的通用数据集上通过无监督学习的方式进行预训练来学习语言的表征/初始化模型权重/学习概率分布。
我们期望在预训练后模型能够处理大量、多种类的数据集，进而可以通过监督学习的方式来微调模型使其适应特定的任务。


预训练时，请将 ``stage`` 设置为 ``pt`` ，并确保使用的数据集符合 :ref:`预训练数据集` 格式 。

下面提供预训练的配置示例：

.. code-block:: yaml
    
    ### model
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

    ### method
    stage: pt
    do_train: true
    finetuning_type: lora
    lora_target: all

    ### dataset
    dataset: c4_demo
    cutoff_len: 1024
    max_samples: 1000
    overwrite_cache: true
    preprocessing_num_workers: 16

    ### output
    output_dir: saves/llama3-8b/lora/sft
    logging_steps: 10
    save_steps: 500
    plot_loss: true
    overwrite_output_dir: true

    ### train
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-4
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000






Post-training
---------------------

在预训练结束后，模型的参数得到初始化，模型能够理解语义、语法以及识别上下文关系，在处理一般性任务时有着不错的表现。
尽管模型涌现出的零样本学习，少样本学习的特性使其能在一定程度上完成特定任务，
但仅通过提示（prompt）并不一定能使其表现令人满意。因此，我们需要后训练(post-training)来使得模型在特定任务上也有足够好的表现。



Supervised Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~

Supervised Fine-Tuning(监督微调)是一种在预训练模型上使用小规模有标签数据集进行训练的方法。
相比于预训练一个全新的模型，对已有的预训练模型进行监督微调是更快速更节省成本的途径。


监督微调时，请将 ``stage`` 设置为 ``sft`` 。
下面提供监督微调的配置示例：

.. code-block:: yaml

    ...
    stage: sft
    finetuning_type: lora
    ...


RLHF
~~~~~~~~~~~~~~~~~~~~~~

由于在监督微调中语言模型学习的数据来自互联网，所以模型可能无法很好地遵循用户指令，甚至可能输出非法、暴力的内容，因此我们需要将模型行为与用户需求对齐(alignment)。
通过 RLHF(Reinforcement Learning from Human Feedback) 方法，我们可以通过人类反馈来进一步微调模型，使得模型能够更好更安全地遵循用户指令。



Reward model
++++++++++++++++++++++++++++++

但是，获取真实的人类数据是十分耗时且昂贵的。一个自然的想法是我们可以训练一个奖励模型（reward model）来代替人类对语言模型的输出进行评价。
为了训练这个奖励模型，我们需要让奖励模型获知人类偏好，而这通常通过输入经过人类标注的偏好数据集来实现。
在偏好数据集中，数据由三部分组成：输入、好的回答、坏的回答。奖励模型在偏好数据集上训练，从而可以更符合人类偏好地评价语言模型的输出。

在训练奖励模型时，请将 ``stage`` 设置为 ``rm`` ，确保使用的数据集符合 :ref:`偏好数据集 <偏好数据集-1>` 格式并且指定奖励模型的保存路径。
以下提供一个示例：

.. code-block:: yaml

    ...
    stage: rm
    dataset: dpo_en_demo
    ...
    output_dir: saves/llama3-8b/lora/reward
    ...


PPO
+++++++++++++++++++++

在训练奖励完模型之后，我们可以开始进行模型的强化学习部分。与监督学习不同，在强化学习中我们没有标注好的数据。语言模型接受prompt作为输入，其输出作为奖励模型的输入。奖励模型评价语言模型的输出，并将评价返回给语言模型。确保两个模型都能良好运行是一个具有挑战性的任务。 一种实现方式是使用近端策略优化（PPO，Proximal Policy Optimization）。其主要思想是：我们既希望语言模型的输出能够尽可能地获得奖励模型的高评价，又不希望语言模型的变化过于“激进”。 通过这种方法，我们可以使得模型在学习趋近人类偏好的同时不过多地丢失其原有的解决问题的能力。

在使用 PPO 进行强化学习时，请将 ``stage`` 设置为 ``ppo``，并且指定所使用奖励模型的路径。
下面是一个示例：

.. code-block:: yaml

    ...
    stage: ppo
    reward_model: saves/llama3-8b/lora/reward
    ...



DPO
~~~~~~~~~~~~~~~~~~~~~~~~

既然同时保证语言模型与奖励模型的良好运行是有挑战性的，一种想法是我们可以丢弃奖励模型，进而直接基于人类偏好训练我们的语言模型，这大大简化了训练过程。

在使用 DPO 时，请将 ``stage`` 设置为 ``dpo``，确保使用的数据集符合 :ref:`偏好数据集-1` 格式并且设置偏好优化相关参数。
以下是一个示例：

.. code-block:: yaml

    ...
    ### method
    stage: dpo
    pref_beta: 0.1
    pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]
    dataset: dpo_en_demo
    ...


KTO
~~~~~~~~~~~~~~~~~~~~~~

KTO(Kahneman-Taversky Optimization) 的出现是为了解决成对的偏好数据难以获得的问题。 KTO使用了一种新的损失函数使其只需二元的标记数据，
即只需标注回答的好坏即可训练，并取得与 DPO 相似甚至更好的效果。

在使用 KTO 时，请将 ``stage`` 设置为 ``kto`` ，设置偏好优化相关参数并使用 KTO 数据集。

以下是一个示例：

.. code-block:: yaml

    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    ...
    stage: kto
    pref_beta: 0.1
    ...
    dataset: kto_en_demo
