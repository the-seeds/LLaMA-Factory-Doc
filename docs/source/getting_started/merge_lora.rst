模型保存、LoRA 合并与量化
#############################

模型保存
~~~~~~~~~~~~~~~~~~~~~~~

模型保存路径
^^^^^^^^^^^^

训练相关产物默认都会围绕 ``output_dir`` 组织和保存。该目录通常包含最终模型、训练过程中的 checkpoint、Trainer state、日志以及 loss 图等内容。

.. code-block:: yaml

   output_dir: saves/qwen3_8b_lora_sft

建议为不同实验设置独立的 ``output_dir``，便于区分不同配置下的训练结果，也方便后续继续训练或导出模型。

Checkpoint 保存策略
^^^^^^^^^^^^^^^^^^^^^

LlamaFactory 的 checkpoint 保存行为主要由以下参数控制：

- ``save_strategy``：控制保存策略，通常可设为 ``steps``、``epoch`` 或 ``no``。
- ``save_steps``：当 ``save_strategy: steps`` 时，每训练多少个 step 保存一次 checkpoint。
- ``save_total_limit``：最多保留多少个 checkpoint，超过数量限制时会自动删除较旧的 checkpoint。

一个按 step 定期保存 checkpoint 的示例如下：

.. code-block:: yaml

   output_dir: saves/qwen3_8b_lora_sft
   save_strategy: steps
   save_steps: 200
   save_total_limit: 3

上述配置表示训练过程中每 200 个 step 保存一次 checkpoint，并且最多保留 3 个最近的 checkpoint。

如果希望按 epoch 保存，可改为：

.. code-block:: yaml

   save_strategy: epoch

如果不希望在训练过程中保存 checkpoint，可设为：

.. code-block:: yaml

   save_strategy: no

仅保存模型权重
^^^^^^^^^^^^^^

在某些只关心最终导出权重的场景下，可以设置 ``save_only_model: true``。此时通常只保存模型权重，而不保存 optimizer、scheduler 等训练状态文件。

.. code-block:: yaml

   save_only_model: true

这样做可以减少磁盘占用，但通常不适合需要中途恢复训练的场景。因为缺少完整的训练状态，后续一般无法进行严格意义上的完整 resume。

断点续训
^^^^^^^^

如果训练过程中中断，或希望从已有 checkpoint 继续训练，可以使用 ``resume_from_checkpoint`` 指定恢复路径。

.. code-block:: yaml

   output_dir: saves/qwen3_8b_lora_sft
   resume_from_checkpoint: saves/qwen3_8b_lora_sft/checkpoint-1000

通常建议 ``resume_from_checkpoint`` 指向某个具体的 checkpoint 目录，例如 ``checkpoint-1000``。恢复训练时，Trainer 会尝试继续加载该 checkpoint 对应的模型参数以及训练状态。

.. note::

   如果此前开启了 ``save_only_model: true``，由于未保存 optimizer、scheduler 等状态，通常不能进行完整的断点续训。此时更适合将保存结果用于推理、评估或后续权重转换，而不是严格接续之前的训练进度。


LoRA 合并
~~~~~~~~~~~~~~~~~~~~~~~

当我们基于预训练模型训练好 LoRA 适配器后，我们不希望在每次推理的时候分别加载预训练模型和 LoRA 适配器，因此我们需要将预训练模型和 LoRA 适配器合并导出成一个模型，并根据需要选择是否量化。根据是否量化以及量化算法的不同，导出的配置文件也有所区别。

您可以通过 ``llamafactory-cli export merge_config.yaml`` 指令来合并模型。其中 ``merge_config.yaml`` 需要您根据不同情况进行配置。

``examples/merge_lora/qwen3_lora_sft.yaml`` 提供了合并时的配置示例。

.. code-block:: yaml

    ### examples/merge_lora/qwen3_lora_sft.yaml
    ### model
    model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
    adapter_name_or_path: saves/qwen3-4b/lora/sft
    template: qwen3_nothink
    trust_remote_code: true

    ### export
    export_dir: saves/qwen3_sft_merged
    export_size: 5
    export_device: cpu
    export_legacy_format: false


.. note::
    * 模型 ``model_name_or_path`` 需要存在且与 ``template`` 相对应。 ``adapter_name_or_path`` 需要与微调中的适配器输出路径 ``output_dir`` 相对应。
    * 合并 LoRA 适配器时，不要使用量化模型或指定量化位数。您可以使用本地或下载的未量化的预训练模型进行合并。


量化
~~~~~~~~~~~~~~~~~~~~~~~

在完成模型合并并获得完整模型后，为了优化部署效果，人们通常会基于显存占用、使用成本和推理速度等因素，选择通过量化技术对模型进行压缩，从而实现更高效的部署。

量化（Quantization）通过数据精度压缩有效地减少了显存使用并加速推理。LlamaFactory 支持多种量化方法，包括:

* AQLM
* AWQ
* GPTQ
* QLoRA
* ...

GPTQ 等后训练量化方法(Post Training Quantization)是一种在训练后对预训练模型进行量化的方法。我们通过量化技术将高精度表示的预训练模型转换为低精度的模型，从而在避免过多损失模型性能的情况下减少显存占用并加速推理，我们希望低精度数据类型在有限的表示范围内尽可能地接近高精度数据类型的表示，因此我们需要指定量化位数 ``export_quantization_bit`` 以及校准数据集 ``export_quantization_dataset``。

.. note::
    在进行模型合并时，请指定：
    
    * ``model_name_or_path``: 预训练模型的名称或路径
    * ``template``: 模型模板
    * ``export_dir``: 导出路径
    * ``export_quantization_bit``: 量化位数
    * ``export_quantization_dataset``: 量化校准数据集
    * ``export_size``: 最大导出模型文件大小
    * ``export_device``: 导出设备
    * ``export_legacy_format``: 是否使用旧格式导出

下面提供一个配置文件示例：

.. code-block:: yaml

    ### examples/merge_lora/qwen3_gptq.yaml
    ### model
    model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
    template: qwen3_nothink
    trust_remote_code: true

    ### export
    export_dir: saves/qwen3_gptq
    export_quantization_bit: 4
    export_quantization_dataset: data/c4_demo.json
    export_size: 2
    export_device: cpu
    export_legacy_format: false


QLoRA 是一种在 4-bit 量化模型基础上使用 LoRA 方法进行训练的技术。它在极大地保持了模型性能的同时大幅减少了显存占用和推理时间。

.. warning:: 
    不要使用量化模型或设置量化位数 ``quantization_bit``

下面提供一个配置文件示例：

.. code-block:: yaml

    ### examples/merge_lora/qwen3_lora_sft.yaml
    ### model
    model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
    adapter_name_or_path: saves/qwen3-4b/lora/sft
    template: qwen3_nothink
    trust_remote_code: true

    ### export
    export_dir: saves/qwen3_sft_merged
    export_size: 5
    export_device: cpu
    export_legacy_format: false
