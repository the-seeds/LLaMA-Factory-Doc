参数介绍
======================


微调参数
-----------------------------

基本参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: FinetuningArguments
   :widths: 10 30 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - pure_bf16
     - bool
     - 是否以纯 bf16 精度训练模型（不使用 AMP）。
     - False
   * - stage
     - Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]
     - 训练阶段
     - sft
   * - finetuning_type
     - Literal["lora", "freeze", "full"]
     - 微调方法
     - lora
   * - use_llama_pro
     - bool
     - 是否仅训练扩展块中的参数（LLaMA Pro 模式）。
     - False
   * - use_adam_mini
     - bool
     - 是否使用 Adam-mini 优化器。
     - False
   * - freeze_vision_tower
     - bool
     - MLLM 训练时是否冻结视觉塔。
     - True
   * - freeze_multi_modal_projector
     - bool
     - MLLM 训练时是否冻结多模态投影器。
     - True
   * - train_mm_proj_only
     - bool
     - 是否仅训练多模态投影器。
     - False
   * - compute_accuracy
     - bool
     - 是否在评估时计算 token 级别的准确率。
     - False
   * - disable_shuffling
     - bool
     - 是否禁用训练集的随机打乱。
     - False
   * - plot_loss
     - bool
     - 是否保存训练过程中的损失曲线。
     - False
   * - include_effective_tokens_per_second
     - bool
     - 是否计算有效的每秒 token 数。
     - False


LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: LoraArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - additional_target
     - Optional[str]
     - 除 LoRA 层之外设置为可训练并保存在最终检查点中的模块名称。使用逗号分隔多个模块。
     - None
   * - lora_alpha
     - Optional[int]
     - LoRA 缩放系数。一般情况下为 lora_rank * 2。
     - None
   * - lora_dropout
     - float
     - LoRA 微调中的 dropout 率。
     - 0
   * - lora_rank
     - int
     - LoRA 微调的本征维数 ``r``，``r`` 越大可训练的参数越多。
     - 8
   * - lora_target
     - str
     - 应用 LoRA 方法的模块名称。使用逗号分隔多个模块，使用 ``all`` 指定所有模块。
     - all
   * - loraplus_lr_ratio
     - Optional[float]
     - LoRA+ 学习率比例(``λ = ηB/ηA``)。 ``ηA, ηB`` 分别是 adapter matrices A 与 B 的学习率。
     - None
   * - loraplus_lr_embedding
     - Optional[float]
     - LoRA+ 嵌入层的学习率。
     - 1e-6
   * - use_rslora
     - bool
     - 是否使用秩稳定 LoRA (Rank-Stabilized LoRA)。
     - False
   * - use_dora
     - bool
     - 是否使用权重分解 LoRA（Weight-Decomposed LoRA）。
     - False
   * - pissa_init
     - bool
     - 是否初始化 PiSSA 适配器。
     - False
   * - pissa_iter
     - Optional[int]
     - PiSSA 中 FSVD 执行的迭代步数。使用 ``-1`` 将其禁用。
     - 16
   * - pissa_convert
     - bool
     - 是否将 PiSSA 适配器转换为正常的 LoRA 适配器。
     - False
   * - create_new_adapter
     - bool
     - 是否创建一个具有随机初始化权重的新适配器。
     - False



RLHF
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: RLHF训练参数介绍
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - pref_beta
     - float
     - 偏好损失中的 beta 参数。
     - 0.1
   * - pref_ftx
     - float
     - DPO 训练中的 sft loss 系数。
     - 0.0
   * - pref_loss
     - Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"]
     - DPO 训练中使用的偏好损失类型。可选值为： sigmoid, hinge, ipo, kto_pair, orpo, simpo。
     - sigmoid
   * - dpo_label_smoothing
     - float
     - 标签平滑系数，取值范围为 [0,0.5]。
     - 0.0
   * - kto_chosen_weight
     - float
     - KTO 训练中 chosen 标签 loss 的权重。
     - 1.0
   * - kto_rejected_weight
     - float
     - KTO 训练中 rejected 标签 loss 的权重。
     - 1.0
   * - simpo_gamma
     - float
     - SimPO 损失中的 reward margin。
     - 0.5
   * - ppo_buffer_size
     - int
     - PPO 训练中的 mini-batch 大小。
     - 1
   * - ppo_epochs
     - int
     - PPO 训练迭代次数。
     - 4
   * - ppo_score_norm
     - bool
     - 是否在 PPO 训练中使用归一化分数。
     - False
   * - ppo_target
     - float
     - PPO 训练中自适应 KL 控制的目标 KL 值。
     - 6.0
   * - ppo_whiten_rewards
     - bool
     - PPO 训练中是否对奖励进行归一化。
     - False
   * - ref_model
     - Optional[str]
     - PPO 或 DPO 训练中使用的参考模型路径。
     - None
   * - ref_model_adapters
     - Optional[str]
     - 参考模型的适配器路径。
     - None
   * - ref_model_quantization_bit
     - Optional[int]
     - 参考模型的量化位数，支持 4 位或 8 位量化。
     - None
   * - reward_model
     - Optional[str]
     - PPO 训练中使用的奖励模型路径。
     - None
   * - reward_model_adapters
     - Optional[str]
     - 奖励模型的适配器路径。
     - None
   * - reward_model_quantization_bit
     - Optional[int]
     - 奖励模型的量化位数。
     - None
   * - reward_model_type
     - Literal["lora", "full", "api"]
     - PPO 训练中使用的奖励模型类型。可选值为： lora, full, api。
     - lora




Freeze
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: FreezeArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - freeze_trainable_layers
     - int
     - 可训练层的数量。正数表示最后 n 层被设置为可训练的，负数表示前 n 层被设置为可训练的。
     - 2
   * - freeze_trainable_modules
     - str
     - 可训练层的名称。使用 all 来指定所有模块。
     - all
   * - freeze_extra_modules
     - Optional[str]
     - 除了隐藏层外可以被训练的模块名称，被指定的模块将会被设置为可训练的。使用逗号分隔多个模块。
     - None


Apollo
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: ApolloArguments
   :widths: 20 25 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - use_apollo
     - bool
     - 是否使用 APOLLO 优化器。
     - False
   * - apollo_target
     - str
     - 适用 APOLLO 的模块名称。使用逗号分隔多个模块，使用 `all` 指定所有线性模块。
     - all
   * - apollo_rank
     - int
     - APOLLO 梯度的秩。
     - 16
   * - apollo_update_interval
     - int
     - 更新 APOLLO 投影的步数间隔。
     - 200
   * - apollo_scale
     - float
     - APOLLO 缩放系数。
     - 32.0
   * - apollo_proj
     - Literal["svd", "random"]
     - APOLLO 低秩投影算法类型（svd 或 random）。
     - random
   * - apollo_proj_type
     - Literal["std", "right", "left"]
     - APOLLO 投影类型。
     - std
   * - apollo_scale_type
     - Literal["channel", "tensor"]
     - APOLLO 缩放类型（channel 或 tensor）。
     - channel
   * - apollo_layerwise
     - bool
     - 是否启用层级更新以进一步节省内存。
     - False
   * - apollo_scale_front
     - bool
     - 是否在梯度缩放前使用范数增长限制器。
     - False


BAdam
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: BAdamArgument
   :widths: 30 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - use_badam
     - bool
     - 是否使用 BAdam 优化器。
     - False
   * - badam_mode
     - Literal
     - BAdam 的使用模式，可选值为 layer 或 ratio。
     - layer
   * - badam_start_block
     - Optional[int]
     - layer-wise BAdam 的起始块索引。
     - None
   * - badam_switch_mode
     - Optional[Literal]
     - layer-wise BAdam 中块更新策略，可选值有： ascending, descending, random, fixed。
     - ascending
   * - badam_switch_interval
     - Optional[int]
     - layer-wise BAdam 中块更新步数间隔。使用 -1 禁用块更新。
     - 50
   * - badam_update_ratio
     - float
     - ratio-wise BAdam 中的更新比例。
     - 0.05
   * - badam_mask_mode
     - Literal
     - BAdam 优化器的掩码模式，可选值为 adjacent 或 scatter。
     - adjacent
   * - badam_verbose
     - int
     - BAdam 优化器的详细输出级别，0 表示无输出，1 表示输出块前缀，2 表示输出可训练参数。
     - 0


GaLore
~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: GaLoreArguments
   :widths: 30 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - use_galore
     - bool
     - 是否使用 GaLore 算法。
     - False
   * - galore_target
     - str
     - 应用 GaLore 的模块名称。使用逗号分隔多个模块，使用 all 指定所有线性模块。
     - all
   * - galore_rank
     - int
     - GaLore 梯度的秩。
     - 16
   * - galore_update_interval
     - int
     - 更新 GaLore 投影的步数间隔。
     - 200
   * - galore_scale
     - float
     - GaLore 的缩放系数。
     - 0.25
   * - galore_proj_type
     - Literal
     - GaLore 投影的类型，可选值有： std, reverse_std, right, left, full。
     - std
   * - galore_layerwise
     - bool
     - 是否启用逐层更新以进一步节省内存。
     - False



数据参数
------------------------
.. list-table:: DataArguments
   :widths: 10 10 50 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - template
     - Optional[str]
     - 训练和推理时构造 prompt 的模板。
     - None
   * - dataset
     - Optional[str]
     - 用于训练的数据集名称。使用逗号分隔多个数据集。
     - None
   * - eval_dataset
     - Optional[str]
     - 用于评估的数据集名称。使用逗号分隔多个数据集。
     - None
   * - eval_on_each_dataset
     - Optional[bool]
     - 是否在每个评估数据集上分开计算loss，默认concate后为整体计算。
     - False
   * - dataset_dir
     - Union[str, Dict[str, Any]]
     - 存储数据集的文件夹路径，可以是字符串或字典。当为字符串时，表示数据集目录的路径，例如 `data <https://github.com/hiyouga/LLaMA-Factory/tree/main/data>`_ ；当为字典时，将覆盖默认从本地 `dataset_info.json <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json>`_ 加载的行为。
     - data
   * - media_dir
     - Optional[str]
     - 存储图像、视频或音频的文件夹路径。如果未指定，默认为 dataset_dir。
     - None
   * - data_shared_file_system
     - Optional[bool]
     - 多机多卡时，不同机器存放数据集的路径是否是共享文件系统。数据集处理在该值为true时只在第一个node发生，为false时在每个node都处理一次。
     - false
   * - cutoff_len
     - int
     - 输入的最大 token 数，超过该长度会被截断。
     - 2048
   * - train_on_prompt
     - bool
     - 是否在输入 prompt 上进行训练。
     - False
   * - mask_history
     - bool
     - 是否仅使用当前对话轮次进行训练。
     - False
   * - streaming
     - bool
     - 是否启用数据流模式。
     - False
   * - buffer_size
     - int
     - 启用 streaming 时用于随机选择样本的 buffer 大小。
     - 16384
   * - mix_strategy
     - Literal["concat", "interleave_under", "interleave_over"]
     - 数据集混合策略，支持 concat、 interleave_under、 interleave_over。
     - concat
   * - interleave_probs
     - Optional[str]
     - 使用 interleave 策略时，指定从多个数据集中采样的概率。多个数据集的概率用逗号分隔。
     - None
   * - overwrite_cache
     - bool
     - 是否覆盖缓存的训练和评估数据集。
     - False
   * - preprocessing_batch_size
     - int
     - 预处理时每批次的示例数量。
     - 1000
   * - preprocessing_num_workers
     - Optional[int]
     - 预处理时使用的进程数量。
     - None
   * - max_samples
     - Optional[int]
     - 每个数据集的最大样本数：设置后，每个数据集的样本数将被截断至指定的 max_samples。
     - None
   * - eval_num_beams
     - Optional[int]
     - 模型评估时的 num_beams 参数。
     - None
   * - ignore_pad_token_for_loss
     - bool
     - 计算 loss 时是否忽略 pad token。
     - True
   * - val_size
     - float
     - 验证集相对所使用的训练数据集的大小。取值在 [0,1) 之间。启用 streaming 时 val_size 应是整数。
     - 0.0
   * - packing
     - Optional[bool]
     - 是否启用 sequences packing。预训练时默认启用。
     - None
   * - neat_packing
     - bool
     - 是否启用不使用 cross-attention 的 sequences packing。
     - False
   * - tool_format
     - Optional[str]
     - 用于构造函数调用示例的格式。
     - None
   * - tokenized_path
     - Optional[str]
     - Tokenized datasets的保存或加载路径。如果路径存在，会加载已有的 tokenized datasets；如果路径不存在，则会在分词后将 tokenized datasets 保存在此路径中。
     - None



模型参数
---------------------------------

基本参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: ModelArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - model_name_or_path
     - Optional[str]
     - 模型路径（本地路径或 Huggingface/ModelScope 路径）。
     - None
   * - adapter_name_or_path
     - Optional[str]
     - 适配器路径（本地路径或 Huggingface/ModelScope 路径）。使用逗号分隔多个适配器路径。
     - None
   * - adapter_folder
     - Optional[str]
     - 包含适配器权重的文件夹路径。
     - None
   * - cache_dir
     - Optional[str]
     - 保存从 Hugging Face 或 ModelScope 下载的模型的本地路径。
     - None
   * - use_fast_tokenizer
     - bool
     - 是否使用 fast_tokenizer 。
     - True
   * - resize_vocab
     - bool
     - 是否调整词表和嵌入层的大小。
     - False
   * - split_special_tokens
     - bool
     - 是否在分词时将 special token 分割。
     - False
   * - new_special_tokens
     - Optional[str]
     - 要添加到 tokenizer 中的 special token。多个 special token 用逗号分隔。
     - None
   * - model_revision
     - str
     - 所使用的特定模型版本。
     - main
   * - low_cpu_mem_usage
     - bool
     - 是否使用节省内存的模型加载方式。
     - True
   * - rope_scaling
     - Optional[Literal["linear", "dynamic", "yarn", "llama3"]]
     - RoPE Embedding 的缩放策略，支持 linear、dynamic、yarn 或 llama3。
     - None
   * - flash_attn
     - Literal["auto", "disabled", "sdpa", "fa2"]
     - 是否启用 FlashAttention 来加速训练和推理。可选值为 auto, disabled, sdpa, fa2。
     - auto
   * - shift_attn
     - bool
     - 是否启用 Shift Short Attention (S^2-Attn)。
     - False
   * - mixture_of_depths
     - Optional[Literal["convert", "load"]]
     - 需要将模型转换为 mixture_of_depths（MoD）模型时指定： convert 需要加载 mixture_of_depths（MoD）模型时指定： load。
     - None
   * - use_unsloth
     - bool
     - 是否使用 unsloth 优化 LoRA 微调。
     - False
   * - use_unsloth_gc
     - bool
     - 是否使用 unsloth 的梯度检查点。
     - False
   * - enable_liger_kernel
     - bool
     - 是否启用 liger 内核以加速训练。
     - False
   * - moe_aux_loss_coef
     - Optional[float]
     - MoE 架构中 aux_loss 系数。数值越大，各个专家负载越均衡。
     - None
   * - disable_gradient_checkpointing
     - bool
     - 是否禁用梯度检查点。
     - False
   * - use_reentrant_gc
     - bool
     - 是否启用可重入梯度检查点
     - True
   * - upcast_layernorm
     - bool
     - 是否将 layernorm 层权重精度提高至 fp32。
     - False
   * - upcast_lmhead_output
     - bool
     - 是否将 lm_head 输出精度提高至 fp32。
     - False
   * - train_from_scratch
     - bool
     - 是否随机初始化模型权重。
     - False
   * - infer_backend
     - Literal["huggingface", "vllm"]
     - 推理时使用的后端引擎，支持 huggingface 或 vllm。
     - huggingface
   * - offload_folder
     - str
     - 卸载模型权重的路径。
     - offload
   * - use_cache
     - bool
     - 是否在生成时使用 KV 缓存。
     - True
   * - infer_dtype
     - Literal["auto", "float16", "bfloat16", "float32"]
     - 推理时使用的模型权重和激活值的数据类型。支持 auto, float16, bfloat16, float32。
     - auto
   * - hf_hub_token
     - Optional[str]
     - 用于登录 HuggingFace 的验证 token。
     - None
   * - ms_hub_token
     - Optional[str]
     - 用于登录 ModelScope Hub 的验证 token。
     - None
   * - om_hub_token
     - Optional[str]
     - 用于登录 Modelers Hub 的验证 token。
     - None
   * - print_param_status
     - bool
     - 是否打印模型参数的状态。
     - False
   * - trust_remote_code
     - bool
     - 是否信任来自 Hub 上数据集/模型的代码执行。
     - False
   * - compute_dtype
     - Optional[torch.dtype]
     - 用于计算模型输出的数据类型，无需手动指定。
     - None
   * - device_map
     - Optional[Union[str, Dict[str, Any]]]
     - 模型分配的设备映射，无需手动指定。
     - None
   * - model_max_length
     - Optional[int]
     - 模型的最大输入长度，无需手动指定。
     - None
   * - block_diag_attn
     - bool
     - 是否使用块对角注意力，无需手动指定。
     - False


多模态模型
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: ProcessorArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - image_max_pixels
     - int
     - 图像输入的最大像素数。
     - 768 x 768
   * - image_min_pixels
     - int
     - 图像输入的最小像素数。
     - 32 x 32
   * - video_max_pixels
     - int
     - 视频输入的最大像素数。
     - 256 x 256
   * - video_min_pixels
     - int
     - 视频输入的最小像素数。
     - 16 x 16
   * - video_fps
     - float
     - 视频输入的采样帧率（每秒采样帧数）。
     - 2.0
   * - video_maxlen
     - int
     - 视频输入的最大采样帧数。
     - 128




vllm 推理
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: VllmArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - vllm_maxlen
     - int
     - 最大序列长度（包括输入文本和生成文本）。
     - 4096
   * - vllm_gpu_util
     - float
     - GPU使用比例，范围在(0, 1)之间。
     - 0.9
   * - vllm_enforce_eager
     - bool
     - 是否禁用 vLLM 中的 CUDA graph。
     - False
   * - vllm_max_lora_rank
     - int
     - 推理所允许的最大的 LoRA Rank。
     - 32
   * - vllm_config
     - Optional[Union[dict, str]]
     - vLLM引擎初始化配置。以字典或JSON字符串输入。
     - None



模型量化
~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: QuantizationArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - quantization_method
     - Literal["bitsandbytes", "hqq", "eetq"]
     - 指定用于量化的算法，支持 "bitsandbytes", "hqq" 和 "eetq"。
     - bitsandbytes
   * - quantization_bit
     - Optional[int]
     - 指定在量化过程中使用的位数，通常是4位、8位等。
     - None
   * - quantization_type
     - Literal["fp4", "nf4"]
     - 量化时使用的数据类型，支持 "fp4" 和 "nf4"。
     - nf4
   * - double_quantization
     - bool
     - 是否在量化过程中使用 double quantization，通常用于 "bitsandbytes" int4 量化训练。
     - True
   * - quantization_device_map
     - Optional[Literal["auto"]]
     - 用于推理 4-bit 量化模型的设备映射。需要 "bitsandbytes >= 0.43.0"。
     - None


模型导出
~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: ExportArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - export_dir
     - Optional[str]
     - 导出模型保存目录的路径。
     - None
   * - export_size
     - int
     - 导出模型的文件分片大小（以GB为单位）。
     - 5
   * - export_device
     - Literal["cpu", "auto"]
     - 导出模型时使用的设备，auto 可自动加速导出。
     - cpu
   * - export_quantization_bit
     - Optional[int]
     - 量化导出模型时使用的位数。
     - None
   * - export_quantization_dataset
     - Optional[str]
     - 用于量化导出模型的数据集路径或数据集名称。
     - None
   * - export_quantization_nsamples
     - int
     - 量化时使用的样本数量。
     - 128
   * - export_quantization_maxlen
     - int
     - 用于量化的模型输入的最大长度。
     - 1024
   * - export_legacy_format
     - bool
     - True： .bin 格式保存。 False： .safetensors 格式保存。
     - False
   * - export_hub_model_id
     - Optional[str]
     - 模型上传至 Huggingface 的仓库名称。
     - None




评估参数
------------------------
.. list-table:: EvalArguments
   :widths: 10 10 40 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - task
     - str
     - 评估任务的名称，可选项有 mmlu_test, ceval_validation, cmmlu_test
     - None
   * - task_dir
     - str
     - 包含评估数据集的文件夹路径。
     - evaluation
   * - batch_size
     - int
     - 每个GPU使用的批量大小。
     - 4
   * - seed
     - int
     - 用于数据加载器的随机种子。
     - 42
   * - lang
     - str
     - 评估使用的语言，可选值为 en、zh。
     - en
   * - n_shot
     - int
     - few-shot 的示例数量。
     - 5
   * - save_dir
     - str
     - 保存评估结果的路径。 如果该路径已经存在则会抛出错误。
     - None
   * - download_mode
     - str
     - 评估数据集的下载模式，如果数据集已经存在则重复使用，否则则下载。
     - DownloadMode.REUSE_DATASET_IF_EXISTS



生成参数
------------------------
.. list-table:: GeneratingArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - do_sample
     - bool
     - 是否使用采样策略生成文本。如果设置为 False，将使用 greedy decoding。
     - True
   * - temperature
     - float
     - 用于调整生成文本的随机性。temperature 越高，生成的文本越随机；temperature 越低，生成的文本越确定。
     - 0.95
   * - top_p
     - float
     - 用于控制生成时候选 token 集合大小的参数。例如：top_p = 0.7 意味着模型会先选择概率最高的若干个 token 直到其累积概率之和大于 0.7，然后在这些 token 组成的集合中进行采样。
     - 0.7
   * - top_k
     - int
     - 用于控制生成时候选 token 集合大小的参数。例如：top_k = 50 意味着模型会在概率最高的50个 token 组成的集合中进行采样。
     - 50
   * - num_beams
     - int
     - 用于 beam_search 的束宽度。值为 1 表示不使用 beam_search。
     - 1
   * - max_length
     - int
     - 文本最大长度（包括输入文本和生成文本的长度）。
     - 1024
   * - max_new_tokens
     - int
     - 生成文本的最大长度。设置 max_new_tokens 会覆盖 max_length。
     - 1024
   * - repetition_penalty
     - float
     - 对生成重复 token 的惩罚系数。对于已经生成过的 token 生成概率乘以 1/repetition_penalty。值小于 1.0 会提高重复 token 的生成概率，大于 1.0 则会降低重复 token 的生成概率。
     - 1.0
   * - length_penalty
     - float
     - 在使用 beam_search 时对生成文本长度的惩罚系数。length_penalty > 0 鼓励模型生成更长的序列，length_penalty < 0 会鼓励模型生成更短的序列。
     - 1.0
   * - default_system
     - str
     - 默认的 system_message，例如: "You are a helpful assistant."
     - None
   * - skip_special_tokens
     - bool
     - 解码时是否忽略特殊 token。
     - True


SwanLab 参数
-------------------------------------------
.. list-table:: SwanLabArguments
   :widths: 20 10 60 15
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - use_swanlab
     - bool
     - 是否使用 SwanLab。
     - False
   * - swanlab_project
     - str
     - SwanLab 中的项目名称。
     - "llamafactory"
   * - swanlab_workspace
     - str
     - SwanLab 中的工作区名称。
     - None
   * - swanlab_run_name
     - str
     - SwanLab 中的实验名称。
     - None
   * - swanlab_mode
     - Literal["cloud", "local"]
     - SwanLab 的运行模式。
     - cloud
   * - swanlab_api_key
     - str
     - SwanLab 的 API 密钥。
     - None


训练参数
-------------------------------------------

RAY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table:: RayArguments
   :widths: 20 20 60 20
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
     - 默认值
   * - ray_run_name
     - Optional[str]
     - 训练结果将保存在 <ray_storage_path>/ray_run_name 路径下。
     - None
   * - ray_storage_path
     - str
     - 保存训练结果的存储路径。
     - ./saves
   * - ray_num_workers
     - int
     - Ray 训练所使用的工作进程数量。
     - 1
   * - resources_per_worker
     - Union[dict, str]
     - 每个工作进程分配的资源。默认使用 1 GPU。
     - {"GPU": 1}
   * - placement_strategy
     - Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"]
     - Ray 训练的资源调度策略。可选值包括 SPREAD、PACK、STRICT_SPREAD 和 STRICT_PACK。
     - PACK


环境变量
--------------------------------------

.. list-table:: Environment Variables
   :widths: 30 20 50
   :header-rows: 1

   * - 名称
     - 类型
     - 介绍
   * - ``API_HOST``
     - API
     - API 服务器监听的主机地址
   * - ``API_PORT``
     - API
     - API 服务器监听的端口号
   * - ``API_KEY``
     - API
     - 访问 API 的密码。
   * - ``API_MODEL_NAME``
     - API
     - 指定 API 服务要加载和使用的模型名称
   * - ``API_VERBOSE``
     - API
     - 控制 API 日志的详细程度
   * - ``FASTAPI_ROOT_PATH``
     - API
     - 设置 FastAPI 应用的根路径
   * - ``MAX_CONCURRENT``
     - API
     - API 的最大并发请求数。
   * - ``DISABLE_VERSION_CHECK``
     - General
     - 是否禁用启动时的版本检查。
   * - ``FORCE_CHECK_IMPORTS``
     - General
     - 强制检查可选的导入
   * - ``ALLOW_EXTRA_ARGS``
     - General
     - 允许在命令行中传递额外参数
   * - ``LLAMAFACTORY_VERBOSITY``
     - General
     - 设置 LLaMA-Factory 的日志级别("DEBUG","INFO","WARN")
   * - ``USE_MODELSCOPE_HUB``
     - General
     - 优先使用 ModelScope 下载模型/数据集或使用缓存路径中的模型/数据集
   * - ``USE_OPENMIND_HUB``
     - General
     - 优先使用 Openmind 下载模型/数据集或使用缓存路径中的模型/数据集
   * - ``USE_RAY``
     - General
     - 是否使用 Ray 进行分布式执行或任务管理。
   * - ``RECORD_VRAM``
     - General
     - 是否记录 VRAM 使用情况。
   * - ``OPTIM_TORCH``
     - General
     - 是否表示启用特定的 PyTorch 优化。
   * - ``NPU_JIT_COMPILE``
     - General
     - 是否为 NPU启用 JIT 编译。
   * - ``CUDA_VISIBLE_DEVICES``
     - General
     - GPU 选择。
   * - ``ASCEND_RT_VISIBLE_DEVICES``
     - General
     - NPU 选择。 
   * - ``FORCE_TORCHRUN``
     - Torchrun
     - 是否强制使用 ``torchrun`` 启动脚本
   * - ``MASTER_ADDR``
     - Torchrun
     - Torchrun部署中主节点 (master node) 的网络地址
   * - ``MASTER_PORT``
     - Torchrun
     - Torchrun部署中主节点用于通信的端口号
   * - ``NNODES``
     - Torchrun
     - 参与分布式部署的总节点数量
   * - ``NODE_RANK``
     - Torchrun
     - 当前节点在所有节点中的 rank，通常从 0 到 ``NNODES-1``。
   * - ``NPROC_PER_NODE``
     - Torchrun
     - 每个节点上的 GPU 数
   * - ``WANDB_DISABLED``
     - Log
     - 是否禁用 wandb
   * - ``WANDB_PROJECT``
     - Log
     - 设置 wandb 中的项目名称。
   * - ``WANDB_API_KEY``
     - Log
     - 访问 wandb 的 api key
   * - ``GRADIO_SHARE``
     - Web UI
     - 是否创建一个可共享的 webui 链接
   * - ``GRADIO_SERVER_NAME``
     - Web UI
     - 设置 Gradio 服务器 IP 地址（例如 ``0.0.0.0``）
   * - ``GRADIO_SERVER_PORT``
     - Web UI
     - 设置 Gradio 服务器的端口
   * - ``GRADIO_ROOT_PATH``
     - Web UI
     - 设置 Gradio 应用的根路径
   * - ``GRADIO_IPV6``
     - Web UI
     - 启用 Gradio 服务器的 IPv6 支持
   * - ``ENABLE_SHORT_CONSOLE``
     - Setting
     - 支持使用 ``lmf`` 表示 ``llamafactory-cli``