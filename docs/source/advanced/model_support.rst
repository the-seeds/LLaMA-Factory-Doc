
模型支持
=====================


LLaMA-Factory 允许用户添加自定义模型支持。我们将以 LLaMA-4 多模态模型为例，详细介绍如何为新模型添加支持。对于多模态模型，我们需要完成两个主要任务：

1. 注册模型的template
2. 解析多模态数据并构建 messages

.. https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/tokenizer_config.json#L9077

注册 template
---------------------

首先，我们可以通过以下方法获取 LLaMA-4 模型的 template：

.. code-block:: python 

    from transformers import AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct")
    messages = [
        {"role": "user", "content": r"{{content}}"},
        {"role": "assistant", "content": r"{{content}}"},
        {"role": "system", "content": r"{{content}}"},
        {"role": "tool", "content": r"{{content}}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    print("========== Template ==========")
    print(text)

输出如下。通过观察输出我们可以得到模型的 chat_template。除此以外也可以通过 `huggingface repo <https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/tokenizer_config.json#L9077>`_  来获取模型的 template.

.. raw:: html

    <pre><code class="python" style="font-family: monospace; font-size: 14px; background-color: #fefefe; color: #000000; padding: 5px; border-radius: 0px;">========== Template ==========
    &lt;|begin_of_text|&gt;<span style='background-color: #fff9b1;'>&lt;|header_start|&gt;user&lt;|header_end|&gt;

    {{content}}&lt;|eot|&gt;</span><span style='background-color: #ffff66;'>&lt;|header_start|&gt;assistant&lt;|header_end|&gt;

    </span><span style='background-color: #ffff66;'>{{content}}&lt;|eot|&gt;</span><span style="background-color: #d6f0ff;">&lt;|header_start|&gt;system&lt;|header_end|&gt;

    {{content}}&lt;|eot|&gt;</span><span style="background-color: #f1e6ff;">&lt;|header_start|&gt;ipython&lt;|header_end|&gt;

    "{{content}}"&lt;|eot|&gt;</span><span style='background-color: #ffff66;'>&lt;|header_start|&gt;assistant&lt;|header_end|&gt;</span>
    </code></pre>

通过观察输出，我们可以得知 LLaMA-4 的 chat_template 主要由以下几部分组成：

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 消息类型
     - 模板格式
   * - 用户消息
     - ``<|header_start|>user<|header_end|>\n\n{{content}}<|eot|>``
   * - 助手消息
     - ``<|header_start|>assistant<|header_end|>\n\n{{content}}<|eot|>``
   * - 系统消息
     - ``<|header_start|>system<|header_end|>\n\n{{content}}<|eot|>``
   * - 工具消息
     - ``<|header_start|>ipython<|header_end|>\n\n"{{content}}"<|eot|>``

我们可以在 ``src/llamafactory/data/template.py`` 中使用 ``register_template`` 方法为自定义模型注册 chat_template。

在实际应用中，我们往往会在用户输入的信息后添加助手回复模板的头部 ``<|header_start|>assistant<|header_end|>`` 来引导模型进行回复。

因此我们可以看到，用户消息和工具输出的模板中都附有了助手回复的头部，而助手消息格式 ``format_assitant`` 也因此省略了助手回复的头部，只保留其内容部分 ``{{content}}<|eot|。>``。

我们可以根据上面的输出完成 ``name``, ``format_user``, ``format_assistant``, ``format_system`` 与 ``format_observation`` 字段的填写。

``format_prefix`` 字段用于指定模型的开头部分，通常可以在 `tokenizer_config.json <https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/tokenizer_config.json#L9076>`_ 中找到。

``stop_words`` 字段用于指定模型的停止词，可以在 `generation_config.json <https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/generation_config.json>`_ 中找到 eos_token_id，再把 eos_token_id 对应的 token 填入。

对于多模态模型，我们还需要在 ``mm_plugin`` 字段中指定多模态插件。

.. raw:: html

    <pre><code class="python" style="font-family: monospace; font-size: 14px; background-color: #fefefe; color: #000000; padding: 5px; border-radius: 0px;">register_template(
        # Template
        name="llama4", 
        # User Message Format, with a generation prompt template at the end
        format_user=StringFormatter(
            slots=["<span style='background-color: #fff9b1;'>&lt;|header_start|&gt;user&lt;|header_end|&gt;\n\n{{content}}&lt;|eot|&gt;</span><span style='background-color: #ffff66;'>&lt;|header_start|&gt;assistant&lt;|header_end|&gt;\n\n</span>"]
        ),
        # Assistant Message format
        <span style='background-color: #ffff66;'>format_assistant</span>=StringFormatter(slots=["<span style='background-color: #ffff66;'>{{content}}&lt;|eot|&gt;</span>"]),
        # System Message Format
        format_system=StringFormatter(slots=["<span style="background-color: #d6f0ff;">&lt;|header_start|&gt;system&lt;|header_end|&gt;\n\n{{content}}&lt;|eot|&gt;</span>"]),
        # Function Call Format
        format_function=FunctionFormatter(slots=["{{content}}&lt;|eot|&gt;"], tool_format="llama3"),
        # Tool Output Format, with a generation prompt template at the end
        format_observation=StringFormatter(
            slots=[
                "<span style='background-color: #f1e6ff;'>&lt;|header_start|&gt;ipython&lt;|header_end|&gt;\n\n{{content}}&lt;|eot|&gt;</span><span style='background-color: #ffff66;'>&lt;|header_start|&gt;assistant&lt;|header_end|&gt;</span>\n\n"
            ]
        ),
        # Tool Call Format
        format_tools=ToolFormatter(tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["&lt;|eot|&gt;", "&lt;|eom|&gt;"],
        mm_plugin=get_mm_plugin(name="llama4", image_token="&lt;|image|&gt;"),
    )
    </code></pre>

多模态数据构建
--------------------

对于多模态模型，我们参照原始模型在 LLaMA-Factory 中实现多模态数据的解析。

我们可以在 ``src/llamafactory/data/mm_plugin.py`` 中实现 ``Llama4Plugin`` 类来解析多模态数据。

``Llama4Plugin`` 类继承自 ``BasePlugin`` 类，并实现了 ``get_mm_inputs`` 和 ``process_messages`` 方法来解析多模态数据。

.. note::

    .. code-block:: python

        @dataclass
        class Llama4Plugin(BasePlugin):
            @override
            def process_messages(
                ...
            @override
            def get_mm_inputs(
                ...

``get_mm_inputs`` 的作用是将图像、视频等多模态数据转化为模型可以接收的输入，如 ``pixel_values``。为实现 ``get_mm_inputs``，首先我们需要检查 llama4 的 processor 是否可以与 `已有实现 <https://github.com/hiyouga/LLaMA-Factory/blob/da971c37640de20f97b4d774e77e6f8d5c00b40a/src/llamafactory/data/mm_plugin.py#L264>`_ 兼容。模型官方仓库中的 `processing_llama4.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/processing_llama4.py#L157>`_ 表明 llama4 的 processor 返回数据包含字段 ``pixel_values``，这与 LLaMA-Factory 中的已有实现兼容。因此，我们只需要参照已有的 ``get_mm_inputs`` 方法实现即可。

.. note::

    .. code-block:: python
        
        # https://github.com/hiyouga/LLaMA-Factory/blob/da971c37640de20f97b4d774e77e6f8d5c00b40a/src/llamafactory/data/mm_plugin.py#L264
        def _get_mm_inputs(
            self,
            images: list["ImageInput"],
            videos: list["VideoInput"],
            audios: list["AudioInput"],
            processor: "MMProcessor",
            imglens: Optional[list[int]] = None,
        ) -> dict[str, "torch.Tensor"]:
            r"""Process visual inputs.

            Returns: (llava and paligemma)
                pixel_values: tensor with shape (B, C, H, W)

            Returns: (qwen2-vl)
                pixel_values: tensor with shape (num_patches, patch_dim)
                image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                where num_patches == torch.prod(image_grid_thw)

            Returns: (mllama)
                pixel_values: tensor with shape
                            (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                            For example, (2, 1, 4, 3, 560, 560).
                aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
                aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
                num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

    ..     """



``process_messages`` 的作用是根据输入图片/视频的大小，数量等信息在 messages 中插入相应数量的占位符，以便模型可以正确解析多模态数据。 我们需要参考 `原仓库实现 <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/processing_llama4.py#L157>`_ 以及 LLaMA-Factory 中的规范返回 ``list[dict[str, str]]`` 类型的 messages。


.. 测试 TODO
.. ----------------------


提供模型路径
---------------------

最后, 在 `src/llamafactory/extras/constants.py <https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/extras/constants.py>`_ 中提供模型的下载路径。例如：

.. code-block:: python 

    register_model_group(
    models={
        "Llama-4-Scout-17B-16E": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Scout-17B-16E",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Scout-17B-16E",
        },
        "Llama-4-Scout-17B-16E-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Scout-17B-16E-Instruct",
        },
        "Llama-4-Maverick-17B-128E": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Maverick-17B-128E",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Maverick-17B-128E",
        },
        "Llama-4-Maverick-17B-128E-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Maverick-17B-128E-Instruct",
        },
    },
    template="llama4",
    multimodal=True,
    )