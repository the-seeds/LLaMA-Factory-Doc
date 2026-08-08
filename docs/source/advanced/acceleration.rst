加速
=====================

LlamaFactory 支持多种加速技术，包括：:ref:`FlashAttention <flashattn>` 、 :ref:`Unsloth <sloth>` 、 :ref:`Liger Kernel <ligerkernel>`  。




.. _flashattn:


FlashAttention
----------------------------

`FlashAttention  <https://github.com/Dao-AILab/flash-attention/>`_ 能够加快注意力机制的运算速度，同时减少对内存的使用。

如果您想使用 FlashAttention,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml 

    flash_attn: fa2



.. _sloth:

Unsloth
---------------------------

`Unsloth <https://github.com/unslothai/unsloth/>`_ 框架支持 Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen等大语言模型并且支持 4-bit 和 16-bit 的 QLoRA/LoRA 微调，该框架在提高运算速度的同时还减少了显存占用。

如果您想使用 Unsloth, 请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml

    use_unsloth: True 


.. _ligerkernel:

Liger Kernel
---------------------------------------
`Liger Kernel <https://github.com/linkedin/Liger-Kernel/>`_  是一个大语言模型训练的性能优化框架, 可有效地提高吞吐量并减少内存占用。

如果您想使用 Liger Kernel,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml

    enable_liger_kernel: True 