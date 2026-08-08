
额外选项
========================

LLaMA Pro
----------------

为了解决大语言模型的遗忘问题， LLaMA Pro 通过在原有模型上增加新模块以适应新的任务，使其在多个新任务上的表现均优于原始模型。
LlamaFactory 支持 LLaMA Pro 的使用。
您可以使用运行 ``expand.sh`` 将 ``Meta-Llama-3-8B-Instruct`` 扩展为 ``llama3-8b-instruct-pro``。

对于 LLaMA Pro 模型进行训练时，您需要指定 ``use_llama_pro`` 为 ``true``。