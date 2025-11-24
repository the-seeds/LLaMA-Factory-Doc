NPU 推理
=============

环境安装
--------

版本需求
~~~~~~~~

- 操作系统：Linux
- Python：>= 3.10, < 3.12
- gcc：>= 9

硬件环境
~~~~~~~~

使用如下命令查看显卡固件和驱动版本。

.. code-block:: shell

   npu-smi info

输出显卡信息则驱动安装正常。

更多细节参考 `快速安装昇腾环境 <https://ascend.github.io/docs/sources/ascend/quick_install.html>`_ 。

软件环境
~~~~~~~~

- CANN >= 8.1.RC1，包括 ``toolkit``、``kernels``、``nnal``。

使用下述命令安装。

.. code-block:: shell

   # Install required python packages.
   pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs 'numpy<2.0.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

   # Download and install the CANN package.
   wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run
   chmod +x ./Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run
   ./Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run --full

   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run
   chmod +x ./Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run
   ./Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run --install

   wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run
   chmod +x ./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run
   ./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run --install

   source /usr/local/Ascend/nnal/atb/set_env.sh

vLLM-Ascend安装
~~~~~~~~~~~~~~~

使用下述命令安装 ``vLLM-Ascend`` 。

.. code-block:: shell

   # Install vllm-project/vllm from pypi
   pip install vllm==0.8.5.post1

   # Install vllm-project/vllm-ascend from pypi.
   pip install vllm-ascend==0.8.5rc1

LLaMA-Factory安装
~~~~~~~~~~~~~~~~~

使用下述命令安装 ``LLaMA-Factory`` 。

.. code-block:: shell

   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]" --no-build-isolation

推理测试
--------

可视化界面
~~~~~~~~~~

使用下述命令启动LLaMA-Factory的可视化界面。

.. code-block:: shell

   llamafactory-cli webui

浏览器访问到如下界面则项目启动成功。

.. image:: ../assets/advanced/npu-inference-webui.png
   :alt: webui

选择模型并切换到chat模式并将推理引擎修改为vLLM，然后点击加载模型。

.. image:: ../assets/advanced/npu-inference-load.png
   :alt: load_model

加载完成后可以进行对话。

.. image:: ../assets/advanced/npu-inference-chat.png
   :alt: chat

性能对比
~~~~~~~~

硬件：``Ascend 910B1 ✖ 2``

+----------------+----------------+----------------+-------------+
|     模型名称   |      vLLM      |   Hugging Face |  速度提升比 |
+================+================+================+=============+
|  qwen2.5-0.5B  | 22.7 tokens/s  | 10.9 tokens/s  |    108.3%   |
+----------------+----------------+----------------+-------------+
|  qwen2.5-7B    | 20.2 tokens/s  |  9.9 tokens/s  |    104.0%   |
+----------------+----------------+----------------+-------------+

在推理性能上。vLLM框架比huggingface的推理速度提升了超过一倍。