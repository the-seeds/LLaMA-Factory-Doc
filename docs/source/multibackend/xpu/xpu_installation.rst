XPU安装及配置
=================

本文档介绍如何在 Intel GPU（PyTorch 中的 ``xpu`` 设备）上安装 LlamaFactory。

硬件和操作系统
------------------------

LlamaFactory 通过 PyTorch 原生的 XPU 后端支持 Intel GPU，无需安装 Intel Extension for PyTorch (IPEX)。

**表 1**  已验证环境

.. list-table::
   :align: left
   :widths: 30 50
   :header-rows: 1

   * - 项目
     - 版本
   * - GPU
     - Intel Arc Pro B60
   * - 操作系统
     - Ubuntu 24.04（Docker 镜像）、Ubuntu 25.10（物理机）
   * - PyTorch
     - 2.13.0+xpu、2.14.0+xpu
   * - Python
     - 3.12

其他 Intel GPU 是否支持，取决于 PyTorch XPU 后端，请参考 `PyTorch XPU 入门文档 <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`__。

可以选择以下两种方式之一进行环境配置：

- :ref:`xpu_install_form_pip`
- :ref:`xpu_install_form_docker`


核心依赖说明
----------------

所有安装方式均依赖以下组件：

- **GPU 驱动**：内核驱动及 Level Zero 计算运行时
- **PyTorch XPU 版本**：从 PyTorch 官方 XPU 索引安装的 ``torch``、``torchvision``、``torchaudio``

根据安装方式不同，所需操作有所区别：

- **手动安装**：需在宿主机上安装驱动、计算运行时和 PyTorch XPU 版本。
- **Docker 本地构建**：宿主机仅需安装内核驱动，计算运行时和 PyTorch 已集成在镜像中。


.. _xpu_install_form_pip:

方式一：手动安装环境
----------------------

1. 驱动
~~~~~~~~~~~~~~~~~~~~

请参考 `Intel GPU 安装指南 <https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html>`__ 安装驱动和计算运行时。安装完成后，需要将当前用户加入 ``render`` 和 ``video`` 用户组，并重新登录：

.. code-block:: bash

    sudo usermod -aG render,video $USER

.. note::
  请在 BIOS 中开启 **Resizable BAR**，否则可能出现 ``Bus error (core dumped)`` 或性能下降。

2. PyTorch XPU 版本
~~~~~~~~~~~~~~~~~~~~

LlamaFactory 在 ``requirements/xpu.txt`` 中固定了配套的 ``torch``、``torchvision``、``torchaudio`` 版本。请先安装 PyTorch，再安装 LlamaFactory：

.. code-block:: bash

    git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
    cd LlamaFactory
    pip install -r requirements/xpu.txt --index-url https://download.pytorch.org/whl/xpu

.. note::
  请使用 ``--index-url`` 而不是 ``--extra-index-url``，否则 pip 可能会从 PyPI 安装非 XPU 版本的 PyTorch。三个包需要同时安装，版本不配套时会出现 ``RuntimeError: operator torchvision::nms does not exist``。

3. LlamaFactory
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install -e .
    pip install -r requirements/metrics.txt

其余步骤与 :doc:`安装 <../../getting_started/installation>` 一致。

4. 验证安装
~~~~~~~~~~~~~~~~

执行以下 Python 脚本：

.. code-block:: python

    import torch
    print(torch.xpu.is_available(), torch.xpu.device_count())

预期输出 ``True`` 及可用的 GPU 数量。如果输出 ``0``，请参考 :ref:`xpu_faq`。

5. 可选依赖
~~~~~~~~~~~~~~~~

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - 依赖
     - 安装说明
   * - DeepSpeed
     - 需要 ``deepspeed>=0.18.7``：``DS_BUILD_OPS=0 pip install "deepspeed>=0.18.7"``。更早的版本在 XPU 上会因 ``ImportError: cannot import name 'DpcppBuildExtension'`` 失败，``requirements/deepspeed.txt`` 中的版本范围目前也包含这些版本。
   * - bitsandbytes
     - ``pip install -r requirements/bitsandbytes.txt``
   * - GPTQ / AWQ
     - ``pip install "gptqmodel==7.2.0" optimum``。更新的 gptqmodel 版本需要高于 LlamaFactory 所支持的 transformers 版本。加载 AWQ 模型还需执行 ``pip install --no-deps autoawq``。
   * - Liger Kernel
     - ``pip install --no-deps liger-kernel``。请务必加上 ``--no-deps``，详见 :ref:`xpu_faq`。
   * - torch.compile
     - 需要 Intel 离线编译器 ``ocloc``，请安装与计算运行时版本一致的 ``intel-ocloc``。


.. _xpu_install_form_docker:

方式二：Docker 本地构建
-----------------------------

.. note::
  请确保宿主机已安装内核驱动。目前 XPU 镜像需要在本地构建，Docker Hub 上暂无预构建镜像。

镜像基于 Intel Deep Learning Essentials（DLE），其中已包含计算运行时和 oneAPI。镜像中的计算运行时版本需要与宿主机的内核驱动兼容，可以通过 ``dpkg -l libze-intel-gpu1`` 查看宿主机版本。

1. 使用 Docker Build 构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在项目根目录下执行以下命令：

.. code-block:: shell

    docker build -f ./docker/docker-xpu/Dockerfile -t llamafactory:xpu .

可用的 Dockerfile 构建参数如下：

.. list-table::
   :align: left
   :widths: 20 40 40
   :header-rows: 1

   * - 参数
     - 默认值
     - 用途
   * - ``BASE_IMAGE``
     - ``intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04``
     - 决定镜像中的计算运行时和 oneAPI 版本
   * - ``PIP_INDEX``
     - ``https://pypi.org/simple``
     - 指定 Python 软件包索引（PyTorch 除外）
   * - ``PYTORCH_INDEX``
     - ``https://download.pytorch.org/whl/xpu``
     - 指定 PyTorch XPU 版本的索引
   * - ``OCLOC_VERSION``
     - ``26.18.38308.1``
     - 固定 ``intel-ocloc`` 版本，需与 ``BASE_IMAGE`` 的计算运行时版本一致

镜像中已安装 DeepSpeed 0.19.6 和 bitsandbytes，并设置了 ``OPTIM_TORCH=0``，详见 :doc:`XPU训练 <xpu_training>`。

启动容器：

.. code-block:: bash

    docker run -it --rm \
        --device /dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        --group-add $(getent group render | cut -d: -f3) \
        --group-add $(getent group video | cut -d: -f3) \
        --ipc=host \
        -p 7860:7860 \
        -p 8000:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --name llamafactory \
        llamafactory:xpu bash

.. note::
  多卡训练需要挂载 ``/dev/dri/by-path``，oneCCL 依赖其中的设备路径进行卡间通信。

2. 使用 Docker Compose 构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

  cd docker/docker-xpu
  docker compose up -d
  docker compose exec llamafactory bash

.. note::
  ``docker-compose.yml`` 中的 ``group_add`` 使用 ``render`` (992) 和 ``video`` (44) 的数字 GID。如果宿主机不同，请先执行 ``getent group render video`` 查看并修改。

进入容器后，可以执行以下命令确认 GPU 可用：

.. code-block:: shell

  python3 -c "import torch; print(torch.xpu.device_count(), 'XPU device(s) found')"
