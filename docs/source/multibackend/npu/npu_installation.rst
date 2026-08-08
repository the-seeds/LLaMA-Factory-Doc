NPU安装及配置
=================

本文档介绍 LlamaFactory 在华为昇腾 NPU 上的环境准备方式。当前主要面向 Atlas A2/A3 训练系列设备；开始安装前，请先确认硬件型号和操作系统兼容性，再根据部署方式选择后续步骤。

硬件配套和支持的操作系统
------------------------

**表 1**  产品硬件支持列表

.. list-table::
   :class: npu-hardware-support-table
   :align: left
   :widths: 40 20
   :header-rows: 1

   * - 产品
     - 是否支持
   * - Ascend 950 系列产品
     - √
   * - Atlas A3 训练系列产品
     - √
   * - Atlas A3 推理系列产品
     - x
   * - Atlas A2 训练系列产品
     - √
   * - Atlas A2 推理系列产品
     - x
   * - Atlas 200I/500 A2 推理产品
     - x
   * - Atlas 推理系列产品
     - x
   * - Atlas 训练系列产品
     - x

.. note::

   本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考 `兼容性查询助手 <https://www.hiascend.com/hardware/compatibility>`_。
- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“`操作系统兼容性说明 <https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum>`__”章节（商用版）或“`操作系统兼容性说明 <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum>`__”章节（社区版）。

确认硬件和操作系统满足上述要求后，可以选择以下三种方式之一进行环境配置及使用：

- :ref:`install_form_pip`
- :ref:`use_form_docker`
- :ref:`install_form_docker`


核心依赖说明
----------------

所有安装方式均依赖以下组件：

- **HDK**：固件及驱动
- **CANN**：异构计算架构
- **TorchNPU**：PyTorch 的昇腾适配插件

根据安装方式不同，所需操作有所区别：

- **手动安装**：需手动安装 HDK、CANN 和 TorchNPU。
- **Docker 镜像/构建**：宿主机仅需安装 HDK (驱动/固件)，CANN 和 TorchNPU 已集成在镜像中。


.. _install_form_pip:

方式一：手动安装环境
----------------------

本方式需要您手动安装 HDK、CANN 和 TorchNPU。


1. 版本及下载链接
~~~~~~~~~~~~~~~~~~~~

本文档列举了最新的依赖版本及下载链接，请根据设备型号选择：

.. list-table::
   :align: left
   :widths: 5 10 50
   :header-rows: 1

   * - 设备
     - 依赖
     - 链接
   * - A3
     - HDK
     - https://www.hiascend.com/hardware/firmware-drivers?ids=d803%2C89dda9ba9de741349efa03687a487678%2C16%2CAArch64%2Conline_Yum
   * -
     - CANN
     - https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0
   * -
     - TorchNPU
     - 2.10.0.post2
   * - A2
     - HDK
     - https://www.hiascend.com/hardware/firmware-drivers?ids=d802%2C26958bcc909e4cd48fa56d4c4a43ebec%2C17%2CAArch64%2Conline_Yum
   * -
     - CANN
     - https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0
   * -
     - TorchNPU
     - 2.10.0.post2

2. 驱动及固件
~~~~~~~~~~~~~~~~~~~~

请根据实际情况选择 ``.run`` 或 ``.deb`` 的 HDK 安装包，并请注意安装包对 ``aarch64`` 和 ``x86`` 做了区分。

以下以 A2 系列为例。A3 系列包 ``firmware`` 和 ``driver`` 包名有所变化，可以根据链接内实际情况选择。

A3 内部包名类似于 ``Atlas-A3-hdk-npu-driver_25.0.rc1.3_linux-aarch64.run`` 和 ``Atlas-A3-hdk-npu-firmware_7.7.0.3.228.run``，实际安装方式没有变化。

(1) 上传安装包，以 root 用户登录，将驱动和固件包上传至服务器（如 ``/home``）。

(2) 增加执行权限，进入安装包目录，执行以下命令。

    .. code-block:: shell

        chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
        chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run

(3) 安装驱动与固件，默认安装路径为 ``/usr/local/Ascend``。

    **安装驱动**：

    .. code-block:: shell

        ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all

    出现 ``Driver package installed successfully!`` 表示成功。

    **安装固件**：

    .. code-block:: shell

        ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full

    出现 ``Firmware package installed successfully!`` 表示成功。

    .. note::

        若未创建默认用户 ``HwHiAiUser``，需在安装命令中指定用户和组： ``./Ascend-hdk-*.run --full --install-username=<username> --install-usergroup=<usergroup>``

(4) 重启系统，根据提示决定是否重启。如需重启：

    .. code-block:: shell

        reboot

(5) 验证安装，执行以下命令查看驱动加载状态：

    .. code-block:: shell

        npu-smi info

    .. image:: ../../assets/advanced/npu-smi.png

3. CANN
~~~~~~~~~~~~~~~~~~~~~

请根据实际情况选择 ``.run`` 或 ``.deb`` 的 CANN 安装包，并请注意安装包对 ``aarch64`` 和 ``x86`` 做了区分。

以下以 A2 系列为例。A3 系列唯一区别是 ``ops`` 包名字有所变化，可以根据链接内实际情况选择。A3 内部包名类似于 ``Ascend-cann-A3-ops_<version>_linux-aarch64.run``，实际安装方式没有变化。


(1) 安装 Toolkit 开发套件
"""""""""""""""""""""""""

Toolkit 用于训练、推理及开发。

.. note::
    请确保安装目录可用空间大于 10G。

1. **授权与安装**：以 root 用户安装，默认安装路径为 ``/usr/local/Ascend``；以普通用户安装，默认安装路径为 ``${HOME}/Ascend``。

   .. code-block:: shell

       chmod +x Ascend-cann-toolkit_<version>_linux-aarch64.run
       ./Ascend-cann-toolkit_<version>_linux-aarch64.run --install

2. **配置环境变量**：以 root 用户为例，建议写入 ``~/.bashrc``。

   .. code-block:: shell

       source /usr/local/Ascend/ascend-toolkit/set_env.sh


(2) 安装 ops 算子包
"""""""""""""""""""""""""

需在安装 Toolkit 后执行。如需安装静态库，请将 ``--install`` 改为 ``--devel``。

.. code-block:: shell

    chmod +x Ascend-cann-<chip_type>-ops_<version>_linux-aarch64.run
    ./Ascend-cann-<chip_type>-ops_<version>_linux-aarch64.run --install


(3) 安装 NNAL 神经网络加速库（可选）
""""""""""""""""""""""""""""""""""""""""

包含 ATB 和 SiP 加速库。需在安装 Toolkit 后执行。

1. **授权与安装**：

   .. code-block:: shell

       chmod +x Ascend-cann-nnal_<version>_linux-aarch64.run
       ./Ascend-cann-nnal_<version>_linux-aarch64.run --install

2. **配置环境变量**：

   （二选一，不可同时配置）

   .. code-block:: shell

       # ATB
       source ${HOME}/Ascend/nnal/atb/set_env.sh

       # SiP
       source ${HOME}/Ascend/nnal/asdsip/set_env.sh



4. TorchNPU
~~~~~~~~~~~~~~~~~~~~

建议在安装 LlamaFactory 时一并安装 TorchNPU 插件，LlamaFactory 依赖内会持续更新稳定版本的 TorchNPU 插件。

.. code-block:: bash

    pip install -r requirements/npu.txt

当然您也可以手动下载后安装 TorchNPU 插件，例如：

.. code-block:: bash

    pip install torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl

安装 TorchNPU 插件需要注意：

- 下载的 TorchNPU 会对支持的 Python 版本做区分，请根据实际环境情况选择对应安装包，``pip install torch_npu`` 时，也会一并安装对应版本的 ``torch``。
- 环境里安装的 TorchNPU 和 ``torch`` 版本需要对齐。例如 TorchNPU 版本为 ``2.10.0.post2`` 时，``torch`` 的版本也需要为 ``2.10.0``。有时依赖互斥，安装不用依赖的过程会导致 ``torch`` 版本被更新，从而导致报错。

5. 验证安装
~~~~~~~~~~~~~~~~

执行以下 Python 脚本：

.. code-block:: python

    import torch
    import torch_npu
    print(torch.npu.is_available())

预期输出：``True``

.. image:: ../../assets/advanced/npu-torch.png

该情况说明 ``HDK``、``CANN`` 和 TorchNPU 都正常安装且生效。



.. _use_form_docker:

方式二：Docker 预安装镜像
---------------------------------

.. note::
  请确保宿主机已安装固件和驱动，可参考前文进行安装。

LlamaFactory 的官方镜像托管于 `Docker Hub <https://hub.docker.com/r/hiyouga/llamafactory/tags>`__ 和 `quay.io <https://quay.io/repository/ascend/llamafactory?tab=tags>`__，二者镜像无区别。

1. 拉取镜像
~~~~~~~~~~~~~~~~~~~~

下载 main 分支最新镜像时，需要同时根据设备和容器操作系统选择 Tag。A2 镜像在 Tag 中使用芯片标识 ``910b``，A3 镜像使用 ``a3``：

.. list-table::
   :align: left
   :widths: 15 20 30
   :header-rows: 1

   * - 硬件系列
     - 容器操作系统
     - Tag
   * - A2
     - Ubuntu 22.04
     - ``latest-910b-ubuntu``
   * - A3
     - Ubuntu 22.04
     - ``latest-a3-ubuntu``
   * - A2
     - openEuler 24.03
     - ``latest-910b-openeuler``
   * - A3
     - openEuler 24.03
     - ``latest-a3-openeuler``

根据上表确定 Tag 后，从 Docker Hub 或 quay.io 中选择一个镜像仓库，并执行对应的一条命令将镜像拉取到本地：

.. code-block:: shell

    # Docker Hub
    docker pull hiyouga/llamafactory:latest-910b-ubuntu
    docker pull hiyouga/llamafactory:latest-a3-ubuntu
    docker pull hiyouga/llamafactory:latest-910b-openeuler
    docker pull hiyouga/llamafactory:latest-a3-openeuler

    # quay.io
    docker pull quay.io/ascend/llamafactory:latest-910b-ubuntu
    docker pull quay.io/ascend/llamafactory:latest-a3-ubuntu
    docker pull quay.io/ascend/llamafactory:latest-910b-openeuler
    docker pull quay.io/ascend/llamafactory:latest-a3-openeuler

``latest`` 镜像使用 ``latest-<910b|a3>-<ubuntu|openeuler>`` 格式，定时构建会更新这些 Tag。Release 镜像则使用包含完整版本信息的 Tag：

.. code-block:: text

    <LlamaFactory-version>-cann<CANN-version>-torch_npu<TorchNPU-version>-<910b|a3>-<OS-and-version>-py<Python-version>


2. 启动容器
~~~~~~~~~~~~~~~~~~~~

使用以下命令启动容器（请根据实际情况修改 ``DOCKER_IMAGE`` 和 ``device``）：

.. code-block:: bash

    CONTAINER_NAME=llamafactory-npu
    DOCKER_IMAGE=hiyouga/llamafactory:latest-910b-ubuntu

    docker run -itd \
        --net=host \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        --shm-size=1200g \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /data:/data \
        --name "$CONTAINER_NAME" \
        "$DOCKER_IMAGE" \
        /bin/bash

.. note::

    配置 ``--privileged=true`` 可开启特权模式，赋予容器对底层硬件管理设备（如 ``/dev/davinci_manager``）的完整访问权限。这能解决多容器并行场景下，因权限限制导致的驱动初始化失败问题，确保 NPU 资源能被多个容器正常复用。

    **注意**：若未配置该参数，可能会出现首个容器占用后，后续容器因无权限而无法读取设备的情况。鉴于特权模式的权限过大，生产环境中请务必评估安全风险后慎重使用。


3. 验证容器环境
~~~~~~~~~~~~~~~~~~~~

容器启动后，进入容器：

.. code-block:: bash

    docker exec -it llamafactory-npu /bin/bash

然后加载 Ascend 环境，并检查软件版本、NPU 可用性与 LlamaFactory 命令：

.. code-block:: bash

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    npu-smi info
    python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
    llamafactory-cli help

.. note::

   部分驱动环境中的 ``npu-smi`` 位于 ``/usr/local/sbin/npu-smi``，此时需要调整挂载源路径。通过 ``--device=/dev/davinci<N>`` 可挂载更多 NPU 设备。容器内设备编号会自动重新映射（例如物理机 davinci6 代表容器内设备 0）。

环境验证通过后，可直接使用 ``llamafactory-cli train`` 启动训练。

.. _install_form_docker:

方式三：Docker 本地构建
-----------------------------

.. note::
  请确保宿主机已安装固件和驱动。

LlamaFactory 提供 :ref:`docker_build` 和 :ref:`docker_compose` 两种构建方式。


.. _docker_build:

1. 使用 Docker Build 构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在项目根目录下执行以下命令，构建 A2 Ubuntu 镜像：

.. code-block:: shell

    docker build \
        -f ./docker/docker-npu/Dockerfile \
        --build-arg BASE_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12 \
        --build-arg PIP_INDEX=https://pypi.org/simple \
        -t llamafactory:npu-910b-ubuntu \
        .

Dockerfile 默认使用 A2 Ubuntu 基础镜像。如需构建其他组合，可以替换 ``BASE_IMAGE`` 和本地镜像 Tag：

.. list-table::
   :align: left
   :widths: 15 20 55
   :header-rows: 1

   * - 硬件系列
     - 容器操作系统
     - ``BASE_IMAGE``
   * - A2
     - Ubuntu 22.04
     - ``quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12``
   * - A3
     - Ubuntu 22.04
     - ``quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12``
   * - A2
     - openEuler 24.03
     - ``quay.io/ascend/cann:9.1.0-910b-openeuler24.03-py3.12``
   * - A3
     - openEuler 24.03
     - ``quay.io/ascend/cann:9.1.0-a3-openeuler24.03-py3.12``

可用的 Dockerfile 构建参数如下：

.. list-table::
   :align: left
   :widths: 20 35 45
   :header-rows: 1

   * - 参数
     - 默认值
     - 用途
   * - ``BASE_IMAGE``
     - ``quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12``
     - 根据设备型号和容器操作系统选择对应的基础镜像
   * - ``PIP_INDEX``
     - ``https://pypi.org/simple``
     - 指定 Python 软件包索引
   * - ``PYTORCH_INDEX``
     - ``https://download.pytorch.org/whl/cpu``
     - 指定配合 TorchNPU 使用的 PyTorch wheel 索引
   * - ``HTTP_PROXY``
     - 空
     - 配置构建期间使用的 HTTP/HTTPS 代理

构建完成后，可以复用 :ref:`use_form_docker` 中的 ``docker run`` 命令，并将 ``DOCKER_IMAGE`` 设置为构建时通过 ``-t`` 指定的本地镜像名称。


.. _docker_compose:

2. 使用 Docker Compose 构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) 进入目录

.. code-block:: shell

  cd docker/docker-npu

(2) 启动容器。若本地镜像不存在，Docker Compose 会先构建镜像。请根据设备型号和容器操作系统选择 profile：

.. code-block:: shell

  # A2 + Ubuntu
  docker compose --profile a2-ubuntu up -d

  # A3 + Ubuntu
  docker compose --profile a3-ubuntu up -d

  # A2 + openEuler
  docker compose --profile a2-openeuler up -d

  # A3 + openEuler
  docker compose --profile a3-openeuler up -d

(3) 进入容器

.. code-block:: shell

  # A2 + Ubuntu
  docker exec -it llamafactory-910b-ubuntu /bin/bash

  # A3 + Ubuntu
  docker exec -it llamafactory-a3-ubuntu /bin/bash

  # A2 + openEuler
  docker exec -it llamafactory-910b-openeuler /bin/bash

  # A3 + openEuler
  docker exec -it llamafactory-a3-openeuler /bin/bash

如果只需要构建镜像而不启动容器，可以使用 ``docker compose --profile <profile> build``。

.. note::
  构建前，请检查 ``docker-compose.yml`` 中的 ``devices`` 列表，当前默认只会挂载卡 0，请根据需要修改。
