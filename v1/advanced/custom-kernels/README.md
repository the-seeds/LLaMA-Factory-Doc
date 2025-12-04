# LLaMA-Factory Kernels 系统

## 概述

LLaMA-Factory Kernels 系统用于管理不同硬件设备提供的高性能计算内核（kernel）实现，该系统通过替换模型中的关键模块（如 RMSNorm、SwiGLU、RoPE、MoE 等）为硬件优化的版本，从而显著提升模型训练和推理的性能。

Kernels 系统采用基于注册表的自动发现机制，能够根据当前运行环境自动检测可用的硬件设备（NPU、CUDA、XPU 等），并使能相应的高性能 kernels。这种设计使得用户无需关心底层实现细节，只需简单调用接口即可获得性能提升。

## 核心特性

- **自动注册机制**：基于元类实现自动注册系统，简化 kernel 的添加和管理。当 kernel 类按照 `MetaKernel` 定义的协议声明 `type` 和 `device` 属性时，会自动注册到全局注册表中，无需手动注册。

- **设备适配感知**：自动检测当前硬件设备（NPU、CUDA、XPU 等）并应用相应的优化。系统会跳过不支持的设备，确保在不同环境下都能正常工作。

- **模块化设计**：每个 kernel 独立实现，互不干扰。可以单独应用某个 kernel，也可以批量应用所有可用的 kernels。

- **后向兼容**：kernel 替换不修改模型权重，保持数值一致性。优化后的实现与原始实现保持精度一致（在浮点误差范围内）。

- **灵活扩展**：通过继承 `MetaKernel` 基类，可以轻松添加新的 kernel 实现，支持新的硬件设备或优化算法。

## 使用方式

### 1. 通过训练 YAML 配置文件使用

要在训练过程中使能 kernels，只需在配置文件中增加如下配置，即可自动使能所有可用 kernels：

```yaml
...
use_v1_kernels: true
...
```

### 2. 调用 API 使能

#### 2.1 apply_available_kernels 使能所有可用 kernels

`apply_available_kernels` API 能够自动发现当前设备上所有可用的 kernels 并应用到模型上：

```python
from transformers import AutoModelForCausalLM
from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_available_kernels

# 加载模型
model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")

# 自动应用所有可用的 kernels
model = apply_available_kernels(model)
```

#### 2.2 apply_kernel 使能特定 kernel

如果需要更精细的控制，例如在某些场合单独应用某个 kernel，可以手动调用 `apply_kernel` 函数来使能特定的 kernel：

```python
from transformers import AutoModelForCausalLM
from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import NpuRMSNormKernel
from llamafactory.v1.plugins.model_plugins.kernels.mlp.npu_swiglu import NpuSwiGluKernel
from llamafactory.v1.plugins.model_plugins.kernels.rope.npu_rope import NpuRoPEKernel

# 加载模型
model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")

# 手动应用各个 kernels
# 在LLaMA-Factory中，加载模型的位置通常位于/src/llamafactory/model/loader.py
model = apply_kernel(model, NpuRoPEKernel)
model = apply_kernel(model, NpuRMSNormKernel)
model = apply_kernel(model, NpuSwiGluKernel)
```

### 3. 查询已注册的可用 kernels

`discover_kernels` 接口可以查询当前设备上所有已注册的可用 kernels，通常而言，这个接口无需用户手动调用，但是在故障检查的时候，如果发现某个 kernel 未被成功使能，可以检查一下这个函数的返回值是否符合预期。

```python
from llamafactory.v1.plugins.model_plugins.kernels.registry import discover_kernels

# 发现所有可用的 kernels
available_kernels = discover_kernels(model)

for kernel in available_kernels:
    print(f"Kernel: {kernel.__name__}, Type: {kernel.type}, Device: {kernel.device}")
```

### 当前已实现的 kernels

| kernel | kernel 类型 | 支持的设备 | 备注 |
|--------|------------|-----------|------|
| [NpuRMSNormKernel](./fused-operators.md/#npufusedrmsnorm) | RMSNORM | NPU | NPU 设备的高性能 RMSNorm 实现 |
| [NpuSwiGluKernel](./fused-operators.md/#npufusedswiglu) | SWIGLU | NPU | NPU 设备的高性能 SwiGLU 实现，部分模型的 MLP 层不支持 |
| [NpuRoPEKernel](./fused-operators.md/#npufusedrope) | ROPE | NPU | NPU 设备的高性能 RoPE 实现，适用于大多数模型 |
| [NpuQwen2VLRoPEKernel](./fused-operators.md/#npufusedrope) | ROPE | NPU | 多模态 RoPE 实现，单独适配 Qwen2-VL 模型，`auto_register = False`，需要手动应用 |
| [NpuMoEFusedMoEKernel](./fused-operators.md/#npufusedmoe) | MOE | NPU | MoE 融合算子，当前仅适配 Qwen3VLMoE 以及 Qwen3-MoE 模型 |

我们会持续适配更多的 kernels，如果您需要自己开发新的 kernels，请参考我们的 [Kernel 开发文档](../../dev-guide/plugins/model-plugins/kernels.md)，欢迎您向 LLaMA-Factory 贡献代码。
