# Kernels plugins

## 概览
LLaMA-Factory 通过 Kernels plugins 系统，依据不同硬件设备提供高性能计算内核（kernel）实现。该系统通过注册表机制管理所有 kernel，通过 `AutoRegisterKernelMeta` 元类实现 kernel 定义后自动注册，由 `apply_kernel` 方法来使能指定的 kernel，`apply_available_kernels` 可使能注册表中当前环境所有可用的 kernels。

## 架构设计

### 核心组件

#### 1. KernelRegistry（注册表）

`KernelRegistry` 是一个单例模式的注册表，用于管理所有 kernel 实现。它维护一个二维字典结构：`{KernelType: {DeviceType: KernelClass}}`。

```python
# 注册表结构
{
    KernelType.RMSNORM: {
        DeviceType.NPU: NpuRMSNormKernel,
        DeviceType.CUDA: CudaRMSNormKernel,
    },
    KernelType.SWIGLU: {
        DeviceType.NPU: NpuSwiGluKernel,
    },
    ...
}
```


#### 2. AutoRegisterKernelMeta (元类)

所有 kernel 均以该类为元类构造，该元类实现了 kernel 自动注册功能，当定义一个新的 kernel 类时，无需手动去调整注册表，也无需手动实例化该类，`AutoRegisterKernelMeta` 可自动完成注册。 

**自动注册机制**：
- `AutoRegisterKernelMeta` 在类创建时自动检查 `type` 和 `device` 属性
- 如果 `type` 和 `device` 属性都已定义且 `auto_register=True`，则自动注册，`auto_register` 默认为 True。
- 可以通过设置 `auto_register=False` 禁用自动注册
- 特别地，当针对同一种设备的同一种 kernel 类型定义了多个 kernel 实现时，注册表中该条目会被多次重写，具体指向的 kernel 实现由 `_ensure_kernels_loaded` 中导入顺序决定。该行为是未定义行为，如果确实要为同一种 kernel 定义不同实现，请保留唯一一个自动注册，其它实现则将自动注册功能禁用。


#### 2. MetaKernel（基类）

所有 kernel 的实现都应当继承自 `MetaKernel` 抽象基类。`MetaKernel` 通过 `AutoRegisterKernelMeta` 元类，在子类定义了 `type` 和 `device` 属性的前提下，会自动注册到全局注册表中。若自定义的 kernel 未继承自该接口，即使注册了，在调用 `discover_kernels` 时也会被静默跳过，通过 `apply_kernel` 强制使能时会抛出 `ValueError`。


#### 3. 类型系统

**KernelType**（kernel 类型）：
当定义一个新的 kernel 不属于这其中任何一种类型的时候，应当为 `KernelType` 新增一个类型定义。当前已预定义如下 kernel type 字段：
- `RMSNORM`：RMSNorm Layer
- `SWIGLU`：SwiGLU 激活的 MLP Layer
- `FLASH_ATTENTION`：Flash Attention Layer
- `ROPE`：旋转位置编码
- `MOE`：MoE Layer

**DeviceType**（设备类型）：
kernel 可支持的设备类型，当前已支持如下设备类型，其他设备暂未适配：

- `CPU`：CPU 
- `CUDA`：NVIDIA GPU
- `NPU`：Ascend NPU
- `XPU`：Intel XPU
- `MPS`：Apple GPU


## Kernel 系统 API 设计

kernel 系统通过注册表管理所有定义的 kernel，并通过 Metaclass 机制实现 kernel 的自动注册。所有 kernel 均需要实现各自的 `apply` 方法，使能时，上层框架会通过 `apply_kernel` 和 `apply_available_kernels` 两个 API 接口自动调用当前运行环境下的可用 kernel。

### **KERNEL_REGISTRY**：全局 kernel 注册表实例（单例）。

KERNEL_REGISTRY 作为所有 kernel 的注册表，被实现为全局单例对象，该类提供了两个 API：`register()` 和 `get_kernel()` 用于注册和查找 kernel。该类的接口定义如下：

```python
class KernelRegistry:
    _instance: Optional["KernelRegistry"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "KernelRegistry":
        # 重载 __new__ 方法以实现单例模式
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self, kernel_type: KernelType, device_type: DeviceType, kernel_impl: Optional[Callable[..., Any]]
    ) -> None:
        """注册一个 kernel 实现.

        Args:
            kernel_type: the type of the kernel (e.g., KernelType.FLASH_ATTENTION).
            device_type: the device type the kernel is adapted to (e.g., DeviceType.CUDA).
            kernel_impl: the actual kernel function or class.
        """
        

    def get_kernel(self, kernel_type: KernelType, device_type: DeviceType) -> Optional[Callable[..., Any]]:
        """根据设备类型和 kernel 类型获取已注册的 kernel 类
        
        Args:
            kernel_type: enum item, defined by `KernelType` class
            device_type: enum item, defined by `DeviceType` class
        
        """

        
# 注册 kernel
KERNEL_REGISTRY.register(kernel_type, device_type, kernel_impl)

# 获取 kernel
kernel = KERNEL_REGISTRY.get_kernel(kernel_type, device_type)
```

### **AutoRegisterKernelMeta**

AutoRegisterKernelMeta 继承自 `abc.ABCMeta`，作为所有 kernel 类的元类，为 kernel 提供了自动注册功能，所有类只要以 AutoRegisterKernelMeta 为元类定义，都会在 import 该类的时候自动尝试向 `KERNEL_REGISTRY` 中注册自身。AutoRegisterKernelMeta 的定义如下：

```python
class AutoRegisterKernelMeta(ABCMeta):
    """MetaKernel 类的元类。

    此元类检查新创建的类是否同时定义了 `type` 和 `device` 属性。
    如果是，它会自动将 kernel 注册到全局 KERNEL_REGISTRY 中，无需手动注册。

    要禁用特定类的自动注册，请设置 `auto_register = False`。
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Check if auto-registration is disabled
        auto_register = namespace.get("auto_register", True)

        # Only auto-register if the class has both type and device attributes defined
        # and they are not None (skip base classes like MetaKernel itself)
        # and auto_register is True
        kernel_type = namespace.get("type")
        device_type = namespace.get("device")

        if auto_register and kernel_type is not None and device_type is not None:
            # Auto-register this kernel
            KERNEL_REGISTRY.register(kernel_type, device_type, cls)

        return cls

```

### **MetaKernel** 

`MetaKernel` 作为所有 kernel 类的模板，定义了 kernel 与上层系统交互的协议，所有继承自 `MetaKernel` 的类必须声明如下属性与接口：

- `type`: 类属性，定义 kernel 类型，值为 `KernelType` 枚举值，默认为 None。

- `device`: 类属性，定义 kernel 支持的设备类型，值为 `DeviceType` 枚举值，默认为 None。

- `kernel`: 类属性，实际的 kernel 内核实现。可以是函数、类或其他可调用对象，默认为 None。

- `auto_register`: 类属性，声明该 kernel 类是否自动注册至 KERNEL_REGISTRY。默认值为 `True`，设置为 `False` 可以禁用自动注册（用于模型特定的或实验性的 kernels）。

- `apply`: 抽象类方法，对模型使能 kernel 的调用接口，用于控制该 kernel 的具体使能逻辑，此方法每个子类必须实现，上层框架通过调用每个 kernel 的 `apply` 方法来实现对应 kernel 的使能。

MetaKernel 的定义如下：

```python
class MetaKernel(ABC, metaclass=AutoRegisterKernelMeta):
    """所有 kernel 实现的基类。

    当子类定义了 `type` 和 `device` 属性时，会自动注册。
    要禁用自动注册，请设置 `auto_register = False`。

    """

    type: Optional[KernelType] = None
    device: Optional[DeviceType] = None
    kernel: Optional[Callable] = None

    @classmethod
    @abstractmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        """将 kernel 应用到模型。

        此方法承载该 kernel 的实际使能逻辑。包括检查 kernel 是否可以被应用（例如，依赖项是否已安装，目标模块是否存在）、执行 kernel 替换等。

        Args:
            model: HFModel 模型实例。

        Returns:
            使能 kernel 后的 HF Model
        """
        raise NotImplementedError

```

### **_ensure_kernels_loaded**

`_ensure_kernels_loaded` 仅被用于触发所有已定义 kernel 的自动注册行为，本接口为受保护接口，不应当在外部调用，也无需任何入参。在新增一个 kernel module 时，应当在 `kernel_modules` 列表中增加该 module 的路径（注意：仅需 python module 层级的增减需要修改 `kernel_modules`，kernel class 层级的增减无需修改 ）。

该接口的实现如下：

```python
def _ensure_kernels_loaded() -> None:
    """确保所有 kernel 实现被导入并注册。

    该函数动态导入所有 kernel 类所在的模块以触发它们的自动注册。
    Python 的 module 系统确保每个模块只执行一次（缓存在 sys.modules 中），
    因此重复调用是安全且快速的。
    """

    # 要导入的 kernel 模块路径列表
    kernel_modules = [
        "rms_norm.npu_rms_norm",
        "rope.npu_rope",
        "mlp.npu_swiglu",
        "mlp.npu_fused_moe",
        # 在此处添加新创建的 kernel 模块，以确保会被自动注册
    ]

    # 导入每个模块以触发 kernel 注册
    for module_name in kernel_modules:
        try:
            __import__(f"{__package__}.{module_name}", fromlist=["*"])
        except ImportError:
            # 静默忽略导入错误（例如，缺少依赖项如 torch_npu）
            pass
```


### **discover_kernels**

`discover_kernels` 接口用于发现并返回当前设备上所有可用的 kernel 类。该接口通过当前设备环境查找 KERNEL_REGISTRY 中对应的已注册 kernel，如果某个 kernel 被禁用了自动注册且未手动注册，则该 kernel 无法被此接口发现，但是依旧可以通过 `apply_kernel` 接口手动调用。该接口的定义如下：

```python
def discover_kernels(model: HFModel = None) -> list[type[MetaKernel]]:
    """发现并返回当前设备上所有已注册的 kernel 类。

    该函数检查运行时环境（设备类型）并返回为该设备注册的所有 MetaKernel 类。
    每个 kernel 的 `apply()` 方法负责检查它是否可以被实际应用（例如，
    所需的依赖项是否已安装，目标模块是否存在于模型中）。

    该函数自动发现 KERNEL_REGISTRY 中注册的所有 kernel，在首次调用时，它会动态导入所有 kernel 实现模块以触发它们的自动注册。

    Args:
        model: 可选的 HuggingFace 模型实例（用于未来基于模型结构的路由检测），默认值为 None

    Returns:
        当前设备上可用的 kernel 类列表
    """

```

### apply_kernel

对模型使能指定的 kernel，该接口至少需要传入两个入参：待使能的模型、指定的 kernel。该接口有如下调用约束条件：

1. 传入的 kernel 必须为 `MetaKernel` 的子类
2. 传入的模型应当为 HFModel 实例
3. 当前运行环境下的设备类型与传入的 kernel 需求设备类型相匹配

若不满足约束条件强行调用，可能会产生 `ValueError` 等异常，在调用该接口时请确认约束条件。该接口的定义如下：

```python
def apply_kernel(model: HFModel, kernel: type[MetaKernel], /, **kwargs) -> "HFModel":
    """调用 kernel 的 `apply` 方法使能指定 kernel
    
    Args:
        model: HFModel 实例
        kernel: 目标 kernel 类，需要满足上述约束条件
    
    Return:
        若调用成功，则返回使能 kernel 后的 HFModel 实例，或由于传入的 kernel 不支持该模型，直接返回原始模型；若由于不满足约束条件而调用则会引发异常。
    """

```

**apply_kernel 用法示例**：
```python
from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
from llamafactory.v1.plugins.model_plugins.kernels.rms_norm.npu_rms_norm import NpuRMSNormKernel

model = apply_kernel(model, NpuRMSNormKernel)
```

### apply_available_kernels

对模型使能所有可用的 kernel，此接口为高级 API，包装了 `discover_kernels` 和 `apply_kernel` 这两个接口，调用时同样需要满足其约束条件。正常情况下，直接调用本接口不会出现异常，除非某些 kernel 的 `apply` 方法实现存在未知问题引发了 runtime error。本接口至少需要一个 `model` 入参，接口定义如下：

```python
def apply_available_kernels(model: HFModel, **kwargs) -> "HFModel":
    """Apply all available kernels to the model.
    
    Args:
        model: HFModel 实例
    
    Return:
        返回使能 kernel 后的 HFModel 实例

    """
```


### 依赖要求

- **NPU Kernels**：需要安装 `torch_npu` 库
- **CUDA Kernels**：需要安装 `torch` 并支持 CUDA
- **其他依赖**：每个 kernel 可能有特定的依赖要求

### 数值精度

- Kernel 替换不会修改模型权重，但可能由于实现差异导致微小的数值差异
- 优化后的实现应该与原始实现保持合理的精度误差（在浮点误差范围内）
- 如果对精度有严格要求，建议进行对比测试

### 性能影响

- Kernel 优化主要提升计算性能，不会改变模型的数值行为
- 性能提升取决于硬件设备和模型架构
- 某些情况下，kernel 优化可能不会带来明显的性能提升

## 扩展 Kernels

当前 LLaMA-Factory 内置的 kernel 有限，如果用户有针对特定模型或者设备的 kernel，可以按照下述步骤去实现并接入 LLaMA-Factory。

### 创建新 Kernel 的步骤

#### 1. 创建 Kernel 实现文件

在相应的子目录中创建新的 kernel 实现文件，例如 `mlp/cuda_swiglu.py`，或者在已有的kernel 实现文件中新增一个Kernel类（此时不需要执行步骤2）：

```python
import types
import re
import torch

from .....extras.types import HFModel
from ..constants import DeviceType, KernelType
from ..registry import MetaSwiGluKernel


def _cuda_swiglu_forward(self, hidden_state):
    """CUDA 优化的 SwiGLU 前向传播"""
    # 实现 CUDA 优化的逻辑
    # 例如使用 torch.cuda 或自定义 CUDA kernel
    gate = self.gate_proj(hidden_state)
    up = self.up_proj(hidden_state)
    # ... CUDA 优化实现 ...
    return self.down_proj(swiglu_output)


class CudaSwiGluKernel(MetaSwiGluKernel):
    type = KernelType.SWIGLU
    device = DeviceType.CUDA
    kernel = _cuda_swiglu_forward
    
    # 可选：排除不兼容的模块
    except_modules = ["DiTMLP", "GPT2MLP"]
    
    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        """应用 kernel 到模型"""
        # 检查依赖是否可用
        if not torch.cuda.is_available():
            return model
        
        # 匹配并替换模块
        swiglu_pattern = re.compile("MLP", re.IGNORECASE)
        for name, module in model.named_modules():
            if (
                re.search(swiglu_pattern, module.__class__.__name__)
                and module.__class__.__name__ not in cls.except_modules
            ):
                module.forward = types.MethodType(cls.kernel, module)
        
        return model
```

#### 2. 注册 Kernel 模块

在 `registry.py` 的 `_ensure_kernels_loaded()` 函数中添加新模块的导入路径：

```python
def _ensure_kernels_loaded() -> None:
    kernel_modules = [
        "rms_norm.npu_rms_norm",
        "rope.npu_rope",
        "mlp.npu_swiglu",
        "mlp.npu_fused_moe",
        "mlp.cuda_swiglu",  # 添加新模块
    ]
    # ...
```

#### 3. 测试 Kernel

创建测试用例验证 kernel 的正确性：

```python
import unittest
from transformers import AutoModelForCausalLM
from llamafactory.v1.plugins.model_plugins.kernels.registry import apply_kernel
from llamafactory.v1.plugins.model_plugins.kernels.mlp.cuda_swiglu import CudaSwiGluKernel

class TestCudaSwiGluKernel(unittest.TestCase):
    def test_apply_kernel(self):
        model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")
        original_forward = model.model.layers[0].mlp.forward
        
        model = apply_kernel(model, CudaSwiGluKernel)
        
        # 验证 forward 方法已被替换
        assert model.model.layers[0].mlp.forward is not original_forward
```

#### 4. 使用 Kernel

完成上述步骤后，直接调用 `apply_available_kernels()` 即可进行使能，如果一切正常，后续训练应当可以正常调用到相应的 kernel 核函数，也可以在自定义 kernel 的 `apply` 方法或者核函数中打印一些日志，以便确认 kernel 生效。

#### 5. 禁用自动注册（可选）

如果某个 kernel 是模型特定的或实验性的，可以禁用自动注册：

```python
class ExperimentalKernel(MetaKernel):
    type = KernelType.RMSNORM
    device = DeviceType.CUDA
    auto_register = False  # 禁用自动注册
    
    @classmethod
    def apply(cls, model: HFModel, **kwargs) -> HFModel:
        # 实现逻辑
        pass
```

禁用自动注册后，需要手动调用 `KERNEL_REGISTRY.register()` 或直接使用 `apply_kernel()`。

## 异常处理

### 依赖不可用

当 kernel 所需的依赖不可用时，`apply()` 方法应该直接返回原模型，而不是抛出异常：

```python
@classmethod
def apply(cls, model: HFModel, **kwargs) -> HFModel:
    if not is_torch_npu_available():
        return model  # 依赖不可用时直接返回，不抛出异常
    
    # 应用 kernel
    # ...
```

### 设备类型不匹配

当尝试应用不匹配当前设备的 kernel 时，`apply_kernel()` 会抛出 `ValueError`：

```python
try:
    model = apply_kernel(model, NpuRMSNormKernel)
except ValueError as e:
    print(f"设备类型不匹配: {e}")
```

### 注册冲突

如果同一个 `(kernel_type, device_type)` 组合被注册多次，会发出警告但不会抛出异常：

```python
if device_type in self._registry[kernel_type]:
    print(f"Warning: Overwriting kernel for {kernel_type.name} on {device_type.name}.")
```

### 常见错误处理

**Kernel 未应用**：
- 检查设备类型：`discover_kernels(model)` 返回的列表是否为空
- 检查依赖：确认所需的库已安装（如 `torch_npu`）
- 检查模块匹配：确认模型中的模块类名符合 kernel 的匹配规则

**Loss 数值不一致**：
- 检查 kernel 实现是否正确进行了等价替换，模型本身是否有特殊处理
- 验证参数传递：确认 epsilon 值、归一化参数等是否正确传递
- 对比测试：对比原始实现和优化实现的中间结果

**导入错误**：
- 确认 kernel 模块路径已添加到 `_ensure_kernels_loaded()` 中
- 检查模块的 `__init__.py` 是否正确配置
- 确认所有依赖都已安装

在发现任何预期外的异常时，欢迎您向社区提交 issue 协助改进 LLaMA-Factory。
