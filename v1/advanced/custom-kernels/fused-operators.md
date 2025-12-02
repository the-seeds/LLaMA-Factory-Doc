# Fused Operators

### NpuFusedRMSNorm
RMSNorm（Root Mean Square Layer Normalization）是一种常用于大模型的归一化方法。在推理或训练中，RMSNorm 融合算子 将bias、residual等操作进行融合，可以减少显存访问次数，加速计算。

 Ascend npu 通过 `torch_npu.npu_rms_norm` 接口提供 RMSNorm 融合算子调用接口，支持 float16, bfloat16, float 等数据格式。RMSNorm 算子常见于Qwen等LLM模型中，由于torch侧没有提供 RMSNorm 算子的接口，因此在模型中通常是以自定义类的形式出现，通过替换 RMSNorm 类的 `forward` 方法即可使能。

 ```python
def _npu_rms_forward(self, hidden_states):
    """NPU forward implementation for RMSNorm.

    Args:
        self: RMSNorm module instance with `weight` and `variance_epsilon`.
        hidden_states: Input hidden states tensor, same shape as the baseline.

    Returns:
        Normalized tensor consistent with the baseline RMSNorm behavior.
    """
    import torch_npu

    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

 ```

 在 LLaMA-Factory 中，通过 `NpuRMSNormKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel(model, NpuRMSNormKernel)` 即可针对已适配的模型使能 npu RMSNorm 融合算子。

### NpuFusedSwiGlu
SwiGLU（Swish-Gated Linear Unit）是一种结合了Swish激活函数和门控线性单元（GLU）的混合激活函数，其主要功能是对输入张量进行门控线性变换，近年来被广泛应用于 LLM 模型中的 MLP 层。SwiGLU 融合算子将分割、激活、矩阵乘等多个操作融合为单一硬件指令，避免多次内核启动开销。

Ascend npu 通过 `torch_npu.npu_swiglu` 接口提供 SwiGLU 融合算子调用接口，支持 float16，bfloat16，float SwiGLU 算子常见于Qwen等LLM模型中，由于torch侧没有提供 SwiGLU 算子的接口，因此在模型中通常是以自定义类的形式出现，通过替换 SwiGLU 类的 `forward` 方法即可使能。替换过程可参考如下示例：

```python
# 原始 MLP forward 方法：
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

# 替换后的 forward 方法：
def _npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )
```

 在 LLaMA-Factory 中，通过 `NpuSwiGluKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel(model, NpuSwiGluKernel)` 即可针对已适配的模型使能 npu SwiGLU 融合算子。对于未适配度模型，如有需要，您可根据示例以及[开发者文档](../../dev-guide/plugins/model-plugins/kernels.md)自行适配。


### NpuFusedRoPE
RoPE（Rotary Positional Embedding，旋转式位置嵌入） 是一种位置编码技术，广泛应用于 Qwen 等 LLM 模型中，用于有效编码文本序列的位置信息。它结合了绝对位置编码的稳定性与相对位置编码的灵活性，同时具备优秀的长度泛化能力。传统 RoPE 算子通常在 LLM 等模型结构中通过自定义函数的形式实现。RoPE 融合算子将原计算流程合并为单个硬件优化算子，从而提升性能。

Ascend npu 通过 `torch_npu.npu_rotary_mul` 提供 RoPE 融合算子调用接口，支持 float16，bfloat16，float32 等数据格式。以 Qwen3 系列模型为例，通过替换其 `apply_rotary_pos_emb` 函数即可实现 RoPE融合算子使能：

```python
# 原始 apply_rotary_pos_emb：
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 替换 RoPE 融合算子后：
def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed
```

 在 LLaMA-Factory 中，通过 `NpuRoPEKernel` 或 `NpuQwen2VLRoPEKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel(model, NpuRoPEKernel)` 即可针对已适配的模型使能 npu RoPE 融合算子。对于未适配度模型，如有需要，您可根据示例以及[开发者文档](../../dev-guide/plugins/model-plugins/kernels.md)自行适配。


### NpuFusedMoE

