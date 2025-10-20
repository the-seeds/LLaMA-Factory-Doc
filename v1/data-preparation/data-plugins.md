# Data Plugins


## Data Converter Plugin

### 1. Data Converter Plugin 简介

Data Converter 负责将非标准格式的数据集转换为 v1 的标准 Messages 格式。这使得用户可以继续使用现有的数据集（如 Alpaca 格式），而无需手动转换。针对自定义格式的数据集，用户也可以通过构建对应的自定义 Data Converter 插件，来负责其数据格式标准化。

### 2. Alpaca Converter 详解

#### 2.1 Alpaca 格式

Alpaca 格式是一种常见的指令微调数据格式：

```json
{
  "system": "You are a helpful assistant.",  // 可选
  "instruction": "Describe a process of making crepes.",
  "input": "",  // 可选
  "output": "Making crepes is an easy and delicious process..."
}
```

#### 2.2 转换逻辑

`alpaca_converter` 函数将 Alpaca 格式转换为标准格式：

```python
def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    messages = []
    
    # 1. 添加系统提示（如果存在）
    if "system" in raw_sample:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "value": raw_sample["system"]}],
            "loss_weight": 0.0
        })
    
    # 2. 添加用户输入（instruction + input）
    if "instruction" in raw_sample or "input" in raw_sample:
        user_content = raw_sample.get("instruction", "") + raw_sample.get("input", "")
        messages.append({
            "role": "user",
            "content": [{"type": "text", "value": user_content}],
            "loss_weight": 0.0
        })
    
    # 3. 添加模型回复
    if "output" in raw_sample:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "value": raw_sample["output"]}],
            "loss_weight": 1.0
        })
    
    return {"messages": messages}
```

#### 2.3 转换示例

**输入（Alpaca 格式）：**

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

**输出（标准格式）：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "What is the capital of France?"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "The capital of France is Paris."}],
      "loss_weight": 1.0
    }
  ]
}
```

### 3. 自定义转换器

#### 3.1 创建自定义转换器

如果您有自己的数据格式，可以轻松添加自定义转换器：

```python
# src/llamafactory/v1/plugins/data_plugins/converter.py

from typing import TypedDict, NotRequired
from ...extras.types import SFTSample

# 1. 定义输入格式的类型
class MyCustomSample(TypedDict, total=False):
    question: str
    answer: str
    context: NotRequired[str]

# 2. 实现转换函数
def my_custom_converter(raw_sample: MyCustomSample) -> SFTSample:
    messages = []
    
    # 构建用户消息
    user_text = raw_sample["question"]
    if "context" in raw_sample:
        user_text = f"Context: {raw_sample['context']}\n\nQuestion: {user_text}"
    
    messages.append({
        "role": "user",
        "content": [{"type": "text", "value": user_text}],
        "loss_weight": 0.0
    })
    
    # 构建助手消息
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "value": raw_sample["answer"]}],
        "loss_weight": 1.0
    })
    
    return {"messages": messages}

# 3. 注册转换器
# 注意：alpaca_converter 需要在前面定义或导入
CONVERTERS = {
    "alpaca": alpaca_converter,
    "my_custom": my_custom_converter,  # 添加您的转换器
}
```

#### 3.2 使用自定义转换器

在 YAML 配置中指定转换器名称：

```yaml
my_dataset:
  file_name: my_data.json
  converter: my_custom
```

---

## Data Loader Plugin

负责从本地文件加载数据集，支持多种文件格式。

#### 1. 支持的文件格式

- **JSON**: `.json`
- **JSONL**: `.jsonl`
- **CSV**: `.csv`
- **Parquet**: `.parquet`
- **Arrow**: `.arrow`
- **Text**: `.txt`

#### 2. 加载逻辑

```python
class DataLoaderPlugin:
    def auto_load_data(self, dataset_info: DatasetInfo) -> HFDataset:
        filepath = os.path.join(dataset_dir, dataset_info["file_name"])
        
        if os.path.isdir(filepath):
            # 目录：加载目录下所有同类型文件
            dataset = load_dataset(filetype, data_dir=filepath, split=split)
        elif os.path.isfile(filepath):
            # 文件：加载单个文件
            dataset = load_dataset(filetype, data_files=filepath, split=split)
        
        if streaming:
            dataset = dataset.to_iterable_dataset()
        
        return dataset
```

---

## Data Index Plugin

负责调整数据索引，支持控制数据集大小和权重。

#### 1. 使用 size 参数

限制使用的样本数量：

```yaml
my_dataset:
  file_name: large_dataset.json
  size: 1000  # 只使用前 1000 个样本
```

#### 2. 使用 weight 参数

调整数据集在混合数据中的权重：

```yaml
dataset_a:
  file_name: data_a.json
  weight: 1.0

dataset_b:
  file_name: data_b.json
  weight: 2.0  # dataset_b 的样本出现频率是 dataset_a 的 2 倍
```

**说明**：

- 当 `weight=1.0` 时，数据集按原始比例采样
- 当 `weight=2.0` 时，该数据集的索引会复制 2 倍，使其样本出现频率翻倍（？）
- 适用于在多个数据集混合训练时，调整不同数据集的重要性

---

## Data Selector Plugin

**状态**：该插件的详细功能和用法正在开发中，暂未完全落地。

**预期功能**：

- 支持灵活的数据访问方式
- 提供数据过滤和选择机制
- 支持基于条件的数据筛选

该功能的具体实现和使用方法将在后续版本中完善

---

