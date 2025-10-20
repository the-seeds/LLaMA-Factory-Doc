# Data Engine

## Data Engine 工作原理

### 1. 整体架构

`DataEngine` 是 v1 数据处理的核心类，继承自 PyTorch 的 `Dataset` 类，负责各种插件的接入，其他功能（如数据格式转换、数据加载等）均通过插件的形式实现并接入 `DataEngine`，工作流程如下：

```
数据配置 → 加载数据集信息 → 加载数据集 → 构建数据索引 → 转换数据格式 → 返回样本
```

### 2. 初始化流程

```python
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine

# 1. 创建数据参数
data_args = DataArguments(
    dataset="v1_sft_demo.jsonl",
    dataset_dir="./data",
    cutoff_len=2048
)

# 2. 初始化 Data Engine
data_engine = DataEngine(data_args=data_args)

# 3. 访问数据
sample = data_engine[0]  # 获取第一个样本
```

### 3. 核心方法

#### 3.1 获取数据集信息：`get_dataset_info()`

根据 `dataset` 参数加载数据集配置：

- **本地 YAML 文件**：`dataset_dir/dataset.yaml`
- **HF Hub YAML 文件**：从 HF Hub 下载
- **本地文件/目录**：自动创建配置 `{"default": {"file_name": dataset}}`
- **HF Hub 数据集**：自动创建配置 `{"default": {"hf_hub_url": dataset}}`

#### 3.2 加载数据集：`load_dataset()`

遍历所有数据集配置，根据不同的数据源加载数据：

```python
for key, value in self.dataset_infos.items():
    split = value.get("split", "train")
    streaming = value.get("streaming", False)
    
    if "hf_hub_url" in value:
        # 从 HF Hub 加载
        dataset = load_dataset(value["hf_hub_url"], split=split, streaming=streaming)
    else:
        # 使用 DataLoaderPlugin 加载本地文件
        dataset = DataLoaderPlugin(args=self.args).auto_load_data(value)
    
    self.datasets[key] = dataset
```

#### 3.3 构建数据索引：`build_data_index()`

为每个数据集创建索引列表 `[(dataset_name, sample_index), ...]`：

```python
for dataset_name, dataset in self.datasets.items():
    # 创建基础索引
    data_index = [(dataset_name, idx) for idx in range(len(dataset))]
    
    # 根据 size 和 weight 调整索引
    size = self.dataset_infos[dataset_name].get("size")
    weight = self.dataset_infos[dataset_name].get("weight")
    if size or weight:
        data_index = DataIndexPlugin().adjust_data_index(data_index, size, weight)
    
    self.data_index.extend(data_index)
```

#### 3.4 转换数据样本：`_convert_data_sample()`

将原始数据转换为标准格式：

```python
def _convert_data_sample(self, raw_sample: dict, dataset_name: str) -> Sample:
    converter = self.dataset_infos[dataset_name].get("converter")
    if converter is not None:
        # 使用指定的转换器
        from ..plugins.data_plugins.converter import get_converter
        return {"_dataset_name": dataset_name, **get_converter(converter)(raw_sample)}
    else:
        # 已经是标准格式
        return {"_dataset_name": dataset_name, **raw_sample}
```

### 4. 数据访问方式

#### 4.1 整数索引访问

```python
sample = data_engine[0]  # 获取第一个样本
```

#### 4.2 切片访问

```python
samples = data_engine[0:10]  # 获取前 10 个样本
```

#### 4.3 列表索引访问

```python
samples = data_engine[[0, 5, 10]]  # 获取指定索引的样本
```
