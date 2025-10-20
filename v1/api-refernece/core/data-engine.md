# Data Engine

---

### 1. 整体结构

```python
class DataEngine(data_args: 'DataArguments'):
    ...
```

`DataEngine` 是 v1 数据处理的核心类，继承自 PyTorch 的 `Dataset`，负责各种插件的接入，其他功能（如数据格式转换、数据加载等）均通过插件的形式实现并接入 `DataEngine`。

`DataEngine`接受一个唯一入参：`DataArguments`的实例，，所有的元数据集信息均通过该参数配置传入，该类的定义如下：

```python
@dataclass
class DataArguments(dataset, dataset_dir, cutoff_len):
    ...

```

`dataset`: 数据集路径，支持本地或远程，当传入本地数据集文件路径时，需要满足该数据集为标准格式；否则，需要传入dataset_info.yaml来配置数据集的元信息，告诉 DataEngine 应当如何处理该数据。

`dataset_dir`: 包含数据集的文件夹的路径，非必须，当`dataset`传入绝对路径时不需要该参数，传入文件名时则需要该参数配合使用。

`cutoff_len`: 数据集的截止长度，即该数据集的最大样本数量。

---

### 2. 核心方法

#### 2.1 `get_dataset_info`
```python
def get_dataset_info(self) -> None:
    ...
```

根据 `dataset` 参数加载数据集配置，获取数据位置、数据格式、插件配置等所有数据元信息，在实例化`DataEngine`时会自动调用此方法。

#### 2.2 `load_dataset`

```python
def load_dataset(self) -> None:
    ...
```

遍历所有数据源，根据不同的数据源加载数据，在实例化`DataEngine`时会自动调用此方法。

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

#### 2.3 构建数据索引：`build_data_index()`

```python
def build_data_index(self) -> None:
    ...
```

为每个数据集创建索引列表 `[(dataset_name, sample_index), ...]`, `DataIndexPlugin`插件在此处被调用，可控制各数据集的采样频率、采样方式等，在实例化`DataEngine`时会自动调用此方法。

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

#### 2.4 转换数据样本：`_convert_data_sample()`

```python
def _convert_data_sample(self, raw_sample: dict[str, Any], dataset_name: str) -> Sample:
    ...
```

将原始数据转换为标准格式，`DataConverter`插件在此处被调用，具体调用的插件由`get_dataset_info`方法获取的converter信息指定，为空则假定数据集为标准格式，此方法由`DataEngine`的` __getitem__`方法调用。

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

---

### 3. 初始化

`DataEngine`初始化过程在实例化时自动完成，只需传入一个构建好的`DataArguments`即可，后续可通过与Python列表的形式使用该`DataEngine`。

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

### 4. 数据访问方式

实例化后的`DataEngine`支持整数索引、列表索引、以及切片等访问方式

```python
sample = data_engine[0]  # 获取第一个样本

sample = data_engine[0:10]  # 获取前 10 个样本

sample = data_engine[[0, 5, 10]]  # 获取指定索引的样本

```
