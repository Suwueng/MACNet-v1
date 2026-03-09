# Raw Data Processing 模块文档

本模块负责将星系模拟的原始输出数据（HDFRA 快照文件和日志文件）解析、处理并转换为可供机器学习模型使用的 HDF5 格式数据集。

---

## 目录

- [模块概览](#模块概览)
- [数据处理流程](#数据处理流程)
- [配置文件](#配置文件)
- [galaxy_data.py](#galaxy_datapy)
  - [GalData 类](#galdata-类)
  - [GalDataSet 类](#galdataset-类)
- [parse_log_file.py](#parse_log_filepy)
- [使用示例](#使用示例)
- [输入输出规范](#输入输出规范)

---

## 模块概览

```
raw_data_processing/
├── galaxy_data.py       # 核心数据类：GalData（单快照）和 GalDataSet（数据集管理）
├── parse_log_file.py    # 日志文件解析：支持 Fortran/普通格式的日志读取与缓存
└── README.md            # 本文档
```

**依赖项**：`numpy`, `h5py`, `pyhdf`, `pandas`, `tqdm`

---

## 数据处理流程

整个处理流水线分为三个阶段：

```
阶段 1：日志解析（parse_log_file.py）
    RawData/<Galaxy>/log_file  ──→  Data/<galaxy_name>.parquet

阶段 2：快照处理（galaxy_data.py → GalData）
    RawData/<Galaxy>/data/hdfra.xxxxx  +  Data/<galaxy_name>.parquet
        ──→  Data/<galaxy_name>/fine/00000.h5    （原始分辨率）
        ──→  Data/<galaxy_name>/coarse/00000.h5  （降采样至 8×8 网格）

阶段 3：数据集构建（galaxy_data.py → GalDataSet，由 Experiment/prepare_data.py 调用）
    Data/<galaxy_name>/coarse/*.h5  ──→  .cache/Exp*_train.pt / val.pt / test.pt
```

### 原始数据结构

```
RawData/
├── DiskGalaxy_Fiducial/
│   ├── data/
│   │   ├── hdfra.00000          # HDFRA 格式的模拟快照（HDF4-SDS）
│   │   ├── hdfra.00001
│   │   └── ...
│   └── Fiducial.log             # 包含吸积率等物理量的日志文件
├── EllipticalGalaxy/
│   ├── data/
│   │   └── hdfra.xxxxx
│   └── zmp_usr                  # 椭圆星系使用不同格式的日志
└── ...
```

### 处理后数据结构

```
Data/
├── disk_galaxy_fiducial.parquet     # 解析后的日志数据（Parquet 格式）
├── disk_galaxy_fiducial/
│   ├── fine/                        # 原始分辨率快照（HDF5）
│   │   ├── 00000.h5
│   │   └── ...
│   └── coarse/                      # 降采样快照（8×8 网格，HDF5）
│       ├── 00000.h5
│       └── ...
└── ...
```

---

## 配置文件

所有处理参数通过 `.config/` 目录下的 JSON 文件驱动：

### BaseConfig.json

定义项目路径：

| 字段           | 说明                          |
| -------------- | ----------------------------- |
| `raw_data_dir` | 原始数据根目录（`RawData/`）  |
| `data_dir`     | 处理后数据输出目录（`Data/`） |
| `results_dir`  | 实验结果目录（`Results/`）    |

### RawDataConfig.json

定义各星系的原始数据信息，按星系类型（`elliptical_galaxy` / `disk_galaxy`）分组：

| 字段          | 说明                                            |
| ------------- | ----------------------------------------------- |
| `folder_name` | 对应 `RawData/` 下的子文件夹名                  |
| `log_file`    | 日志文件路径（相对于 `folder_name`）            |
| `snapshot_dt` | 快照之间的时间间隔（Gyr）                       |
| `range`       | 快照索引范围 `[start, end)`                     |
| `offset`      | 时间偏移量（Gyr），用于校正快照到模拟时间的映射 |
| `mode`        | 可选，`"concatenate"` 表示拼接多段索引范围      |
| `ranges`      | 当 `mode="concatenate"` 时，多段索引范围列表    |

### HdfraConfig.json

定义 HDFRA 文件中变量名到物理量名称的映射关系：

| 字段             | 说明                                                              |
| ---------------- | ----------------------------------------------------------------- |
| `colnames`       | 日志文件的列名列表                                                |
| `correspondence` | HDFRA 数据集名到物理变量名的映射，例如 `"Data-Set-5" → "density"` |

**椭圆星系变量映射**：

| HDFRA 名称   | 物理量                   |
| ------------ | ------------------------ |
| `Data-Set-2` | `v1`（径向速度）         |
| `Data-Set-3` | `v2`（角向速度）         |
| `Data-Set-4` | `v3`（第三速度分量）     |
| `Data-Set-5` | `density`（密度）        |
| `Data-Set-6` | `gas_energy`（气体内能） |
| `Data-Set-8` | `dnewstar`（新星形成率） |

**盘状星系变量映射**：

| HDFRA 名称    | 物理量       |
| ------------- | ------------ |
| `Data-Set-2`  | `v1`         |
| `Data-Set-3`  | `v2`         |
| `Data-Set-4`  | `v3`         |
| `Data-Set-5`  | `density`    |
| `Data-Set-10` | `gas_energy` |
| `Data-Set-13` | `dnewstar`   |

> **注意**：`temperature` 字段由 `density` 和 `gas_energy` 自动派生计算：$T = 93 \times E_{\text{gas}} / \rho$

---

## galaxy_data.py

### GalData 类

`GalData` 是单个模拟快照的数据容器，负责读取 HDFRA 文件、设置坐标系、计算派生物理量、降采样，以及持久化为 HDF5。

#### 构造函数

```python
GalData(ndim=2, coordinate_mode="polar")
```

| 参数              | 类型  | 默认值    | 说明                                                   |
| ----------------- | ----- | --------- | ------------------------------------------------------ |
| `ndim`            | `int` | `2`       | 数据维度，支持 2 或 3                                  |
| `coordinate_mode` | `str` | `"polar"` | 坐标系模式：`"polar"` / `"p"` 或 `"cartesian"` / `"c"` |

#### 数据读取

```python
gal = GalData(ndim=2, coordinate_mode="polar")
gal.read_hdfra(
    hdfra_path="RawData/DiskGalaxy_Fiducial/data/hdfra.00100",
    time=1.25,           # 模拟时间 (Gyr)
    log_path="Data/disk_galaxy_fiducial.parquet",
    func_parse_log=ll_dg,  # 日志解析函数
    scope=0.00625,        # 时间窗口半宽，用于日志数据时间平均
)
gal.set_corr(correspondence)  # 设置变量名映射
gal.set_coord(["fakeDim2", "fakeDim1"])  # 设置坐标轴
```

#### 属性（Properties）

**快照数据**：

| 属性             | 返回类型     | 说明                                       |
| ---------------- | ------------ | ------------------------------------------ |
| `snapshot`       | `dict`       | 所有物理量的字典，键为变量名，值为 ndarray |
| `snapshot_array` | `np.ndarray` | 所有物理量堆叠为数组                       |

**坐标相关**：

| 属性                | 返回类型            | 说明                                   |
| ------------------- | ------------------- | -------------------------------------- |
| `raw_axes`          | `list`              | 各坐标轴的一维数组列表                 |
| `broadcast_coords`  | `np.ndarray`        | 广播后的坐标数组（用于索引）           |
| `coordinate`        | `tuple[np.ndarray]` | `np.meshgrid` 生成的坐标网格           |
| `r`, `theta`, `phi` | `np.ndarray`        | 极坐标分量（自动根据坐标模式转换）     |
| `x`, `y`, `z`       | `np.ndarray`        | 笛卡尔坐标分量（自动根据坐标模式转换） |

**物理量**：

| 属性          | 返回类型     | 说明                                      |
| ------------- | ------------ | ----------------------------------------- |
| `grid_volume` | `np.ndarray` | 每个网格单元的体积                        |
| `gas_mass`    | `np.ndarray` | 每个网格单元的气体质量 ($\rho \times dV$) |
| `mbh`         | `float`      | 黑洞质量（来自日志数据）                  |
| `mdot_macer`  | `float`      | MACER 吸积率（来自日志/计算）             |
| `mdot_edd`    | `float`      | 爱丁顿吸积率（来自日志/计算）             |

#### 计算方法

##### `mdot_bondi(r_acc=1.0, G=112.0, gamma=5/3)`

计算 Bondi 吸积率。在吸积半径 `r_acc`（kpc）内对气体密度和声速做质量加权平均，然后代入 Bondi 公式：

$$\dot{M}_{\text{Bondi}} = 4\pi \lambda_c \frac{G^2 M_{\text{BH}}^2 \rho_\infty}{c_{s,\infty}^3}$$

其中 $\lambda_c$ 是依赖绝热指数 $\gamma$ 的无量纲常数。

#### 数据操作

##### `rescale(new_shape, weights=None)`

将快照数据降采样到指定形状。

- `new_shape`：目标形状，例如 `(8, 8)`
- `weights`：各变量的权重字典。未指定的变量默认使用质量加权平均；密度变量应使用体积加权（`grid_volume`）
- 返回新的 `GalData` 实例

```python
gal_coarse = gal.rescale(new_shape=(8, 8), weights={"density": gal.grid_volume})
```

##### `set_corr(corr)`

设置 HDFRA 数据集名到物理变量名的映射关系。

##### `set_coord(coords)`

设置坐标轴名称列表（HDFRA 文件中的维度名）。对于 2D 极坐标数据，通常为 `["fakeDim2", "fakeDim1"]`。

#### 持久化

##### `save_h5(file_path)`

将 `GalData` 实例保存为 HDF5 文件。文件内部结构：

```
├── snapshot/           # 物理量数据
│   ├── density
│   ├── gas_energy
│   ├── temperature
│   ├── v1, v2, v3
│   └── dnewstar
├── coordinates/        # 坐标轴
│   ├── axis_0, axis_1
│   └── attrs: coordinate_mode, ndim
├── log_data/           # 日志数据
│   ├── mbh, mdot_macer, mdot_edd, ...
├── correlation/        # 变量名映射关系
└── attrs: time         # 模拟时间
```

##### `load_h5(file_path)`

从 HDF5 文件恢复 `GalData` 实例。

---

### GalDataSet 类

`GalDataSet` 用于管理多个 `GalData` 快照的数据集，提供数据加载、清洗、划分、标准化等机器学习数据准备功能。

#### 数据加载

```python
dataset = GalDataSet()
dataset.load_data([
    ("Data/disk_galaxy_fiducial/coarse", "disk_galaxy_fiducial"),
    ("Data/disk_galaxy_fiducial_4/coarse", "disk_galaxy_fiducial_4"),
])
```

参数为 `(folder_path, group_label)` 元组的列表。`group_label` 用于标记样本所属的星系组，后续可按组做分层采样。

#### 属性

| 属性         | 返回类型     | 说明                                                                 |
| ------------ | ------------ | -------------------------------------------------------------------- |
| `x`          | `np.ndarray` | 输入特征，形状 `(N, C, H, W)`，C 为物理量通道数                      |
| `y`          | `np.ndarray` | 目标变量：$\log_{10}(\dot{M}_{\text{MACER}} / \dot{M}_{\text{Edd}})$ |
| `y_baseline` | `np.ndarray` | 基线预测：$\log_{10}(\dot{M}_{\text{Bondi}} / \dot{M}_{\text{Edd}})$ |
| `mbh`        | `np.ndarray` | 黑洞质量数组                                                         |
| `coordinate` | `np.ndarray` | 坐标网格数组                                                         |
| `time`       | `np.ndarray` | 模拟时间数组                                                         |
| `groups`     | `np.ndarray` | 组标签数组                                                           |
| `n_groups`   | `int`        | 组数量                                                               |
| `mdot_macer` | `np.ndarray` | MACER 吸积率数组                                                     |
| `mdot_edd`   | `np.ndarray` | 爱丁顿吸积率数组                                                     |

#### 数据清洗

##### `drop_invalid(require_positive_targets=True, require_positive_mbh=True, require_positive_bondi=True)`

删除包含 NaN/Inf 或非正值的样本。

##### `filter(threshold)`

过滤掉目标变量 `y` 低于阈值的样本。

#### 数据增强

##### `mirror_data()`

沿第一个坐标轴镜像翻转数据，用于数据增强。

##### `balance_groups(method="oversample")`

通过过采样（`"oversample"`）或欠采样（`"undersample"`）平衡各组样本数量。

#### 数据划分

```python
train_set, val_set, test_set = dataset.split(
    train_size=0.6,
    validation_size=0.2,
    stratify="groups",   # 按组分层采样；也可用 "y" 按目标值分箱，或 None 随机划分
    random_state=42,
)
```

| 参数              | 说明                                                                            |
| ----------------- | ------------------------------------------------------------------------------- |
| `train_size`      | 训练集比例 (0, 1)                                                               |
| `validation_size` | 验证集比例，设为 0 则只返回训练集和测试集                                       |
| `stratify`        | 分层策略：`None`（随机）/ `"groups"`（按组）/ `"y"`（按目标值分箱）/ 自定义数组 |
| `n_bins`          | 当 `stratify="y"` 时的分箱数量，默认 10                                         |

#### 标准化

```python
# 首次调用：计算并返回均值和标准差
x_normed, mean, std = dataset.standardize(log_transform=True)

# 后续调用：使用已有统计量
x_normed = other_dataset.standardize(mean=mean, std=std, log_transform=True)
```

标准化流程：
1. 对非负通道执行 `log1p` 变换（可选，默认开启）
2. 沿样本维度和空间维度 `(0, 2, 3)` 计算均值和标准差
3. 执行 Z-score 标准化：$(x - \mu) / \sigma$

#### 运算符重载

```python
combined = dataset_a + dataset_b  # 合并两个数据集
len(dataset)                       # 返回样本数量
```

---

## parse_log_file.py

日志文件解析模块，负责将模拟产生的原始日志解析为结构化的 DataFrame。

### 核心函数

#### `parse_logfile(path, colnames, save_path=None, force_parse=False)`

解析日志文件的主入口函数。

| 参数          | 类型          | 说明                                    |
| ------------- | ------------- | --------------------------------------- |
| `path`        | `str`         | 日志文件路径                            |
| `colnames`    | `list[str]`   | 列名列表（来自 `HdfraConfig.json`）     |
| `save_path`   | `str \| None` | 可选 Parquet 缓存路径，已存在则直接读取 |
| `force_parse` | `bool`        | 为 `True` 时强制重新解析，忽略缓存      |

特性：
- **自动格式检测**：通过 `detect_fortran_style()` 检测日志是否使用 Fortran 风格科学计数法（如 `1.0D+03` 或 `1.0+03`），自动选择对应的解析器
- **Parquet 缓存**：解析结果可缓存为 Parquet 文件，后续调用直接读取，避免重复解析

#### `mean_around_time(df, time, scope, time_col="time")`

在时间窗口 `[time - scope, time + scope]` 内对 DataFrame 做均值聚合。用于将日志数据的高时间分辨率信息平滑到快照的时间精度。

### 物理量计算函数

#### `ll_eg(path, time, scope)` — 椭圆星系

从 Parquet 读取日志数据，在时间窗口内取平均，并计算：

$$\dot{M}_{\text{MACER}} = \dot{M}_{\text{Edd}} \times \text{mdot\_ratio}$$

#### `ll_dg(path, time, scope)` — 盘状星系

从 Parquet 读取日志数据，在时间窗口内取平均，并计算爱丁顿吸积率：

$$\dot{M}_{\text{Edd}} = \frac{4\pi G M_{\text{BH}} m_p}{\eta \, c \, \sigma_T}$$

其中使用的模拟单位常数：
- $G = 112$（$\text{kpc}^3 / (2.5 \times 10^7 M_\odot \cdot \text{Gyr}^2)$）
- $m_p = 3.365 \times 10^{-65}$
- $\eta = 0.1$（辐射效率）
- $c = 3.07 \times 10^5$
- $\sigma_T = 6.99 \times 10^{-68}$

### 辅助函数

#### `load_config(config_dir=".config")`

加载 `.config/` 目录下所有 JSON 配置文件，返回以文件名（不含后缀）为键的字典。

---

## 使用示例

### 完整处理流程

```python
# 1. 解析日志文件
from raw_data_processing.parse_log_file import parse_logfile, load_config

configs = load_config(".config")
parse_logfile(
    path="RawData/DiskGalaxy_Fiducial/Fiducial.log",
    colnames=configs["HdfraConfig"]["disk_galaxy"]["colnames"],
    save_path="Data/disk_galaxy_fiducial.parquet",
)

# 2. 处理单个快照
from raw_data_processing.galaxy_data import GalData
from raw_data_processing.parse_log_file import ll_dg

gal = GalData(ndim=2, coordinate_mode="polar")
gal.read_hdfra(
    hdfra_path="RawData/DiskGalaxy_Fiducial/data/hdfra.00100",
    time=1.25,
    log_path="Data/disk_galaxy_fiducial.parquet",
    func_parse_log=ll_dg,
    scope=0.00625,
)
corr = configs["HdfraConfig"]["disk_galaxy"]["correspondence"]
gal.set_corr(corr).set_coord(["fakeDim2", "fakeDim1"])

# 保存原始分辨率和降采样版本
gal.save_h5("Data/disk_galaxy_fiducial/fine/00100.h5")
gal_coarse = gal.rescale((8, 8), weights={"density": gal.grid_volume})
gal_coarse.save_h5("Data/disk_galaxy_fiducial/coarse/00100.h5")

# 3. 构建数据集
from raw_data_processing.galaxy_data import GalDataSet

dataset = GalDataSet()
dataset.load_data([
    ("Data/disk_galaxy_fiducial/coarse", "disk_galaxy_fiducial"),
    ("Data/disk_galaxy_fiducial_4/coarse", "disk_galaxy_fiducial_4"),
])
dataset.drop_invalid()

train, val, test = dataset.split(
    train_size=0.6, validation_size=0.2, stratify="groups"
)
x_train, mean, std = train.standardize()
x_val = val.standardize(mean=mean, std=std)
```

### 批量处理（命令行）

```bash
# 步骤 1：解析所有日志文件
python -m raw_data_processing.parse_log_file

# 步骤 2：处理所有 HDFRA 快照
python -m raw_data_processing.galaxy_data

# 步骤 3：构建实验数据集（生成 .cache/Exp*_*.pt）
python -m Experiment.prepare_data
```

---

## 输入输出规范

### 输入文件格式

| 文件类型            | 格式     | 说明                                            |
| ------------------- | -------- | ----------------------------------------------- |
| `hdfra.xxxxx`       | HDF4-SDS | 模拟快照，包含多个 `Data-Set-N` 数据集          |
| `*.log` / `zmp_usr` | 文本     | 模拟日志，空格分隔，可能使用 Fortran 科学计数法 |

### 输出文件格式

| 文件类型    | 格式           | 说明                                                    |
| ----------- | -------------- | ------------------------------------------------------- |
| `*.parquet` | Apache Parquet | 解析后的日志数据，通过 pandas 读写                      |
| `*.h5`      | HDF5           | 处理后的快照数据，包含 snapshot/coordinates/log_data 组 |
| `*.pt`      | PyTorch        | 最终训练数据，包含标准化后的特征、标签等张量            |

### HDF5 文件内部结构

```
root
├── [attr] time: float                        # 模拟时间 (Gyr)
├── snapshot/                                  # 物理量数据组
│   ├── density: float[H, W]                  # 气体密度
│   ├── gas_energy: float[H, W]               # 气体内能
│   ├── temperature: float[H, W]              # 温度（派生量）
│   ├── v1: float[H, W]                       # 径向速度
│   ├── v2: float[H, W]                       # 角向速度
│   ├── v3: float[H, W]                       # 第三速度分量
│   └── dnewstar: float[H, W]                 # 新星形成率
├── coordinates/
│   ├── [attr] coordinate_mode: str            # "polar" 或 "cartesian"
│   ├── [attr] ndim: int                       # 维度数
│   ├── axis_0: float[H]                      # 第一坐标轴（r 或 x）
│   └── axis_1: float[W]                      # 第二坐标轴（θ 或 y）
├── log_data/
│   ├── mbh: float                             # 黑洞质量
│   ├── mdot_macer: float                      # MACER 吸积率
│   ├── mdot_edd: float                        # 爱丁顿吸积率
│   └── ...                                    # 其他日志量
└── correlation/
    └── [attrs] 0_orig, 0_new, ...             # 变量名映射
```
