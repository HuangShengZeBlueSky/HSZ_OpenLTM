# HSZ_OpenLTM

基于 OpenLTM 的时序大模型预测与异常检测实现，包含 Timer-XL 系列与 AutoTimes 等模型，支持训练、批量异常检测与一次性参数寻优。

An OpenLTM-based time-series foundation model implementation for forecasting and anomaly detection, covering Timer-XL and AutoTimes variants with training, batch anomaly detection, and one-pass parameter search.

## 适用任务与模型 | Tasks & Models

### 任务模式 | Task modes
- forecast：训练与常规测试
- forecast_test_all：遍历目录下全部测试 CSV
- forecast_test_all2：校准后批量异常检测
- forecast_test_all3：一次性参数寻优并输出 Top 结果

### 支持模型 | Supported models
- autotimes
- gpt4ts
- moirai
- moment
- time_llm
- timer
- timer_xl
- ttm

### 数据集分类（标准基准 vs BJTU 专用）| Datasets (Benchmark vs BJTU)
- 标准基准别名映射（多变量基准类）：ETTh1、ETTh2、ETTm1、ETTm2、ECL、Electricity、Weather、Traffic
- 标准基准注册名：UnivariateDatasetBenchmark、MultivariateDatasetBenchmark、Global_Temp、Global_Wind、Era5_Pretrain、Era5_Pretrain_Test、Utsd、Utsd_Npy
- BJTU 专用：BJTU（异常检测专用加载逻辑）

来源： [data_provider/data_factory.py](data_provider/data_factory.py) 与 [data_provider/data_loader.py](data_provider/data_loader.py)

## 安装与环境 | Installation

建议使用 Python 3.10+。依赖见 [requirements.txt](requirements.txt)。

    pip install -r requirements.txt

LLM 本地权重目录由参数 `--local_path` 指定，默认值见参数表。

## 最小可运行配置 | Minimal runnable configs

### 1) 训练（forecast）

    python run.py \
      --task_name forecast --is_training 1 --model_id demo --model autotimes \
      --data BJTU --root_path <TRAIN_ROOT> --data_path <TRAIN_CSV> \
      --n_vars 3 --seq_len 2880 --input_token_len 96 --output_token_len 96 \
      --batch_size 32 --train_epochs 10 --gpu 0 \
      --llm_model LLAMA --local_path <LLM_WEIGHTS_DIR>

### 2) 校准 + 批量异常检测（forecast_test_all2）

    python run.py \
      --task_name forecast_test_all2 --is_training 0 --model_id demo --model autotimes \
      --data BJTU --root_path <TEST_ROOT> --test_dir <CHECKPOINT_DIR_NAME> \
      --test_file_name checkpoint.pth --n_vars 3 \
      --seq_len 2880 --input_token_len 96 --output_token_len 96 \
      --test_seq_len 2880 --test_pred_len 96 --batch_size 32 --gpu 0 \
      --percentile 99 --vote_rate 1.0 --llm_model LLAMA --local_path <LLM_WEIGHTS_DIR>

### 3) 一次性参数寻优（forecast_test_all3）

    python run.py \
      --task_name forecast_test_all3 --is_training 0 --model_id demo --model autotimes \
      --data BJTU --root_path <TEST_ROOT> --test_dir <CHECKPOINT_DIR_NAME> \
      --test_file_name checkpoint.pth --n_vars 3 \
      --seq_len 2880 --input_token_len 96 --output_token_len 96 \
      --test_seq_len 2880 --test_pred_len 96 --batch_size 32 --gpu 0 \
      --llm_model LLAMA --local_path <LLM_WEIGHTS_DIR>

说明：`--vote_rate` 在命令行以百分比输入，内部会除以 100 转为比例。

## 推荐默认参数组合 | Recommended presets

### 训练（forecast）起点建议
- 基于默认值 + BJTU 常用时序长度：`--seq_len 2880`、`--input_token_len 96`、`--output_token_len 96`
- 通道数：`--n_vars` 设为数据维度
- 模型：`--model autotimes`，LLM 权重目录由 `--local_path` 指定

### 异常检测（forecast_test_all2）起点建议
- `--percentile 99` 与 `--vote_rate 1.0` 作为初始阈值
- `--test_dir` 指向训练产出的 checkpoint 目录名

## 参数表 | Parameter tables

### 表 1：基础与数据参数 | Basic & data parameters

| 参数 | 默认值 | 说明（中文） | Description (EN) | 来源 |
| --- | --- | --- | --- | --- |
| `task_name` | forecast | 任务模式 | Task mode | [run.py](run.py) |
| `is_training` | 1 | 1 训练，0 仅测试 | 1=train, 0=test only | [run.py](run.py) |
| `model_id` | test | 实验标识 | Experiment id | [run.py](run.py) |
| `model` | timer_xl | 模型名称 | Model name | [run.py](run.py) |
| `seed` | 2021 | 随机种子 | Random seed | [run.py](run.py) |
| `data` | ETTh1 | 数据集标识 | Dataset key | [run.py](run.py) |
| `root_path` | ./dataset/ETT-small/ | 数据根目录 | Dataset root | [run.py](run.py) |
| `data_path` | ETTh1.csv | 数据文件 | Data file | [run.py](run.py) |
| `checkpoints` | ./checkpoints/ | 模型保存目录 | Checkpoint directory | [run.py](run.py) |
| `test_flag` | T | 测试域标识 | Test domain flag | [run.py](run.py) |
| `seq_len` | 672 | 输入序列长度 | Input sequence length | [run.py](run.py) |
| `input_token_len` | 576 | 输入 token 长度 | Input token length | [run.py](run.py) |
| `output_token_len` | 96 | 输出 token 长度 | Output token length | [run.py](run.py) |
| `test_seq_len` | 672 | 测试输入长度 | Test input length | [run.py](run.py) |
| `test_pred_len` | 96 | 测试预测长度 | Test prediction length | [run.py](run.py) |
| `dropout` | 0.1 | Dropout 概率 | Dropout ratio | [run.py](run.py) |
| `e_layers` | 1 | 编码层数 | Encoder layers | [run.py](run.py) |
| `d_model` | 512 | 模型维度 | Model width | [run.py](run.py) |
| `n_heads` | 8 | 注意力头数 | Attention heads | [run.py](run.py) |
| `d_ff` | 2048 | FFN 维度 | FFN dimension | [run.py](run.py) |
| `activation` | relu | 激活函数 | Activation | [run.py](run.py) |
| `covariate` | False | 是否使用协变量 | Use covariates | [run.py](run.py) |
| `node_num` | 100 | 节点数量 | Node count | [run.py](run.py) |
| `node_list` | 23,37,40 | 节点划分列表 | Node list | [run.py](run.py) |
| `use_norm` | False | 是否使用归一化 | Use normalization | [run.py](run.py) |
| `nonautoregressive` | False | 是否非自回归 | Non-autoregressive | [run.py](run.py) |
| `test_dir` | ./test | 测试模型目录名 | Checkpoint folder name | [run.py](run.py) |
| `test_file_name` | checkpoint.pth | 测试模型文件 | Checkpoint file | [run.py](run.py) |
| `output_attention` | False | 输出注意力 | Output attention | [run.py](run.py) |
| `visualize` | False | 输出可视化 | Save plots | [run.py](run.py) |
| `flash_attention` | False | 使用 FlashAttention | Use flash attention | [run.py](run.py) |
| `adaptation` | False | 是否加载预训练权重 | Enable adaptation | [run.py](run.py) |
| `pretrain_model_path` | pretrain_model.pth | 预训练模型路径 | Pretrained weight path | [run.py](run.py) |
| `subset_rand_ratio` | 1 | 少样本比例 | Few-shot ratio | [run.py](run.py) |
| `n_vars` | 7 | 变量/通道数 | Variable count | [run.py](run.py) |
| `factor` | 2 | 隐层扩展倍数 | Expansion factor | [run.py](run.py) |
| `mode` | mix_channel | TTM 模式 | TTM mode | [run.py](run.py) |
| `AP_levels` | 0 | Attention patching 层数 | AP levels | [run.py](run.py) |
| `use_decoder` | True | TTM decoder 开关 | Use decoder | [run.py](run.py) |
| `d_mode` | common_channel | TTM 通道模式 | TTM channel mode | [run.py](run.py) |
| `layers` | 8 | TTM 层数 | TTM layers | [run.py](run.py) |
| `hidden_dim` | 16 | TTM 隐层维度 | TTM hidden dim | [run.py](run.py) |

### 表 2：训练与优化参数 | Training & optimization

| 参数 | 默认值 | 说明（中文） | Description (EN) | 来源 |
| --- | --- | --- | --- | --- |
| `num_workers` | 10 | 数据加载线程数 | DataLoader workers | [run.py](run.py) |
| `itr` | 1 | 重复实验次数 | Repeated runs | [run.py](run.py) |
| `train_epochs` | 10 | 训练轮数 | Training epochs | [run.py](run.py) |
| `batch_size` | 32 | 批大小 | Batch size | [run.py](run.py) |
| `patience` | 3 | 早停耐心 | Early stopping patience | [run.py](run.py) |
| `learning_rate` | 0.0001 | 学习率 | Learning rate | [run.py](run.py) |
| `des` | test | 实验描述 | Experiment tag | [run.py](run.py) |
| `loss` | MSE | 损失函数 | Loss name | [run.py](run.py) |
| `lradj` | type1 | 学习率调度 | LR schedule | [run.py](run.py) |
| `cosine` | False | 使用 CosineAnnealing | Use cosine annealing | [run.py](run.py) |
| `tmax` | 10 | CosineAnnealing 的 T_max | T_max for cosine | [run.py](run.py) |
| `weight_decay` | 0 | 权重衰减 | Weight decay | [run.py](run.py) |
| `valid_last` | False | 验证时只用尾段 | Validate on tail | [run.py](run.py) |
| `last_token` | False | 仅使用最后通道 | Use last token | [run.py](run.py) |

### 表 3：异常检测参数 | Anomaly detection

| 参数 | 默认值 | 说明（中文） | Description (EN) | 来源 |
| --- | --- | --- | --- | --- |
| `percentile` | 99 | 点级阈值分位数 | Point threshold percentile | [run.py](run.py), [exp/exp_AnomalyDetection.py](exp/exp_AnomalyDetection.py) |
| `vote_rate` | 1.0 | 样本异常投票率（百分比） | Vote rate in percent | [run.py](run.py), [exp/exp_AnomalyDetection.py](exp/exp_AnomalyDetection.py) |
| `method` | mad | 异常判别方法 | Detection method | [exp/testall_exp_anomaly_detection.py](exp/testall_exp_anomaly_detection.py) |
| `alpha` | 3.5 | MAD 阈值系数 | MAD threshold factor | [exp/testall_exp_anomaly_detection.py](exp/testall_exp_anomaly_detection.py) |
| `q` | 0.995 | Quantile 阈值 | Quantile threshold | [exp/testall_exp_anomaly_detection.py](exp/testall_exp_anomaly_detection.py) |
| `only_idx` | -1 | 仅在指定通道检测 | Detect only one channel | [exp/testall_exp_anomaly_detection.py](exp/testall_exp_anomaly_detection.py) |

### 表 4：LLM 与硬件参数 | LLM & hardware

| 参数 | 默认值 | 说明（中文） | Description (EN) | 来源 |
| --- | --- | --- | --- | --- |
| `gpu` | 0 | GPU 编号 | GPU id | [run.py](run.py) |
| `ddp` | False | 分布式训练 | Distributed data parallel | [run.py](run.py) |
| `dp` | False | 数据并行 | Data parallel | [run.py](run.py) |
| `devices` | 0,1,2,3 | 多卡设备列表 | Multi-GPU device ids | [run.py](run.py) |
| `gpt_layers` | 6 | GPT 层数 | GPT layers | [run.py](run.py) |
| `patch_size` | 16 | Patch 大小 | Patch size | [run.py](run.py) |
| `kernel_size` | 25 | 卷积核大小 | Kernel size | [run.py](run.py) |
| `stride` | 8 | Patch stride | Patch stride | [run.py](run.py) |
| `ts_vocab_size` | 1000 | Time-LLM 原型词表 | Time-LLM prototype vocab | [run.py](run.py) |
| `domain_des` | Electricity Transformer Temperature 描述 | 领域描述文本 | Domain description | [run.py](run.py) |
| `llm_model` | LLAMA | LLM 类型 | LLM backbone | [run.py](run.py) |
| `llm_layers` | 6 | LLM 层数 | LLM layers | [run.py](run.py) |
| `local_path` | /home/ubuntu/hsz/models | 本地 LLM 权重根目录 | LLM weights root | [run.py](run.py) |

## 缺失但需存在的参数 | Required but missing args

下列参数在标准基准数据集分支中会被读取，但未在 [run.py](run.py) 中显式定义。若使用标准基准数据集，请在外部补齐或在代码中加入默认值。

The following args are required by the benchmark datasets but are not defined in [run.py](run.py). Provide them externally or add defaults when using benchmark datasets.

- `embed`：时间编码方式
- `label_len`：label 序列长度
- `pred_len`：预测序列长度
- `features`：特征模式（M/S/MS）
- `target`：目标列名
- `seasonal_patterns`：季节性模式
- `freq`：时间频率

来源： [data_provider/data_factory.py](data_provider/data_factory.py)

## 结果文件解释 | Output artifacts

- 训练检查点：checkpoints/<setting>/checkpoint.pth
- 训练评估指标：result_long_term_forecast.txt
- 批量异常检测汇总：test_results/<setting>/result_summary.txt
- 异常标记数组：test_results/<setting>/<csv>_flags.npy
- 自动寻优最佳结果：test_results/<setting>/best_results.npy

对应目录与文件： [checkpoints/](checkpoints/), [test_results/](test_results/), [result_long_term_forecast.txt](result_long_term_forecast.txt)

## 项目结构 | Project structure

- 入口脚本：[run.py](run.py)
- 实验逻辑：[exp/](exp/)
- 数据加载：[data_provider/](data_provider/)
- 模型实现：[models/](models/)

## 注意事项 | Notes

- `--vote_rate` 为百分比输入，内部会除以 100 转为比例。
- BJTU 数据集读取逻辑为无表头 CSV，训练/验证集采用 80/20 切分。
# 基于OpenLTM的时序大模型异常检测功能：包含timerXL系列的异常检测功能实现，以及autotimes功能的实现。

使用环境 conda activate openltm 

192.168.65.35机器上

```
(openltm) lqy@DGXUSER1:/raid/hsz$ conda list
# packages in environment at /home/dgxuser1/miniconda3/envs/openltm:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
anyio                     4.10.0                   pypi_0    pypi
aom                       3.6.0                h6a678d5_0  
argon2-cffi               25.1.0                   pypi_0    pypi
argon2-cffi-bindings      25.1.0                   pypi_0    pypi
arrow                     1.3.0                    pypi_0    pypi
asttokens                 3.0.0                    pypi_0    pypi
async-lru                 2.0.5                    pypi_0    pypi
attrs                     25.3.0                   pypi_0    pypi
babel                     2.17.0                   pypi_0    pypi
beautifulsoup4            4.13.4                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    6.2.0                    pypi_0    pypi
brotlipy                  0.7.0           py310h7f8727e_1002  
bzip2                     1.0.8                h5eee18b_6  
ca-certificates           2025.2.25            h06a4308_0  
cairo                     1.16.0               he5ede1b_6  
certifi                   2025.7.14       py310h06a4308_0  
cffi                      1.17.1          py310h1fdaa30_1  
chardet                   4.0.0           py310h06a4308_1003  
charset-normalizer        3.4.2                    pypi_0    pypi
click                     8.3.1                    pypi_0    pypi
cmake                     3.25.0                   pypi_0    pypi
comm                      0.2.2                    pypi_0    pypi
conda-pack                0.8.1                    pypi_0    pypi
contourpy                 1.3.2                    pypi_0    pypi
cryptography              44.0.1          py310h7825ff9_0  
cuda-cudart               12.1.105                      0    nvidia
cuda-cupti                12.1.105                      0    nvidia
cuda-libraries            12.1.0                        0    nvidia
cuda-nvrtc                12.1.105                      0    nvidia
cuda-nvtx                 12.1.105                      0    nvidia
cuda-opencl               12.4.127                      0    nvidia
cuda-runtime              12.1.0                        0    nvidia
cudatoolkit               11.3.1               h2bc3f7f_2  
cycler                    0.12.1                   pypi_0    pypi
dav1d                     1.2.1                h5eee18b_0  
debugpy                   1.8.14                   pypi_0    pypi
decorator                 5.2.1                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
einops                    0.8.1                    pypi_0    pypi
exceptiongroup            1.3.0                    pypi_0    pypi
executing                 2.2.0                    pypi_0    pypi
expat                     2.7.1                h6a678d5_0  
fastjsonschema            2.21.1                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
filelock                  3.17.0          py310h06a4308_0  
fontconfig                2.14.1               h55d465d_3  
fonttools                 4.59.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.13.3               h4a9f257_0  
fribidi                   1.0.10               h7b6447c_0  
fsspec                    2025.7.0                 pypi_0    pypi
gmp                       6.3.0                h6a678d5_0  
gmpy2                     2.2.1           py310h5eee18b_0  
gnutls                    3.6.15               he1e5248_0  
graphite2                 1.3.14               h295c915_1  
h11                       0.16.0                   pypi_0    pypi
harfbuzz                  10.2.0               hdfddeaa_1  
hf-xet                    1.2.0                    pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
huggingface-hub           0.34.3                   pypi_0    pypi
icu                       73.1                 h6a678d5_0  
idna                      2.10               pyhd3eb1b0_0  
intel-openmp              2025.0.0          h06a4308_1171  
ipykernel                 6.29.5                   pypi_0    pypi
ipython                   8.37.0                   pypi_0    pypi
ipywidgets                8.1.7                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.19.2          py310h06a4308_0  
jinja2                    3.1.6           py310h06a4308_0  
joblib                    1.5.1                    pypi_0    pypi
jpeg                      9e                   h5eee18b_3  
json5                     0.12.0                   pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.25.0                   pypi_0    pypi
jsonschema-specifications 2025.4.1                 pypi_0    pypi
jupyter                   1.1.1                    pypi_0    pypi
jupyter-client            8.6.3                    pypi_0    pypi
jupyter-console           6.6.3                    pypi_0    pypi
jupyter-core              5.8.1                    pypi_0    pypi
jupyter-events            0.12.0                   pypi_0    pypi
jupyter-lsp               2.2.6                    pypi_0    pypi
jupyter-server            2.16.0                   pypi_0    pypi
jupyter-server-terminals  0.5.3                    pypi_0    pypi
jupyterlab                4.4.5                    pypi_0    pypi
jupyterlab-pygments       0.3.0                    pypi_0    pypi
jupyterlab-server         2.27.3                   pypi_0    pypi
jupyterlab-widgets        3.0.15                   pypi_0    pypi
kiwisolver                1.4.8                    pypi_0    pypi
lame                      3.100                h7b6447c_0  
lark                      1.2.2                    pypi_0    pypi
lcms2                     2.16                 h92b89f2_1  
ld_impl_linux-64          2.40                 h12ee557_0  
lerc                      4.0.0                h6a678d5_0  
libavif                   1.1.1                h5eee18b_0  
libcublas                 12.1.0.26                     0    nvidia
libcufft                  11.0.2.4                      0    nvidia
libcufile                 1.9.1.3                       0    nvidia
libcurand                 10.3.5.147                    0    nvidia
libcusolver               11.4.4.55                     0    nvidia
libcusparse               12.0.2.55                     0    nvidia
libdeflate                1.22                 h5eee18b_0  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libglib                   2.84.2               h37c7471_0  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h5eee18b_3  
libidn2                   2.3.4                h5eee18b_0  
libjpeg-turbo             2.0.0                h9bf148f_0    pytorch
libnpp                    12.0.2.50                     0    nvidia
libnvjitlink              12.1.105                      0    nvidia
libnvjpeg                 12.1.1.14                     0    nvidia
libpng                    1.6.39               h5eee18b_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtasn1                  4.19.0               h5eee18b_0  
libtiff                   4.7.0                hde9077f_0  
libunistring              0.9.10               h27cfd23_0  
libuuid                   1.41.5               h5eee18b_0  
libwebp-base              1.3.2                h5eee18b_1  
libxcb                    1.17.0               h9b100fa_0  
libxml2                   2.13.8               hfdd30dd_0  
lit                       15.0.7                   pypi_0    pypi
llvm-openmp               14.0.6               h9e868ea_0  
lz4-c                     1.9.4                h6a678d5_1  
markupsafe                3.0.2           py310h5eee18b_0  
matplotlib                3.10.5                   pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
mistune                   3.1.3                    pypi_0    pypi
mkl                       2025.0.0           hacee8c2_941  
modelscope                1.32.0                   pypi_0    pypi
mpc                       1.3.1                h5eee18b_0  
mpfr                      4.2.1                h5eee18b_0  
mpmath                    1.3.0           py310h06a4308_0  
nbclient                  0.10.2                   pypi_0    pypi
nbconvert                 7.16.6                   pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.6.0                    pypi_0    pypi
nettle                    3.7.3                hbbd107a_1  
networkx                  3.4.2           py310h06a4308_0  
notebook                  7.4.5                    pypi_0    pypi
notebook-shim             0.2.4                    pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openjpeg                  2.5.2                h0d4d230_1  
openssl                   3.0.17               h5eee18b_0  
overrides                 7.7.0                    pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
pandas                    2.3.1                    pypi_0    pypi
pandocfilters             1.5.1                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pcre2                     10.42                hebb0a14_1  
pexpect                   4.9.0                    pypi_0    pypi
pillow                    11.3.0          py310hb1c3d2d_0  
pip                       25.1               pyhc872135_2  
pixman                    0.40.0               h7f8727e_1  
platformdirs              4.3.8                    pypi_0    pypi
prometheus-client         0.22.1                   pypi_0    pypi
prompt-toolkit            3.0.51                   pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1  
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.19.2                   pypi_0    pypi
pyopenssl                 25.0.0          py310h06a4308_0  
pyparsing                 3.2.3                    pypi_0    pypi
pysocks                   1.7.1           py310h06a4308_0  
python                    3.10.18              h1a3bd86_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
python-json-logger        3.3.0                    pypi_0    pypi
pytorch-cuda              12.1                 ha16c6d3_6    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.2           py310h5eee18b_0  
pyzmq                     27.0.0                   pypi_0    pypi
readline                  8.2                  h5eee18b_0  
referencing               0.36.2                   pypi_0    pypi
regex                     2025.7.34                pypi_0    pypi
requests                  2.32.4                   pypi_0    pypi
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
rfc3987-syntax            1.1.0                    pypi_0    pypi
rpds-py                   0.26.0                   pypi_0    pypi
safetensors               0.5.3                    pypi_0    pypi
scikit-learn              1.7.1                    pypi_0    pypi
scipy                     1.15.3                   pypi_0    pypi
send2trash                1.8.3                    pypi_0    pypi
setuptools                78.1.1          py310h06a4308_0  
shellingham               1.5.4                    pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
soupsieve                 2.7                      pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
stack-data                0.6.3                    pypi_0    pypi
sympy                     1.13.3          py310h06a4308_1  
tbb                       2022.0.0             hdb19cb5_0  
tbb-devel                 2022.0.0             hdb19cb5_0  
terminado                 0.18.1                   pypi_0    pypi
threadpoolctl             3.6.0                    pypi_0    pypi
tinycss2                  1.4.0                    pypi_0    pypi
tk                        8.6.14               h993c535_1  
tokenizers                0.21.4                   pypi_0    pypi
tomli                     2.2.1                    pypi_0    pypi
torch                     2.1.2+cu121              pypi_0    pypi
torchaudio                2.1.2+cu121              pypi_0    pypi
torchvision               0.16.2+cu121             pypi_0    pypi
tornado                   6.5.1                    pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
transformers              4.55.0                   pypi_0    pypi
triton                    2.1.0                    pypi_0    pypi
typer-slim                0.20.0                   pypi_0    pypi
types-python-dateutil     2.9.0.20250708           pypi_0    pypi
typing-extensions         4.14.1                   pypi_0    pypi
typing_extensions         4.12.2          py310h06a4308_0  
tzdata                    2025.2                   pypi_0    pypi
uri-template              1.3.0                    pypi_0    pypi
urllib3                   1.26.4             pyhd3eb1b0_0  
wcwidth                   0.2.13                   pypi_0    pypi
webcolors                 24.11.1                  pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.8.0                    pypi_0    pypi
wheel                     0.45.1          py310h06a4308_0  
widgetsnbextension        4.0.14                   pypi_0    pypi
xorg-libx11               1.8.12               h9b100fa_1  
xorg-libxau               1.0.12               h9b100fa_0  
xorg-libxdmcp             1.1.5                h9b100fa_0  
xorg-libxext              1.3.6                h9b100fa_0  
xorg-libxrender           0.9.12               h9b100fa_0  
xorg-xorgproto            2024.1               h5eee18b_1  
xz                        5.6.4                h5eee18b_1  
yaml                      0.2.5                h7b6447c_0  
zlib                      1.2.13               h5eee18b_1  
zstd                      1.5.6                hc292b87_0  
```



## 整体结构

- 入口： [run.py] 统一解析参数、设随机种子、选择任务分支（forecast / forecast_test_all / forecast_test_all2 / forecast_test_all3），并将 vote_rate 从百分比换算为比例。
- 训练与基础推理： exp/exp_forecast.py 负责模型构建、优化器/损失选择、训练循环、验证与测试推理，支持 DDP/DP。

- 数据加载： data_provider/data_factory.py 负责数据集选择与别名映射；为 BJTU 异常数据集提供专门分支，透传 test_data_path、scaler 等自定义参数。
- 数据集实现： data_provider/data_loader.py 中的 BJTUAnomalyloader（其余数据类保持原有逻辑）。
- 异常检测方案：
  - 校准+批量检测： exp/exp_AnomalyDetection.py（Exp_Forecast_TestAll2）
  - 一次性快速寻优： exp/TestInOnce.py（Exp_Forecast_TestAll3）

##data_factory（工厂模式）引入了BJTU

#### A. 路由解析 (Alias Resolution)

- **机制**: 利用 `DATA_ALIASES` 字典建立映射层。
- **目的**: 将分散的物理数据集名（如 `ETTh1`, `Traffic`）归一化为统一的逻辑处理类（`MultivariateDatasetBenchmark`），减少冗余代码。
- **异常处理**: 显式检查 `data_key` 是否存在，若未注册则抛出包含所有可选值的 `ValueError`，便于调试。

#### B. 上下文配置 (Context Configuration)

- **Shuffle 策略**:
  - Train 阶段: `True`
  - Val/Test 阶段: `False`

#### C. 分支实例化 (Branching Instantiation)

- **分支 1: 定制化业务 (Target: BJTU)**
  - **触发条件**: `data_key == 'BJTU'`
  - **参数清洗**: 使用 `kwargs.pop('test_data_path')`。
    - *原理*: 从 `kwargs` 中**取出并删除**该键值对。
    - *作用*: 防止后续将 `**kwargs` 传给 `BJTUAnomalyloader` 构造函数时，出现 "multiple values for keyword argument" 错误（即避免显式传参和 kwargs 隐式传参冲突）。
  - **接口适配**: 将 Transformer 视角的 `input_token_len` 映射为 Dataset 视角的 `patch_len`。
- **分支 2: 标准化基准 (Target: Benchmark/Standard)**
  - **触发条件**: 其他所有数据集。
  - **特征工程**: 计算 `timeenc`（是否使用 TimeFeatures 嵌入）。
  - **参数传递**: 传递标准的 `size` 列表 `[seq, label, pred]` 和 `features` 类型 (M/S/MS)。

#### D.例子

用户在命令行启动测试脚本。 `python run.py --task_name anomaly_detection --data BJTU --batch_size 32 --seq_len 96`

exp中调用train_loader = data_provider(args, flag='train')

**Step 1: 查表** Factory 收到请求，查看 `args.data="BJTU"`。它识别出这是定制分支，不是通用 Benchmark。

**Step 2: 构造 Dataset (实例化)** Factory 开始组装参数字典 `kwargs`。

- 它发现 BJTU 数据集需要一个特殊的参数 `patch_len`，但 `args` 里只有 `seq_len` 和 `input_token_len`。
- **关键动作**: 代码执行 `patch_len = args.input_token_len`，完成了参数适配。
- 它调用: `dataset = BJTUAnomalyloader(root_path='...', mode='train', patch_len=..., ...)`
- 此时，`BJTUAnomalyloader` 在内存中完成了文件的读取。

**Step 3: 构造 DataLoader (封装)** Factory 检查 `args.task_name`。

- 发现是 `anomaly_detection`。
- **关键决策**: 设置 `drop_last = False`。这意味着如果最后的数据不够一个 Batch (32个)，通过补齐或保留原样输出，绝不丢弃
- 它调用: `loader = DataLoader(dataset, batch_size=32, shuffle=True, ...)`

## data_provider/data_loader.py 中 BJTUAnomalyloader 的主要实现：

在 `data_provider` 体系中，不同的 Dataset 类实际上充当了 **适配器（Adapter）** 的角色。



上层的 `DataLoader` 和模型（Model）期望的数据格式是**标准化的**：

- **输入**: Index (索引)
- **输出**: Tensor `[Seq_Len, Channel]`

但底层物理数据的存储方式是**千奇百怪的**：

- **Standard Dataset**: 单个 CSV 文件，包含时间戳列，需要 `TimeFeature` 编码。
- **BJTU Dataset (本例)**: 可能是文件夹而非文件，CSV **没有表头 (Headerless)**，甚至包含脏数据（非数字字符），且训练/验证集是在单一文件中硬切分的（80/20切分）。
- 如果强行用 `MultivariateDatasetBenchmark` 加载 BJTU 数据，会导致：
  1. **解析错误**: 通用类默认第一行是表头，会把 BJTU 的第一行数据吃掉。
  2. **逻辑污染**: 为了兼容 BJTU 的“文件夹拼接逻辑”和“80/20 内存切分”，必须在通用类里写大量的 `if dataset == 'BJTU': ...`，这违反了 **开闭原则 (Open/Closed Principle)**。





- 目的：面向 BJTU 异常检测数据，支持训练/验证从 train 目录切分，测试按指定 CSV 单文件滑窗；统一使用 StandardScaler，可外部注入 scaler 复用训练归一化。
- 路径与分割：
  - train/val：若 root_path 已含 train 则直接用；否则优先 root_path/train（存在则用），再回退原 root_path。train 取前 80%，val 取后 20%。
  - test：支持传入 test_data_path（或 kwargs['data_path']）；若路径不存在则尝试 root_path 拼接；最终要求找到单个 CSV。
- stride（步长）：train/val 用 stride=1；test 用 seq_len - 2*patch_len，若小于 1 则夹到 1，用于非重叠/少重叠滑窗检测。
- 读取：CSV 强制数值化，丢弃无法转数字的行；数据堆叠后返回 N×D。





## 例子：运行一个寻优的脚本

1. 定位检查点
   - 在 checkpoints/ 下按时间找最新的目录，形如 checkpoints/forecast_永济工况1_转速1370转每秒_微调_autotimes_*，取其 basename 作为 --test_dir，模型文件用 checkpoint.pth。
2. 入口命令
   - python -u run.py --task_name forecast_test_all3 --is_training 0，模型 autotimes，数据集 BJTU，根目录 root_path 指向工况1测试集目录。n_vars=3，seq_len=2880，input_token_len=96，output_token_len=96，test_pred_len=96，batch_size=512，GPU 选择 0。未显式传 percentile/vote_rate，内部网格搜索自动处理。
3. run.py 解析参数
   - 将 vote_rate 由百分数转为比例（这里没传，后续在 TestAll3 内部自定列表），设置随机种子和 CUDA 设备，按 task_name=forecast_test_all3 选择 Exp_Forecast_TestAll3，并传入 args。
4. data_factory 路由
   - args.data="BJTU" 命中 data_dict 的 BJTUAnomalyloader 分支。test 阶段 shuffle=False，drop_last=False。test_data_path 为空，BJTU loader 会用 root_path 下的文件。
5. BJTUAnomalyloader 读取数据
   - flag="test"：stride = seq_len - 2*patch_len = 2880 - 2*96 = 2688，最小夹到 1（这里 2688）。
   - test_data_path 为空时，尝试用 root_path（已指向 test 目录）下的 CSV；如路径不存在会再拼接 root_path。要求最终找到单个 CSV。
   - 读 CSV：pd.read_csv(header=None) → to_numeric → dropna，仅保留数值。
   - 归一化：若上层传入 scaler 则复用；否则自行 fit_transform(StandardScaler)。
   - **len** = (len(data) - seq_len) // stride + 1；**getitem** 返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)，seq_x/seq_y 同一个 2880 长度窗口、3 维；mark 为占位零张量。
6. Exp_Forecast_TestAll3 流程（一次性寻优）
   - 加载 checkpoint（strict=False），MSELoss(reduction='none')。
   - 先跑一遍 root_path/test 下所有 CSV，逐文件收集窗口级 MSE（使用上面的 dataloader，步长 2688）。文件名含 “正常/normal/m0_g0_la0_ra0” 的样本加入 calibration_losses，其他记录在 test_data_map。
   - 用 calibration_losses 构建基准分布 val_flattened，打印各文件平均 MSE，基准均值通常 ~0.02。若无正常文件则报错退出。
   - 网格搜索 percentile 列表 + vote_rate 列表，约束：基准池 FPR ≤ 10%；按 gap=检测率-误报率 排序，展示 Top-8，并将最佳结果存 test_results/<setting>/best_results.npy。
   - 输出日志与榜单写入 all_tuning_logs/GridSearch_<model_id>_<model_name>.log。



## exp实验类型的扩展，exp2是进行测试，exp3是网格化搜索的快速测试和参数寻优

#### exp/exp_forecast.py（基础训练与预测）

- 输入（核心参数）：
  - 数据：--data（如 ETTh1/BJTU）、--root_path、--data_path
  - 时序：--seq_len、--input_token_len、--output_token_len、--test_pred_len
  - 训练：--batch_size、--train_epochs、--learning_rate、--cosine/--lradj、--patience
  - 模型：--model、--d_model、--n_heads、--e_layers、--d_ff 等
  - 并行：--ddp/--dp、--gpu
- 训练输出：
  - checkpoints/<setting>/checkpoint.pth
  - 日志打印；early stopping 触发后加载最佳模型
- 测试输出：
  - test_results/<setting>/ 下可视化（若 visualize）
  - result_long_term_forecast.txt 中写入 mse/mae
- 数据流：
  1. _get_data → data_provider → data_factory 选择数据集（BJTU 走 BJTUAnomalyloader）。
  2. 训练：MSELoss + Adam，支持 CosineAnnealingLR；每 epoch 验证/测试。
  3. 测试：自回归多步推理，拼接 output_token_len 片段直到 test_pred_len，计算指标。

75 x 19

以下为三份代码的输入/输出与用法概览。

------

## exp/exp_forecast.py（基础训练与预测）

- 输入（核心参数）：
  - 数据：--data（如 ETTh1/BJTU）、--root_path、--data_path
  - 时序：--seq_len、--input_token_len、--output_token_len、--test_pred_len
  - 训练：--batch_size、--train_epochs、--learning_rate、--cosine/--lradj、--patience
  - 模型：--model、--d_model、--n_heads、--e_layers、--d_ff 等
  - 并行：--ddp/--dp、--gpu
- 训练输出：
  - checkpoints/<setting>/checkpoint.pth
  - 日志打印；early stopping 触发后加载最佳模型
- 测试输出：
  - test_results/<setting>/ 下可视化（若 visualize）
  - result_long_term_forecast.txt 中写入 mse/mae
- 数据流：
  1. _get_data → data_provider → data_factory 选择数据集（BJTU 走 BJTUAnomalyloader）。
  2. 训练：MSELoss + Adam，支持 CosineAnnealingLR；每 epoch 验证/测试。
  3. 测试：自回归多步推理，拼接 output_token_len 片段直到 test_pred_len，计算指标。

示例（ETTh1 训练 + 测试）：

------

#### exp/exp_AnomalyDetection.py（TestAll2：校准+批量异常检测）

- 输入（核心参数）：
  - --task_name forecast_test_all2 --is_training 0
  - --percentile（默认 99.9），--vote_rate（命令行传百分数，内部 /100）
  - 数据：--data BJTU、--root_path（若传 root_path/test 会自动上移一层）、--seq_len、--input_token_len、--output_token_len
  - 模型：--test_dir、--test_file_name
- 输出：
  - test_results/<setting>/result_summary.txt（每 CSV 的异常率）
  - 每文件 *_flags.npy（样本级 0/1）
- 流程：
  1. 读取/加载 checkpoint，设 eval。
  2. 校准：val_loader（降采样 10% 批次）计算点级 MSE 分布，阈值=percentile。
  3. 测试：遍历 root_path/test 下 CSV，滑窗推理，点级 MSE 超阈计票；若异常点比例 > vote_rate，则样本记为异常，统计异常率。

#### exp/TestInOnce.py（TestAll3：一次推理内自动寻优）

- 输入（核心参数）：
  - --task_name forecast_test_all3 --is_training 0
  - 数据：--data BJTU、--root_path（可直接指向 test/ 或其上层）、--seq_len、--input_token_len、--output_token_len、--n_vars
  - 模型：--test_dir、--test_file_name
- 输出：
  - test_results/<setting>/best_results.npy（Rank1 组合各文件异常率）
  - 终端打印 Top-8 (P,V) 组合榜单
- 流程：
  1. 加载 checkpoint，MSELoss(reduction='none')。
  2. 用 _get_data(flag='val') 仅获取 scaler，保证与训练归一化一致。
  3. 遍历 root_path/test 下 CSV，滑窗推理，收集每窗口点级均值 MSE。文件名含“正常/normal/m0_g0_la0_ra0”者进入 calibration pool，其余进入 test_data_map。
  4. 用 calibration pool 计算阈值分布；若无正常文件则报错退出。
  5. 网格搜索 p_list × v_list，约束基准误报 ≤10%、正常平均误报 ≤10%，按 gap(检出-误报) 排序，展示 Top-8；保存 Rank1 结果。

## **autotimes.py（模型层）**

- 输入：[x_enc]形状 [B, L, M]，[x_mark_enc/x_mark_dec]未使用（占位）。
- 逻辑：
  1. 禁用 transformers 的安全加载检查（覆盖 check_torch_load_is_safe）。
  3. 冻结 LLM 参数，仅训练 tokenizer/decoder（Linear 或 MLP）。
  4. 前向：按变量展开 → patch 切分（size=input_token_len，step 同步）→ encoder → LLM（输出 hidden_states[-1]）→ decoder → 还原形状 [B, L', M]，可选去标准化。
- 输出：预测序列，形状与输入时间维匹配（token_len 及解码长度决定）。

## **run.py（入口脚本）**


- 任务分派：forecast → Exp_Forecast；forecast_test_all2 → Exp_Forecast_TestAll2；forecast_test_all3 → Exp_Forecast_TestAll3。

- vote_rate 命令行为百分数，内部自动 /100。
- 例如的输入vote_rate=1，内部会变成1%
- BJTU 测试滑窗步长=seq_len-2*input_token_len，至少 1。
- AutoTimes 需本地 LLM 权重目录（llm_model=OPT/GPT2/LLAMA 对应子目录），已禁用 transformers 加载安全检查。



## 训练和测试的脚本用例

#### 训练-工况1.sh（autotimes，OPT，BJTU，工况1）/raid/hsz/HSZ_OpenLTM/autotimes测试opt/永济电机轴承数据/训练-工况1.sh

- 任务/模型：`task_name=forecast`，模型 `autotimes`，LLM 选 `OPT`。
- 数据：`data=BJTU`，`root_path=/raid/hsz/OpenLTM_data_backup/datasets/永济电机轴承数据集/工况1.../train/`，自动取首个 CSV 为 `data_path`。
- 序列配置：`token_num=30`，`token_len=96` ⇒ `seq_len=2880`；`input_token_len=96`，`output_token_len=96`，`n_vars=3`。
- 模型超参：`d_model=1024`，`d_ff=2048`，`e_layers=8`，`n_heads=8`，`use_norm` 开启。
- 训练超参：`batch_size=256`，`train_epochs=5`，`learning_rate=1e-4`，GPU 0。
- 预训练权重路径在脚本里声明了 `pretrain_model_path`，但未传给 run.py，也未启用 `--adaptation`，因此当前写法是从头训练/微调，不会加载该权重。
- 运行行为：调用 `python run.py --task_name forecast --is_training 1 ...`，走 Exp_Forecast，输出到 `checkpoints/forecast_<setting>/checkpoint.pth`。

输出：模型 checkpoint（`checkpoints/.../checkpoint.pth`），训练日志；预测指标写入 `result_long_term_forecast.txt`（测试阶段才会生成）。

------

#### 测试-工况1.sh（异常检测 TestAll2，OPT，BJTU，工况1）/raid/hsz/HSZ_OpenLTM/autotimes测试opt/永济电机轴承数据/测试-工况1.sh

- 任务/模型：`task_name=forecast_test_all2`，模型 `autotimes`，LLM 选 `OPT`。
- 数据：`root_path=/raid/hsz/OpenLTM_data_backup/datasets/永济电机轴承数据集/工况1.../test/`，`data=BJTU`；测试阶段 BJTUAnomalyloader 会遍历该目录的 CSV。
- 序列配置需与训练一致：`seq_len=2880`，`input_token_len=96`，`output_token_len=96`，`test_pred_len=96`，`n_vars=3`；模型超参同训练（d_model/d_ff/e_layers/n_heads）。
- 异常检测参数：
  - 脚本中 `PARAMS` 定义多组 `(percentile, vote_rate)`，如 99.9/1.0、99.5/1.0 等。
  - **注意**：run.py 会把传入的 vote_rate 当作“百分数”再除以 100；要表示 1% 应传 `1.0`，不要传 `0.01`，否则变成 0.0001。
- 检查点查找：`CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* | head -n1)`，取最新目录 basename 作为 `--test_dir`，文件名 `checkpoint.pth`。
- 运行行为：对 PARAMS 中的每组 (p,v) 调用一次 `python run.py --task_name forecast_test_all2 ...`。流程：
  1. 用验证集（降采样 10% 批次）计算点级 MSE 分布，取 percentile 得到阈值。
  2. 遍历 test 目录 CSV，滑窗推理，点级 MSE 超阈计票；若异常点比例 > vote_rate 判该窗口/样本为异常，统计异常率。
  3. 结果写入 `test_results/<setting>/result_summary.txt` 和各文件的 `*_flags.npy`；日志追加到 `all_tuning_logs/_${model_id}_${model_name}_all_experiments.log`。

输出：每组参数的结果在 `test_results/<setting>/result_summary.txt`，并生成样本级标记 NPY；日志在 `all_tuning_logs/`。

### 自动寻优-工况1.sh（autotimes + TestAll3，路径 `/autotimes测试opt/...`）

- 任务：`forecast_test_all3`（一次推理内自动寻优），模型 `autotimes`，LLM `OPT`，数据 `BJTU`，3 维。
- 参数：`seq_len=2880`，`input_token_len=96`，`output_token_len=96`，`test_pred_len=96`，`d_model=1024`，`d_ff=2048`，`e_layers=8`，`n_heads=8`，`batch_size=512`，GPU 0。
- 检查点：取最新 `checkpoints/forecast_${model_id}_${model_name}_*` 的 basename 作为 `--test_dir`，加载 `checkpoint.pth`。
- 流程：不需要手动传 percentile/vote_rate，脚本去掉循环，TestAll3 内部自动：
  1. 用 val loader 仅获取 scaler；
  2. 遍历 test 目录 CSV，滑窗推理，收集 MSE；
  3. 自动识别文件名含 “正常/normal/m0_g0_la0_ra0” 的样本为基准池，计算阈值分布；
  4. 内置网格搜索 (P,V)，筛掉正常误报 >10% 的组合，按 gap 排序输出 Top-8，保存 Rank1 结果到 `test_results/<setting>/best_results.npy`；
  5. 日志写 `all_tuning_logs/GridSearch_${model_id}_${model_name}.log`。

### 训练-工况1.sh（timer_xl 版本，路径 `/zhaojiash/timer_xl/...`）

- 任务：`forecast`，模型 `timer_xl`，数据 `BJTU`，3 维。
- 序列与模型：`seq_len=30*96=2880`，`input_token_len=96`，`output_token_len=96`，`n_vars=3`，`d_model=1024`，`d_ff=2048`，`e_layers=8`，`n_heads=8`，`use_norm` 开启。
- 训练：`batch_size=256`，`train_epochs=10`，`learning_rate=5e-6`，GPU 0。
- 预训练：开启 `--adaptation`，加载 `pretrain_model_path=/home/ubuntu/zhaojia/checkpoints/Timer_xl/checkpoint.pth`。
- 数据：`root_path` 指向工况1训练目录，自动取首个 CSV 作为 data_path。
- 输出：`checkpoints/forecast_<setting>/checkpoint.pth`，日志打印，训练后 result_long_term_forecast.txt（测试阶段写）。

### 测试-工况1.sh（timer_xl + TestAll2，路径 `/zhaojiash/timer_xl/...`）

- 任务：`forecast_test_all2`（校准+批量异常检测），模型 `timer_xl`，数据 `BJTU`，3 维。
- 参数保持与训练一致：`seq_len=2880`，`input_token_len=96`，`output_token_len=96`，`test_pred_len=96`，`d_model/d_ff/e_layers/n_heads` 同步。
- 检测参数：循环多组 `(percentile, vote_rate)`。**注意** run.py 会把 vote_rate 当“百分数”再除以 100，要表示 1% 应传 `1.0`，脚本里用 `0.01` 会变成 0.0001，需自行修正。
- 流程：寻找最新 `checkpoints/forecast_${model_id}_${model_name}_*` 目录 → 加载 `checkpoint.pth` → 用验证集降采样校准阈值 → 遍历 test 目录 CSV 滑窗推理，统计异常率 → 结果写 `test_results/<setting>/result_summary.txt` 与各文件 `*_flags.npy`，日志汇总到 `all_tuning_logs/_...log`。
