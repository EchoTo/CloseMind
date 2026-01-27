# Windows 部署指南（NVIDIA GPU版）

本文档介绍如何在Windows台式机上部署CloseMind量化预测系统，并利用NVIDIA GPU加速模型训练。

## 硬件要求

- **CPU**: Intel/AMD 多核处理器
- **内存**: 16GB+（推荐32GB）
- **GPU**: NVIDIA RTX 5070Ti（12GB显存）
- **硬盘**: 100GB+ 可用空间（SSD推荐）

## 软件环境

- Windows 10/11 64位
- Python 3.10+
- CUDA 12.x
- cuDNN 8.x+

---

## 第一步：安装NVIDIA驱动和CUDA

### 1.1 安装最新显卡驱动

1. 访问 [NVIDIA驱动下载](https://www.nvidia.cn/Download/index.aspx)
2. 选择：
   - 产品类型：GeForce
   - 产品系列：GeForce RTX 50 Series
   - 产品：GeForce RTX 5070 Ti
   - 操作系统：Windows 11 64-bit
3. 下载并安装最新驱动

### 1.2 安装CUDA Toolkit

1. 访问 [CUDA Toolkit下载](https://developer.nvidia.com/cuda-downloads)
2. 选择：
   - Operating System: Windows
   - Architecture: x86_64
   - Version: 11 或 10
   - Installer Type: exe (local)
3. 下载并安装（建议选择"自定义安装"，只安装CUDA）

### 1.3 安装cuDNN

1. 访问 [cuDNN下载](https://developer.nvidia.com/cudnn)（需要NVIDIA账号）
2. 下载与CUDA版本匹配的cuDNN
3. 解压后将文件复制到CUDA安装目录：
   ```
   bin\*.dll -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   include\*.h -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include
   lib\x64\*.lib -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64
   ```

### 1.4 验证CUDA安装

打开命令提示符（CMD）：
```cmd
nvidia-smi
nvcc --version
```

应该能看到GPU信息和CUDA版本。

---

## 第二步：安装Python环境

### 2.1 安装Miniconda（推荐）

1. 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. 运行安装程序，勾选"Add to PATH"
3. 安装完成后重启终端

### 2.2 创建虚拟环境

打开 Anaconda Prompt：
```cmd
conda create -n closemind python=3.10
conda activate closemind
```

---

## 第三步：安装TA-Lib（关键步骤）

Windows上安装TA-Lib比较特殊，有两种方法：

### 方法一：使用预编译wheel（推荐）

1. 访问 [TA-Lib Windows Wheels](https://github.com/cgohlke/talib-build/releases)
2. 下载对应Python版本的wheel文件，例如：
   - `TA_Lib-0.4.32-cp310-cp310-win_amd64.whl`（Python 3.10）
3. 安装：
```cmd
pip install TA_Lib-0.4.32-cp310-cp310-win_amd64.whl
```

### 方法二：使用conda-forge

```cmd
conda install -c conda-forge ta-lib
```

### 验证安装

```python
python -c "import talib; print(talib.__version__)"
```

---

## 第四步：安装PyTorch（GPU版）

### 4.1 安装支持CUDA的PyTorch

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取最新安装命令。

对于CUDA 12.x：
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

对于CUDA 11.8：
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.2 验证GPU可用

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

应该输出：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

---

## 第五步：安装项目依赖

### 5.1 克隆项目

```cmd
cd D:\Projects
git clone <repository-url> CloseMind
cd CloseMind
```

### 5.2 安装其他依赖

```cmd
pip install -r requirements.txt
```

如果某些包安装失败，可以单独安装：
```cmd
pip install pandas numpy scipy
pip install akshare
pip install lightgbm xgboost
pip install scikit-learn
pip install matplotlib seaborn plotly
pip install streamlit
pip install pyyaml loguru tqdm
pip install jupyter ipykernel
pip install pyarrow joblib
pip install einops              :: Transformer tensor操作
```

### 5.3 创建数据目录

```cmd
mkdir data_storage\raw
mkdir data_storage\processed
mkdir data_storage\qlib
mkdir checkpoints
mkdir logs
mkdir reports
```

---

## 第六步：配置GPU训练

### 6.1 修改配置文件

编辑 `config/config.yaml`，确保GPU配置正确：

```yaml
# GPU配置
gpu:
  enabled: true
  device: "cuda:0"
  mixed_precision: true  # 5070Ti支持FP16/BF16，可加速训练

# XGBoost GPU配置
models:
  xgboost:
    params:
      tree_method: "gpu_hist"
      gpu_id: 0

# 深度学习模型（12GB显存可适当增大batch_size）
models:
  lstm:
    training:
      batch_size: 512
  patchtst:
    training:
      batch_size: 512
  itransformer:
    training:
      batch_size: 512
  mamba:
    training:
      batch_size: 512
  moe:
    training:
      batch_size: 512    # MoE显存占用较大，酌情调小
```

### 6.2 检查显存使用

训练时可以用以下命令监控GPU：
```cmd
nvidia-smi -l 1
```

---

## 第七步：运行系统

### 7.1 下载数据

```cmd
python main.py download --all
```

首次下载全A股数据可能需要较长时间（1-2小时）。

### 7.2 训练模型

```cmd
:: 训练集成模型（全部7个子模型）
python main.py train

:: 使用脚本单独训练
python scripts/train.py --model lgb
python scripts/train.py --model xgb
python scripts/train.py --model lstm
python scripts/train.py --model patchtst
python scripts/train.py --model itransformer
python scripts/train.py --model mamba
python scripts/train.py --model moe
python scripts/train.py --model ensemble
python scripts/train.py --model all
```

**模型说明**（RTX 5070Ti 12GB显存）：

| 模型 | 类型 | 集成权重 |
|------|------|---------|
| LightGBM | 树模型 (CPU) | 10% |
| XGBoost | 树模型 (GPU) | 10% |
| Bi-LSTM+Attention | RNN+注意力 (GPU) | 5% |
| PatchTST | Transformer (GPU) | 25% |
| iTransformer | 反转Transformer (GPU) | 20% |
| Mamba | 状态空间 (GPU) | 15% |
| MoE | 混合专家 (GPU) | 15% |

### 7.3 生成预测

```cmd
:: 基础预测
python main.py predict

:: 增强版预测（带持仓跟踪和准确率分析）
python scripts/predict_enhanced.py --analyze-accuracy
```

### 7.4 启动Web界面

```cmd
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

---

## 常见问题

### Q1: CUDA out of memory

减小batch_size，或禁用部分模型：
```yaml
models:
  patchtst:
    training:
      batch_size: 128
  itransformer:
    training:
      batch_size: 128
  mamba:
    training:
      batch_size: 128
  moe:
    enabled: false  # MoE显存占用最大，可先禁用
```

### Q2: TA-Lib安装失败

使用conda-forge：
```cmd
conda install -c conda-forge ta-lib
```

### Q3: akshare连接超时

设置代理或更换网络：
```python
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
```

### Q4: PyTorch找不到CUDA

1. 确认CUDA版本匹配：
```cmd
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

2. 重新安装匹配版本的PyTorch

### Q5: 训练时GPU利用率低

增大batch_size和num_workers：
```yaml
models:
  patchtst:
    training:
      batch_size: 1024
  itransformer:
    training:
      batch_size: 1024
  mamba:
    training:
      batch_size: 1024

data:
  download:
    parallel_workers: 8
```

### Q6: Windows Defender影响性能

将项目目录添加到排除列表：
1. 设置 → Windows安全中心 → 病毒和威胁防护
2. 管理设置 → 排除项 → 添加排除项
3. 添加项目文件夹

---

## 性能优化建议

### 针对RTX 5070Ti的优化

1. **启用混合精度训练**（已在配置中启用）
   - 所有深度模型(LSTM/PatchTST/iTransformer/Mamba/MoE)均支持FP16混合精度
   - 显著加速训练，减少显存占用

2. **增大batch_size**
   - 12GB显存参考: batch_size=512 可同时训练大部分模型
   - MoE由于多专家并行，显存占用最大，可适当调小

3. **使用SSD存储数据**
   - 数据读取速度提升明显
   - 建议将data_storage放在SSD

4. **多进程数据加载**
   ```yaml
   data:
     download:
       parallel_workers: 8  # 根据CPU核心数调整
   ```

---

## 定时任务设置

### 每日自动更新数据和预测

1. 创建批处理脚本 `daily_update.bat`：
```batch
@echo off
cd /d D:\Projects\CloseMind
call conda activate closemind

echo [%date% %time%] Starting daily update...

:: 更新数据
python main.py download --stocks

:: 生成预测
python scripts/predict_enhanced.py

echo [%date% %time%] Daily update completed.
```

2. 使用Windows任务计划程序设置定时运行：
   - 打开"任务计划程序"
   - 创建基本任务
   - 设置触发器（如每天18:00）
   - 操作选择"启动程序"，指向bat文件

---

## 从Mac同步代码

如果在Mac上开发，Windows上训练：

```cmd
:: 使用Git同步
cd D:\Projects\CloseMind
git pull origin main

:: 或使用rsync（需要安装）
rsync -avz user@mac:/Library/Git/CloseMind/ D:\Projects\CloseMind\
```

---

## 联系与支持

如遇到问题，请检查：
1. Python环境是否正确激活
2. CUDA版本与PyTorch是否匹配
3. 显卡驱动是否最新
4. 项目依赖是否完整安装

查看日志文件获取详细错误信息：
```cmd
type logs\*.log
```
