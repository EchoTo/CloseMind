# CloseMind - A股量化预测系统

一个基于机器学习的A股量化预测系统，支持多周期预测、多模型集成和可视化分析。

## 功能特点

- **多周期预测**: 日度 + 周度预测
- **全市场覆盖**: 支持全A股约5000只股票
- **7模型集成**: LightGBM + XGBoost + Bi-LSTM+Attention + PatchTST + iTransformer + Mamba + MoE
- **150+特征**: 技术指标、量价特征、市场特征、Alpha因子、舆情特征
- **持仓跟踪**: 跟踪买卖信号历史，展示持仓收益和预期
- **可视化界面**: 基于Streamlit的本地Web界面
- **GPU加速**: 支持CUDA加速深度学习模型

## 系统架构

```
CloseMind/
├── config/                 # 配置文件
│   └── config.yaml        # 全局配置
├── data/                   # 数据层
│   ├── downloader.py      # 数据下载（akshare）
│   ├── processor.py       # 数据清洗和处理
│   └── qlib_converter.py  # 转换为qlib格式
├── features/               # 特征工程
│   ├── technical.py       # 技术指标（MA/MACD/RSI/KDJ等）
│   ├── volume_price.py    # 量价特征
│   ├── market.py          # 市场特征
│   ├── alpha.py           # Alpha因子
│   └── sentiment.py       # 舆情特征
├── models/                 # 模型层
│   ├── lgb_model.py       # LightGBM排序模型
│   ├── xgb_model.py       # XGBoost排序模型
│   ├── lstm_model.py      # Bi-LSTM + Multi-Head Attention
│   ├── patchtst_model.py  # PatchTST (ICLR 2023)
│   ├── itransformer_model.py # iTransformer (ICLR 2024)
│   ├── mamba_model.py     # Mamba状态空间模型 (ICLR 2025)
│   ├── moe_model.py       # 混合专家MoE (ICLR 2025 Spotlight)
│   └── ensemble.py        # 7模型动态集成
├── strategy/               # 策略层
│   ├── signal.py          # 信号生成
│   ├── portfolio.py       # 组合优化
│   └── position_tracker.py # 持仓跟踪
├── backtest/               # 回测层
│   └── evaluator.py       # 回测评估
├── report/                 # 报告层
│   └── visualization.py   # 可视化
├── scripts/                # 脚本
│   ├── download_data.py   # 下载数据脚本
│   ├── train.py           # 训练脚本
│   ├── predict.py         # 预测脚本
│   └── predict_enhanced.py # 增强版预测（带持仓跟踪）
├── app.py                  # Web界面入口
└── main.py                 # CLI入口
```

## 部署指南

- **macOS部署**: 见下方说明
- **Windows部署（GPU训练）**: 见 [docs/WINDOWS_DEPLOY.md](docs/WINDOWS_DEPLOY.md)

---

## macOS 本地部署（MacBook Pro）

### 环境要求

- macOS 12.0+
- Python 3.9+
- 8GB+ 内存（推荐16GB）
- 50GB+ 磁盘空间（用于存储数据）

### 安装步骤

#### 1. 克隆项目

```bash
cd /Library/Git
git clone <repository-url> CloseMind
cd CloseMind
```

#### 2. 创建虚拟环境

```bash
# 使用conda（推荐）
conda create -n closemind python=3.10
conda activate closemind

# 或使用venv
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装TA-Lib依赖

TA-Lib需要先安装C库：

```bash
# 使用Homebrew安装
brew install ta-lib
```

#### 4. 安装Python依赖

```bash
pip install -r requirements.txt
```

如果PyTorch安装有问题，可以单独安装：

```bash
# CPU版本（MacBook无NVIDIA GPU）
pip install torch torchvision

# 或使用MPS加速（Apple Silicon）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 5. 创建数据目录

```bash
mkdir -p data_storage/{raw,processed,qlib}
mkdir -p checkpoints logs reports
```

## 使用指南

### 方式一：Web可视化界面（推荐）

启动Web界面：

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，界面包含：

- **今日预测**: 查看当日买入/卖出建议
- **个股分析**: 单只股票的K线图和技术分析
- **回测评估**: 查看历史回测报告
- **系统状态**: 查看系统运行状态和快捷操作

### 方式二：命令行界面

#### 下载数据

```bash
# 下载全部数据（首次运行）
python main.py download --all

# 仅下载股票数据
python main.py download --stocks

# 仅下载指数数据
python main.py download --indices
```

#### 训练模型

```bash
# 训练集成模型（默认训练全部7个子模型）
python main.py train

# 使用脚本单独训练
python scripts/train.py --model lgb
python scripts/train.py --model xgb
python scripts/train.py --model lstm
python scripts/train.py --model patchtst
python scripts/train.py --model itransformer
python scripts/train.py --model mamba
python scripts/train.py --model moe
python scripts/train.py --model ensemble
python scripts/train.py --model all

# 训练后评估
python scripts/train.py --model ensemble --evaluate
```

#### 生成预测

```bash
# 基础预测
python main.py predict

# 增强版预测（带持仓跟踪）
python scripts/predict_enhanced.py --analyze-accuracy

# 指定日期
python main.py predict --date 2025-01-20
```

### 方式三：脚本直接运行

```bash
# 下载数据
python scripts/download_data.py

# 训练模型
python scripts/train.py --model all

# 生成预测
python scripts/predict.py --top-n 50
```

## 配置说明

配置文件位于 `config/config.yaml`，主要配置项：

### 数据配置

```yaml
data:
  source: "akshare"
  start_date: "2022-01-01"
  stock_filter:
    exclude_st: true           # 排除ST股
    exclude_new_stock_days: 60 # 排除上市不满60天的新股
```

### 模型配置

系统包含7个模型，按类型分为两类：

**树模型（表格数据）：**
- LightGBM — LambdaRank排序模型
- XGBoost — Pairwise排序模型（GPU加速）

**深度学习模型（时序数据）：**
- Bi-LSTM + Attention — 双向LSTM + 多头时间注意力
- PatchTST — 基于Patch的Transformer (ICLR 2023)
- iTransformer — 反转Transformer，变量作token (ICLR 2024)
- Mamba — 选择性状态空间模型，双向扫描 (ICLR 2025)
- MoE — 混合专家Top-K门控网络 (ICLR 2025 Spotlight)

```yaml
models:
  ensemble:
    method: "weighted_average"  # 或 "stacking"
    weights:
      lightgbm: 0.10
      xgboost: 0.10
      lstm: 0.05
      patchtst: 0.25       # 主力模型
      itransformer: 0.20
      mamba: 0.15
      moe: 0.15
```

### 信号阈值

```yaml
signal:
  threshold:
    strong_buy: 0.8    # 强烈买入
    buy: 0.6           # 买入
    hold: 0.4          # 持有
    sell: 0.2          # 卖出
```

### 舆情特征（可选）

```yaml
features:
  sentiment:
    enabled: true      # 启用舆情特征
    news_weight: 0.6
    announcement_weight: 0.4
```

## 预测报告说明

系统生成的预测报告包含：

### 买入/持有建议

| 字段 | 说明 |
|------|------|
| 代码 | 股票代码 |
| 信号 | strong_buy/buy/hold |
| 分数 | 综合预测分数（0-1） |
| 持有天数 | 从首次买入信号到现在的天数 |
| 当前收益 | 相对入场价的收益率 |
| 预期收益 | 基于历史数据估算的本轮预期收益 |
| 预期天数 | 建议继续持有的天数 |
| 趋势 | 上涨/下跌/震荡 |

### 卖出建议

| 字段 | 说明 |
|------|------|
| 代码 | 股票代码 |
| 信号 | sell/strong_sell |
| 上次买入 | 上次买入信号的日期 |
| 持有天数 | 本轮持有的天数 |
| 本轮收益 | 本轮交易的收益率 |

## 常见问题

### Q: TA-Lib安装失败？

确保先安装C库：
```bash
brew install ta-lib
```

如果还有问题，尝试：
```bash
export TA_LIBRARY_PATH=/opt/homebrew/lib
export TA_INCLUDE_PATH=/opt/homebrew/include
pip install ta-lib
```

### Q: 数据下载很慢？

akshare使用国内数据源，正常网络应该较快。可以调整配置：
```yaml
data:
  download:
    parallel_workers: 8  # 增加并行数
    batch_size: 50       # 减少批量大小
```

### Q: 内存不足/显存不足？

减少训练数据量或批次大小：
```yaml
training:
  train_start: "2023-01-01"  # 缩短训练时间范围

models:
  # 所有深度学习模型都支持调整batch_size
  patchtst:
    training:
      batch_size: 128  # 减小批次大小
  itransformer:
    training:
      batch_size: 128
  mamba:
    training:
      batch_size: 128
  moe:
    training:
      batch_size: 128
```

也可以在 `config.yaml` 中将部分模型设为 `enabled: false` 来减少资源占用。

### Q: Apple Silicon如何加速？

PyTorch支持MPS后端（Apple Silicon GPU）：
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

系统会自动检测并使用可用的加速设备。

## 免责声明

本系统仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。作者不对使用本系统造成的任何损失负责。

## License

MIT License
