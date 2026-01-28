"""
Bi-LSTM + Multi-Head Attention时序模型
升级版: 双向LSTM + 多头注意力机制
使用PyTorch实现,支持GPU加速
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class StockSequenceDataset(Dataset):
    """股票时序数据集"""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        sequence_length: int = 20,
        group_col: str = "code"
    ):
        """
        初始化数据集

        Args:
            data: 数据DataFrame
            feature_cols: 特征列
            label_col: 标签列
            sequence_length: 序列长度
            group_col: 分组列(股票代码)
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.label_col = label_col

        self.sequences = []
        self.labels = []
        self.dates = []
        self.codes = []

        # 按股票分组构建序列
        for code, group in data.groupby(group_col):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[feature_cols].values
            labels = group[label_col].values

            # 构建滑动窗口序列
            for i in range(sequence_length, len(group)):
                seq = features[i - sequence_length:i]
                label = labels[i]

                if not np.isnan(label) and not np.isnan(seq).any():
                    self.sequences.append(seq)
                    self.labels.append(label)
                    self.dates.append(group.iloc[i]["date"])
                    self.codes.append(code)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        logger.info(f"Created dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


class MultiHeadTemporalAttention(nn.Module):
    """多头时间注意力机制"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.norm(x)
        attn_out, attn_weights = self.attention(x, x, x)
        return residual + self.dropout(attn_out), attn_weights


class LSTMNet(nn.Module):
    """Bi-LSTM + Multi-Head Attention网络"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_attention_heads: int = 4
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # LSTM输出维度
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # 投影到attention维度
        self.attn_proj = nn.Linear(lstm_out_dim, hidden_size)

        # Multi-Head Temporal Attention
        self.temporal_attention = MultiHeadTemporalAttention(
            hidden_size, n_attention_heads, dropout
        )

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # Bi-LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        # 投影
        attn_input = self.attn_proj(lstm_out)  # (batch, seq_len, hidden_size)

        # Multi-Head Attention
        attn_out, _ = self.temporal_attention(attn_input)  # (batch, seq_len, hidden_size)

        # 加权聚合: 使用attention对时间步加权
        # 简单方案: 取最后时间步 + 全局均值池化的加权组合
        last_step = attn_out[:, -1]  # (batch, hidden_size)
        avg_pool = attn_out.mean(dim=1)  # (batch, hidden_size)
        out = last_step + 0.5 * avg_pool

        # 预测
        out = self.head(out)
        return out


class LSTMModel:
    """LSTM模型封装"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        model_config = config.get("models", {}).get("lstm", {})

        self.enabled = model_config.get("enabled", True)
        self.params = model_config.get("params", {})
        self.training_config = model_config.get("training", {})

        # 模型参数
        self.hidden_size = self.params.get("hidden_size", 128)
        self.num_layers = self.params.get("num_layers", 2)
        self.dropout = self.params.get("dropout", 0.2)
        self.bidirectional = self.params.get("bidirectional", True)
        self.n_attention_heads = self.params.get("n_attention_heads", 4)
        self.sequence_length = self.params.get("sequence_length", 20)

        # 训练参数
        self.batch_size = self.training_config.get("batch_size", 512)
        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.epochs = self.training_config.get("epochs", 100)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 10)

        # 设备选择：CUDA > MPS > CPU
        gpu_config = config.get("gpu", {})
        if gpu_config.get("enabled", True):
            if torch.cuda.is_available():
                self.device = torch.device(gpu_config.get("device", "cuda:0"))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Apple Silicon GPU
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # 混合精度训练（仅CUDA支持）
        self.use_mixed_precision = (
            gpu_config.get("mixed_precision", True) and
            torch.cuda.is_available()
        )

        self.model = None
        self.feature_cols = None

        # 模型保存路径
        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LSTMModel (Bi-LSTM+Attention) initialized. Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")

    def _create_model(self, input_size: int) -> LSTMNet:
        """
        创建模型

        Args:
            input_size: 输入特征数

        Returns:
            LSTM网络
        """
        model = LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            n_attention_heads=self.n_attention_heads
        )
        return model.to(self.device)

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank"
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_data: 训练数据
            valid_data: 验证数据
            feature_cols: 特征列
            label_col: 标签列

        Returns:
            训练结果
        """
        logger.info(f"Training LSTM model, train size: {len(train_data)}")

        self.feature_cols = feature_cols

        # 创建数据集
        train_dataset = StockSequenceDataset(
            train_data, feature_cols, label_col, self.sequence_length
        )
        valid_dataset = StockSequenceDataset(
            valid_data, feature_cols, label_col, self.sequence_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 创建模型
        self.model = self._create_model(len(feature_cols))

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler_config = self.training_config.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5)
        )

        criterion = nn.MSELoss()

        # 混合精度训练组件
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        # 训练循环
        best_valid_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        train_losses = []
        valid_losses = []

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # 混合精度前向传播
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            valid_loss = 0

            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            # 学习率调整
            scheduler.step(valid_loss)

            # 早停
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型
                self._save_checkpoint("best")
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # 加载最佳模型
        self._load_checkpoint("best")

        logger.info(f"Training completed. Best epoch: {best_epoch + 1}")

        return {
            "best_epoch": best_epoch + 1,
            "best_valid_loss": best_valid_loss,
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            data: 数据DataFrame

        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if self.feature_cols is None:
            raise ValueError("Feature columns not set!")

        self.model.eval()

        # 创建数据集(使用虚拟标签)
        data = data.copy()
        if "dummy_label" not in data.columns:
            data["dummy_label"] = 0

        dataset = StockSequenceDataset(
            data, self.feature_cols, "dummy_label", self.sequence_length
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

    def predict_with_dates(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, List, List]:
        """
        预测并返回日期和股票代码

        Args:
            data: 数据DataFrame

        Returns:
            (预测值, 日期列表, 股票代码列表)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        data = data.copy()
        if "dummy_label" not in data.columns:
            data["dummy_label"] = 0

        dataset = StockSequenceDataset(
            data, self.feature_cols, "dummy_label", self.sequence_length
        )

        predictions = self.predict(data)

        return predictions, dataset.dates, dataset.codes

    def _save_checkpoint(self, name: str):
        """保存检查点"""
        path = self.model_dir / f"lstm_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)

    def _load_checkpoint(self, name: str):
        """加载检查点"""
        path = self.model_dir / f"lstm_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.model = self._create_model(len(checkpoint["feature_cols"]))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_cols = checkpoint["feature_cols"]

    def save(self, name: str = "lstm_model"):
        """
        保存模型

        Args:
            name: 模型名称
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)

        logger.info(f"Model saved to {path}")

    def load(self, name: str = "lstm_model"):
        """
        加载模型

        Args:
            name: 模型名称
        """
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.feature_cols = checkpoint["feature_cols"]
        self.params = checkpoint.get("params", self.params)

        self.model = self._create_model(len(self.feature_cols))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {path}")
