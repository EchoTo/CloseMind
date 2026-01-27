"""
iTransformer模型
基于 "Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)
将变量维度作为token(而非时间维度),天然适合多因子股票预测场景
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        self.dates = []
        self.codes = []

        for code, group in data.groupby(group_col):
            group = group.sort_values("date").reset_index(drop=True)
            features = group[feature_cols].values
            labels = group[label_col].values

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
        logger.info(f"Created iTransformer dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


class iTransformerNet(nn.Module):
    """
    iTransformer网络
    核心思想: 将变量(因子)作为token,时间序列作为token的嵌入
    这样attention机制直接建模因子间的交互关系
    """

    def __init__(
        self,
        n_vars: int,
        seq_len: int = 20,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.seq_len = seq_len

        # 变量嵌入: 将每个变量的时间序列映射到d_model维
        self.var_embedding = nn.Linear(seq_len, d_model)
        self.var_norm = nn.LayerNorm(d_model)

        # 可学习的变量位置编码
        self.var_pos = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)

        # CLS token用于聚合
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer Encoder - 建模变量间交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 时间聚合模块: 使用1D卷积进一步提取时间特征
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_vars, n_vars, kernel_size=3, padding=1, groups=n_vars),
            nn.GELU(),
            nn.Conv1d(n_vars, n_vars, kernel_size=1),
        )

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch = x.shape[0]

        # 反转: 变量维度作为token维度
        # (batch, seq_len, n_vars) -> (batch, n_vars, seq_len)
        x_inv = x.transpose(1, 2)

        # 时间卷积增强
        x_conv = self.temporal_conv(x_inv)
        x_inv = x_inv + x_conv

        # 变量嵌入: (batch, n_vars, seq_len) -> (batch, n_vars, d_model)
        x_emb = self.var_embedding(x_inv)
        x_emb = self.var_norm(x_emb)

        # 加变量位置编码
        x_emb = x_emb + self.var_pos

        # 添加CLS token
        cls = self.cls_token.expand(batch, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)  # (batch, n_vars+1, d_model)

        # Transformer编码 - attention建模变量间关系
        x_enc = self.encoder(x_emb)  # (batch, n_vars+1, d_model)

        # 使用CLS token输出
        cls_out = x_enc[:, 0]  # (batch, d_model)

        # 预测
        out = self.head(cls_out)  # (batch, 1)
        return out


class iTransformerModel:
    """iTransformer模型封装"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_config = config.get("models", {}).get("itransformer", {})

        self.enabled = model_config.get("enabled", True)
        self.params = model_config.get("params", {})
        self.training_config = model_config.get("training", {})

        # 模型参数
        self.d_model = self.params.get("d_model", 256)
        self.n_heads = self.params.get("n_heads", 8)
        self.n_layers = self.params.get("n_layers", 4)
        self.d_ff = self.params.get("d_ff", 512)
        self.dropout = self.params.get("dropout", 0.2)
        self.sequence_length = self.params.get("sequence_length", 20)

        # 训练参数
        self.batch_size = self.training_config.get("batch_size", 256)
        self.learning_rate = self.training_config.get("learning_rate", 1e-4)
        self.epochs = self.training_config.get("epochs", 100)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        self.weight_decay = self.training_config.get("weight_decay", 1e-4)

        # 设备
        gpu_config = config.get("gpu", {})
        if gpu_config.get("enabled", True) and torch.cuda.is_available():
            self.device = torch.device(gpu_config.get("device", "cuda:0"))
        else:
            self.device = torch.device("cpu")

        self.use_mixed_precision = (
            gpu_config.get("mixed_precision", True) and torch.cuda.is_available()
        )

        self.model = None
        self.feature_cols = None

        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"iTransformerModel initialized. Device: {self.device}")

    def _create_model(self, input_size: int) -> iTransformerNet:
        model = iTransformerNet(
            n_vars=input_size,
            seq_len=self.sequence_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank"
    ) -> Dict[str, Any]:
        logger.info(f"Training iTransformer model, train size: {len(train_data)}")

        self.feature_cols = feature_cols

        train_dataset = StockSequenceDataset(
            train_data, feature_cols, label_col, self.sequence_length
        )
        valid_dataset = StockSequenceDataset(
            valid_data, feature_cols, label_col, self.sequence_length
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        self.model = self._create_model(len(feature_cols))

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        best_valid_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        train_losses = []
        valid_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            scheduler.step()

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

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                patience_counter = 0
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

        self._load_checkpoint("best")
        logger.info(f"iTransformer training completed. Best epoch: {best_epoch + 1}")

        return {
            "best_epoch": best_epoch + 1,
            "best_valid_loss": best_valid_loss,
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if self.feature_cols is None:
            raise ValueError("Feature columns not set!")

        self.model.eval()
        data = data.copy()
        if "dummy_label" not in data.columns:
            data["dummy_label"] = 0

        dataset = StockSequenceDataset(
            data, self.feature_cols, "dummy_label", self.sequence_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

    def predict_with_dates(self, data: pd.DataFrame) -> Tuple[np.ndarray, List, List]:
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
        path = self.model_dir / f"itransformer_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)

    def _load_checkpoint(self, name: str):
        path = self.model_dir / f"itransformer_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self._create_model(len(checkpoint["feature_cols"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_cols = checkpoint["feature_cols"]

    def save(self, name: str = "itransformer_model"):
        if self.model is None:
            logger.warning("No model to save")
            return
        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)
        logger.info(f"iTransformer model saved to {path}")

    def load(self, name: str = "itransformer_model"):
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_cols = checkpoint["feature_cols"]
        self.params = checkpoint.get("params", self.params)
        self.model = self._create_model(len(self.feature_cols))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"iTransformer model loaded from {path}")