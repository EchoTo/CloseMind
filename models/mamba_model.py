"""
Mamba状态空间模型
基于 MambaStock (2024) + MambaTS (ICLR 2025)
线性复杂度处理长序列,选择性状态空间机制过滤金融噪声
纯PyTorch实现,不依赖mamba-ssm库
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
        logger.info(f"Created Mamba dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型 (S6)
    Mamba的核心: 输入依赖的选择机制,使模型能够选择性地传播或遗忘信息
    纯PyTorch实现,兼容CPU和任意GPU
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D因果卷积
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM参数 - 输入依赖(选择性)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A参数: 离散化后用于状态转移
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D参数: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # 输入投影得到 x 和 z (gate)
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_in, z = xz.chunk(2, dim=-1)  # 各 (batch, seq_len, d_inner)

        # 1D因果卷积
        x_conv = x_in.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # 截断为因果
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)

        # 选择性SSM参数(输入依赖)
        x_ssm = self.x_proj(x_conv)  # (batch, seq_len, d_state*2 + 1)
        B = x_ssm[:, :, :self.d_state]  # (batch, seq_len, d_state)
        C = x_ssm[:, :, self.d_state:self.d_state * 2]  # (batch, seq_len, d_state)
        dt = F.softplus(x_ssm[:, :, -1])  # (batch, seq_len) - 时间步长

        # 离散化A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 选择性扫描 (sequential scan)
        y = self._selective_scan(x_conv, dt, A, B, C)

        # skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate
        y = y * F.silu(z)

        # 输出投影
        y = self.out_proj(y)

        return y + residual

    def _selective_scan(self, x, dt, A, B, C):
        """
        选择性扫描算法
        x: (batch, seq_len, d_inner)
        dt: (batch, seq_len)
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state)
        C: (batch, seq_len, d_state)
        """
        batch, seq_len, d_inner = x.shape

        # 初始化隐藏状态
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)

            # 离散化
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (batch, d_inner, d_state)
            dB = dt_t * B[:, t].unsqueeze(1)  # (batch, d_inner, d_state) broadcast

            # 状态更新: h = dA * h + dB * x
            x_t = x[:, t].unsqueeze(-1)  # (batch, d_inner, 1)
            h = dA * h + dB * x_t

            # 输出: y = C * h
            C_t = C[:, t].unsqueeze(1)  # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)  # (batch, d_inner)

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """Mamba Block = SelectiveSSM + FFN"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.ssm(x)
        x = x + self.ffn(x)
        return x


class MambaStockNet(nn.Module):
    """MambaStock网络"""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 输入嵌入
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # 双向: 反向Mamba
        self.reverse_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
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
        # x: (batch, seq_len, input_size)

        # 输入嵌入
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 正向Mamba
        fwd = x
        for layer in self.layers:
            fwd = layer(fwd)

        # 反向Mamba
        rev = x.flip(dims=[1])
        for layer in self.reverse_layers:
            rev = layer(rev)
        rev = rev.flip(dims=[1])

        # 双向融合(取最后时间步)
        fwd_last = fwd[:, -1]  # (batch, d_model)
        rev_last = rev[:, 0]   # (batch, d_model)
        merged = torch.cat([fwd_last, rev_last], dim=-1)  # (batch, d_model*2)
        merged = self.fusion(merged)  # (batch, d_model)

        # 预测
        out = self.head(merged)  # (batch, 1)
        return out


class MambaModel:
    """Mamba模型封装"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_config = config.get("models", {}).get("mamba", {})

        self.enabled = model_config.get("enabled", True)
        self.params = model_config.get("params", {})
        self.training_config = model_config.get("training", {})

        # 模型参数
        self.d_model = self.params.get("d_model", 256)
        self.d_state = self.params.get("d_state", 16)
        self.d_conv = self.params.get("d_conv", 4)
        self.expand = self.params.get("expand", 2)
        self.n_layers = self.params.get("n_layers", 4)
        self.dropout = self.params.get("dropout", 0.2)
        self.sequence_length = self.params.get("sequence_length", 20)

        # 训练参数
        self.batch_size = self.training_config.get("batch_size", 256)
        self.learning_rate = self.training_config.get("learning_rate", 1e-4)
        self.epochs = self.training_config.get("epochs", 100)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        self.weight_decay = self.training_config.get("weight_decay", 1e-4)

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

        self.use_mixed_precision = (
            gpu_config.get("mixed_precision", True) and torch.cuda.is_available()
        )

        self.model = None
        self.feature_cols = None

        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MambaModel initialized. Device: {self.device}")

    def _create_model(self, input_size: int) -> MambaStockNet:
        model = MambaStockNet(
            input_size=input_size,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            n_layers=self.n_layers,
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
        logger.info(f"Training Mamba model, train size: {len(train_data)}")

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
        logger.info(f"Mamba training completed. Best epoch: {best_epoch + 1}")

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
        path = self.model_dir / f"mamba_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)

    def _load_checkpoint(self, name: str):
        path = self.model_dir / f"mamba_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self._create_model(len(checkpoint["feature_cols"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_cols = checkpoint["feature_cols"]

    def save(self, name: str = "mamba_model"):
        if self.model is None:
            logger.warning("No model to save")
            return
        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)
        logger.info(f"Mamba model saved to {path}")

    def load(self, name: str = "mamba_model"):
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_cols = checkpoint["feature_cols"]
        self.params = checkpoint.get("params", self.params)
        self.model = self._create_model(len(self.feature_cols))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Mamba model loaded from {path}")