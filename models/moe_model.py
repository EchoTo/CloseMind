"""
混合专家(Mixture of Experts)门控网络模型
基于 MIGA (2024) + Time-MoE (ICLR 2025 Spotlight)
不同市场状态(牛/熊/震荡)由不同专家处理,动态路由
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
        logger.info(f"Created MoE dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


class ExpertNetwork(nn.Module):
    """单个专家网络: 轻量级Transformer"""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class TopKGating(nn.Module):
    """
    Top-K门控网络
    根据输入动态选择最合适的K个专家
    包含负载均衡损失以避免专家坍塌
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 负载均衡相关
        self.register_buffer("expert_counts", torch.zeros(num_experts))

    def forward(self, x):
        """
        x: (batch, d_model) - 用于路由的表示
        Returns:
            gates: (batch, num_experts) - 门控权重
            load_balance_loss: scalar - 负载均衡损失
        """
        logits = self.gate(x)  # (batch, num_experts)

        # 训练时添加噪声,增强探索
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-K选择
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # 构建完整的门控权重矩阵
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # 负载均衡损失 (auxiliary loss-free inspired by DeepSeek-V3)
        # 鼓励所有专家被均匀使用
        expert_usage = gates.sum(dim=0)  # (num_experts,)
        target_usage = gates.sum() / self.num_experts
        load_balance_loss = ((expert_usage - target_usage) ** 2).mean()

        return gates, load_balance_loss


class MoELayer(nn.Module):
    """MoE层: 多个专家 + 门控"""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.experts = nn.ModuleList([
            ExpertNetwork(d_model, n_heads, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.gate = TopKGating(d_model, num_experts, top_k)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        # 用序列均值作为路由输入
        routing_input = x.mean(dim=1)  # (batch, d_model)

        # 获取门控权重
        gates, lb_loss = self.gate(routing_input)  # (batch, num_experts)

        # 专家计算
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (batch, num_experts, seq_len, d_model)

        # 加权组合
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # (batch, num_experts, 1, 1)
        output = (expert_outputs * gates).sum(dim=1)  # (batch, seq_len, d_model)

        return output, lb_loss


class MoEStockNet(nn.Module):
    """MoE股票预测网络"""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        n_moe_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 输入嵌入
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # 共享的Transformer层(底层)
        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(shared_layer, num_layers=2)

        # MoE层(上层)
        self.moe_layers = nn.ModuleList([
            MoELayer(d_model, num_experts, top_k, n_heads // 2, d_ff // 2, dropout)
            for _ in range(n_moe_layers)
        ])

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.num_experts = num_experts

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape

        # 输入嵌入 + 位置编码
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :seq_len]

        # 共享Transformer编码
        x = self.shared_encoder(x)

        # MoE层
        total_lb_loss = 0
        for moe_layer in self.moe_layers:
            x, lb_loss = moe_layer(x)
            total_lb_loss += lb_loss

        # 取最后时间步
        x_last = x[:, -1]  # (batch, d_model)

        # 预测
        out = self.head(x_last)  # (batch, 1)

        return out, total_lb_loss / len(self.moe_layers)


class MoEModel:
    """MoE模型封装"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_config = config.get("models", {}).get("moe", {})

        self.enabled = model_config.get("enabled", True)
        self.params = model_config.get("params", {})
        self.training_config = model_config.get("training", {})

        # 模型参数
        self.d_model = self.params.get("d_model", 256)
        self.num_experts = self.params.get("num_experts", 4)
        self.top_k = self.params.get("top_k", 2)
        self.n_moe_layers = self.params.get("n_moe_layers", 3)
        self.n_heads = self.params.get("n_heads", 8)
        self.d_ff = self.params.get("d_ff", 512)
        self.dropout = self.params.get("dropout", 0.2)
        self.sequence_length = self.params.get("sequence_length", 20)
        self.lb_loss_weight = self.params.get("lb_loss_weight", 0.01)

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

        logger.info(f"MoEModel initialized. Experts: {self.num_experts}, Top-K: {self.top_k}, Device: {self.device}")

    def _create_model(self, input_size: int) -> MoEStockNet:
        model = MoEStockNet(
            input_size=input_size,
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_moe_layers=self.n_moe_layers,
            n_heads=self.n_heads,
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
        logger.info(f"Training MoE model, train size: {len(train_data)}")

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
            train_lb_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs, lb_loss = self.model(batch_x)
                    pred_loss = criterion(outputs, batch_y)
                    loss = pred_loss + self.lb_loss_weight * lb_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += pred_loss.item()
                train_lb_loss += lb_loss.item()

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
                        outputs, _ = self.model(batch_x)
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
                avg_lb = train_lb_loss / len(train_loader)
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, "
                    f"LB Loss: {avg_lb:.6f}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self._load_checkpoint("best")
        logger.info(f"MoE training completed. Best epoch: {best_epoch + 1}")

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
                    outputs, _ = self.model(batch_x)
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
        path = self.model_dir / f"moe_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)

    def _load_checkpoint(self, name: str):
        path = self.model_dir / f"moe_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self._create_model(len(checkpoint["feature_cols"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_cols = checkpoint["feature_cols"]

    def save(self, name: str = "moe_model"):
        if self.model is None:
            logger.warning("No model to save")
            return
        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "params": self.params
        }, path)
        logger.info(f"MoE model saved to {path}")

    def load(self, name: str = "moe_model"):
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_cols = checkpoint["feature_cols"]
        self.params = checkpoint.get("params", self.params)
        self.model = self._create_model(len(self.feature_cols))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"MoE model loaded from {path}")