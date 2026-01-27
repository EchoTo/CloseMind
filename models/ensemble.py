"""
模型集成
支持7模型集成: LightGBM, XGBoost, Bi-LSTM+Attention, PatchTST, iTransformer, Mamba, MoE
支持加权平均和Stacking两种集成方式
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .lgb_model import LightGBMModel
from .xgb_model import XGBoostModel
from .lstm_model import LSTMModel
from .patchtst_model import PatchTSTModel
from .itransformer_model import iTransformerModel
from .mamba_model import MambaModel
from .moe_model import MoEModel


# 深度学习模型列表(使用序列数据接口)
DL_MODELS = {"lstm", "patchtst", "itransformer", "mamba", "moe"}

# 模型名称到类的映射
MODEL_CLASSES = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "patchtst": PatchTSTModel,
    "itransformer": iTransformerModel,
    "mamba": MambaModel,
    "moe": MoEModel,
}

# 模型名称到配置键的映射
MODEL_CONFIG_KEYS = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "lstm": "lstm",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "mamba": "mamba",
    "moe": "moe",
}

# 模型保存后缀
MODEL_SAVE_NAMES = {
    "lightgbm": "lgb",
    "xgboost": "xgb",
    "lstm": "lstm",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "mamba": "mamba",
    "moe": "moe",
}


class EnsembleModel:
    """模型集成"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ensemble_config = config.get("models", {}).get("ensemble", {})

        self.method = ensemble_config.get("method", "weighted_average")
        self.weights = ensemble_config.get("weights", {
            "lightgbm": 0.10,
            "xgboost": 0.10,
            "lstm": 0.05,
            "patchtst": 0.25,
            "itransformer": 0.20,
            "mamba": 0.15,
            "moe": 0.15,
        })
        self.stacking_config = ensemble_config.get("stacking", {})

        # 初始化所有子模型
        self.all_models = {}
        for name, cls in MODEL_CLASSES.items():
            self.all_models[name] = cls(config)

        # 已训练的可用模型
        self.models = {}

        # Stacking元学习器
        self.meta_learner = None
        self.scaler = None

        # 模型保存路径
        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 统计启用的模型
        enabled = [name for name, m in self.all_models.items() if m.enabled]
        logger.info(f"EnsembleModel initialized. Method: {self.method}, Enabled models: {enabled}")

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank",
        group_col: str = "date"
    ) -> Dict[str, Any]:
        logger.info("Training ensemble model...")

        results = {}

        # 准备表格模型的数据
        X_train = train_data[feature_cols]
        y_train = train_data[label_col]
        group_train = train_data[group_col]

        X_valid = valid_data[feature_cols]
        y_valid = valid_data[label_col]
        group_valid = valid_data[group_col]

        # ========== 训练表格模型 ==========
        for name in ["lightgbm", "xgboost"]:
            model = self.all_models[name]
            if model.enabled:
                logger.info(f"Training {name}...")
                result = model.train(
                    X_train, y_train, group_train,
                    X_valid, y_valid, group_valid
                )
                results[name] = result
                self.models[name] = model

        # ========== 训练深度学习模型 ==========
        for name in ["lstm", "patchtst", "itransformer", "mamba", "moe"]:
            model = self.all_models[name]
            if model.enabled:
                logger.info(f"Training {name}...")
                result = model.train(
                    train_data, valid_data, feature_cols, label_col
                )
                results[name] = result
                self.models[name] = model

        # ========== 训练Stacking元学习器 ==========
        if self.method == "stacking" and len(self.models) > 1:
            logger.info("Training meta-learner for stacking...")
            self._train_meta_learner(valid_data, feature_cols, label_col)

        # ========== 计算集成IC ==========
        ensemble_pred = self.predict(valid_data, feature_cols)
        if len(ensemble_pred) == len(y_valid):
            ensemble_ic = np.corrcoef(ensemble_pred, y_valid)[0, 1]
            results["ensemble_ic"] = ensemble_ic
            logger.info(f"Ensemble IC on valid set: {ensemble_ic:.4f}")

        return results

    def _get_model_predictions(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """获取所有可用模型的预测"""
        X = data[feature_cols]
        predictions = {}
        target_len = len(X)

        for name, model in self.models.items():
            try:
                if name in DL_MODELS:
                    pred, dates, codes = model.predict_with_dates(data)
                    if len(pred) == target_len:
                        predictions[name] = pred
                    else:
                        logger.warning(
                            f"{name} prediction length mismatch: {len(pred)} vs {target_len}, skipping"
                        )
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {e}")

        return predictions

    def _train_meta_learner(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ):
        predictions = self._get_model_predictions(data, feature_cols)
        y = data[label_col]

        if len(predictions) < 2:
            logger.warning("Not enough models for stacking, falling back to weighted average")
            self.method = "weighted_average"
            return

        # 保存模型顺序
        self._meta_model_order = sorted(predictions.keys())

        meta_features = np.column_stack(
            [predictions[name] for name in self._meta_model_order]
        )

        self.scaler = StandardScaler()
        meta_features = self.scaler.fit_transform(meta_features)

        meta_type = self.stacking_config.get("meta_learner", "ridge")
        if meta_type == "ridge":
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = LinearRegression()

        valid_mask = ~np.isnan(y.values) & ~np.isnan(meta_features).any(axis=1)
        self.meta_learner.fit(meta_features[valid_mask], y.values[valid_mask])

        logger.info(
            f"Meta-learner trained. Models: {self._meta_model_order}, "
            f"Coefficients: {dict(zip(self._meta_model_order, self.meta_learner.coef_))}"
        )

    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        predictions = self._get_model_predictions(data, feature_cols)

        if len(predictions) == 0:
            raise ValueError("No model available for prediction!")

        if self.method == "weighted_average":
            return self._weighted_average_predict(predictions)
        elif self.method == "stacking":
            return self._stacking_predict(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _weighted_average_predict(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        available_models = list(predictions.keys())
        weights = {k: self.weights.get(k, 1.0 / len(available_models)) for k in available_models}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        result = np.zeros(len(list(predictions.values())[0]))
        for model_name, pred in predictions.items():
            result += weights[model_name] * pred

        return result

    def _stacking_predict(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.meta_learner is None:
            logger.warning("Meta-learner not trained, falling back to weighted average")
            return self._weighted_average_predict(predictions)

        meta_features = []
        for model_name in self._meta_model_order:
            if model_name in predictions:
                meta_features.append(predictions[model_name])
            else:
                logger.warning(f"{model_name} not available for stacking, using zeros")
                meta_features.append(np.zeros_like(list(predictions.values())[0]))

        meta_features = np.column_stack(meta_features)

        if self.scaler is not None:
            meta_features = self.scaler.transform(meta_features)

        return self.meta_learner.predict(meta_features)

    def predict_with_details(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        predictions = self._get_model_predictions(data, feature_cols)
        results = dict(predictions)
        results["ensemble"] = self.predict(data, feature_cols)
        return results

    def get_model_weights(self) -> Dict[str, float]:
        if self.method == "stacking" and self.meta_learner is not None:
            return dict(zip(self._meta_model_order, self.meta_learner.coef_))
        else:
            available = [m for m in self.weights if m in self.models]
            weights = {k: self.weights.get(k, 0) for k in available}
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            return weights

    def save(self, name: str = "ensemble"):
        # 保存各子模型
        for model_name, model in self.models.items():
            suffix = MODEL_SAVE_NAMES[model_name]
            model.save(f"{name}_{suffix}")

        # 保存集成配置
        meta_path = self.model_dir / f"{name}_ensemble_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "weights": self.weights,
                "available_models": list(self.models.keys()),
                "meta_learner": self.meta_learner,
                "scaler": self.scaler,
                "meta_model_order": getattr(self, "_meta_model_order", None),
            }, f)

        logger.info(f"Ensemble model saved. Models: {list(self.models.keys())}")

    def load(self, name: str = "ensemble"):
        meta_path = self.model_dir / f"{name}_ensemble_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble meta not found: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.method = meta["method"]
        self.weights = meta["weights"]
        self.meta_learner = meta.get("meta_learner")
        self.scaler = meta.get("scaler")
        self._meta_model_order = meta.get("meta_model_order")
        available_models = meta["available_models"]

        for model_name in available_models:
            suffix = MODEL_SAVE_NAMES[model_name]
            model = self.all_models[model_name]
            model.load(f"{name}_{suffix}")
            self.models[model_name] = model

        logger.info(f"Ensemble model loaded. Available models: {available_models}")

    def evaluate_models(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank"
    ) -> pd.DataFrame:
        X = data[feature_cols]
        y = data[label_col]

        results = []
        predictions = self._get_model_predictions(data, feature_cols)

        for model_name, pred in predictions.items():
            if len(pred) == len(y):
                ic = np.corrcoef(pred, y)[0, 1]
                results.append({"model": model_name, "IC": ic})

        # Ensemble
        ensemble_pred = self.predict(data, feature_cols)
        if len(ensemble_pred) == len(y):
            ic = np.corrcoef(ensemble_pred, y)[0, 1]
            results.append({"model": "Ensemble", "IC": ic})

        return pd.DataFrame(results)