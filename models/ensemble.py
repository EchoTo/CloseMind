"""
模型集成
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


class EnsembleModel:
    """模型集成"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        ensemble_config = config.get("models", {}).get("ensemble", {})

        self.method = ensemble_config.get("method", "weighted_average")
        self.weights = ensemble_config.get("weights", {
            "lightgbm": 0.4,
            "xgboost": 0.4,
            "lstm": 0.2
        })
        self.stacking_config = ensemble_config.get("stacking", {})

        # 初始化子模型
        self.models = {}
        self.lgb_model = LightGBMModel(config)
        self.xgb_model = XGBoostModel(config)
        self.lstm_model = LSTMModel(config)

        # Stacking元学习器
        self.meta_learner = None
        self.scaler = None

        # 模型保存路径
        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EnsembleModel initialized. Method: {self.method}")

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank",
        group_col: str = "date"
    ) -> Dict[str, Any]:
        """
        训练集成模型

        Args:
            train_data: 训练数据
            valid_data: 验证数据
            feature_cols: 特征列
            label_col: 标签列
            group_col: 分组列

        Returns:
            训练结果
        """
        logger.info("Training ensemble model...")

        results = {}

        # 准备数据
        X_train = train_data[feature_cols]
        y_train = train_data[label_col]
        group_train = train_data[group_col]

        X_valid = valid_data[feature_cols]
        y_valid = valid_data[label_col]
        group_valid = valid_data[group_col]

        # 训练LightGBM
        if self.lgb_model.enabled:
            logger.info("Training LightGBM...")
            lgb_result = self.lgb_model.train(
                X_train, y_train, group_train,
                X_valid, y_valid, group_valid
            )
            results["lightgbm"] = lgb_result
            self.models["lightgbm"] = self.lgb_model

        # 训练XGBoost
        if self.xgb_model.enabled:
            logger.info("Training XGBoost...")
            xgb_result = self.xgb_model.train(
                X_train, y_train, group_train,
                X_valid, y_valid, group_valid
            )
            results["xgboost"] = xgb_result
            self.models["xgboost"] = self.xgb_model

        # 训练LSTM
        if self.lstm_model.enabled:
            logger.info("Training LSTM...")
            # LSTM需要完整数据
            lstm_result = self.lstm_model.train(
                train_data, valid_data, feature_cols, label_col
            )
            results["lstm"] = lstm_result
            self.models["lstm"] = self.lstm_model

        # 如果使用Stacking,训练元学习器
        if self.method == "stacking" and len(self.models) > 1:
            logger.info("Training meta-learner for stacking...")
            self._train_meta_learner(valid_data, feature_cols, label_col)

        # 计算集成模型在验证集上的IC
        ensemble_pred = self.predict(valid_data, feature_cols)
        if len(ensemble_pred) == len(y_valid):
            ensemble_ic = np.corrcoef(ensemble_pred, y_valid)[0, 1]
            results["ensemble_ic"] = ensemble_ic
            logger.info(f"Ensemble IC on valid set: {ensemble_ic:.4f}")

        return results

    def _train_meta_learner(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ):
        """
        训练Stacking元学习器

        Args:
            data: 数据
            feature_cols: 特征列
            label_col: 标签列
        """
        # 获取各模型预测
        predictions = []

        X = data[feature_cols]
        y = data[label_col]

        if "lightgbm" in self.models:
            pred_lgb = self.lgb_model.predict(X)
            predictions.append(pred_lgb)

        if "xgboost" in self.models:
            pred_xgb = self.xgb_model.predict(X)
            predictions.append(pred_xgb)

        if "lstm" in self.models:
            pred_lstm, dates, codes = self.lstm_model.predict_with_dates(data)
            # LSTM预测长度可能不同,需要对齐
            # 这里简化处理,仅使用LGB和XGB
            if len(pred_lstm) == len(X):
                predictions.append(pred_lstm)

        if len(predictions) < 2:
            logger.warning("Not enough models for stacking, falling back to weighted average")
            self.method = "weighted_average"
            return

        # 构建元特征
        meta_features = np.column_stack(predictions)

        # 标准化
        self.scaler = StandardScaler()
        meta_features = self.scaler.fit_transform(meta_features)

        # 训练元学习器
        meta_type = self.stacking_config.get("meta_learner", "ridge")
        if meta_type == "ridge":
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = LinearRegression()

        # 处理标签中的nan
        valid_mask = ~np.isnan(y.values) & ~np.isnan(meta_features).any(axis=1)
        self.meta_learner.fit(meta_features[valid_mask], y.values[valid_mask])

        logger.info(f"Meta-learner trained. Coefficients: {self.meta_learner.coef_}")

    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        集成预测

        Args:
            data: 数据
            feature_cols: 特征列

        Returns:
            预测值
        """
        X = data[feature_cols]
        predictions = {}

        # 获取各模型预测
        if "lightgbm" in self.models:
            predictions["lightgbm"] = self.lgb_model.predict(X)

        if "xgboost" in self.models:
            predictions["xgboost"] = self.xgb_model.predict(X)

        if "lstm" in self.models:
            pred_lstm, dates, codes = self.lstm_model.predict_with_dates(data)
            if len(pred_lstm) == len(X):
                predictions["lstm"] = pred_lstm

        if len(predictions) == 0:
            raise ValueError("No model available for prediction!")

        # 集成
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
        """
        加权平均预测

        Args:
            predictions: 各模型预测

        Returns:
            加权预测值
        """
        # 归一化权重
        available_models = list(predictions.keys())
        weights = {k: self.weights.get(k, 1.0) for k in available_models}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # 加权平均
        result = np.zeros(len(list(predictions.values())[0]))
        for model_name, pred in predictions.items():
            result += weights[model_name] * pred

        return result

    def _stacking_predict(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Stacking预测

        Args:
            predictions: 各模型预测

        Returns:
            Stacking预测值
        """
        if self.meta_learner is None:
            logger.warning("Meta-learner not trained, falling back to weighted average")
            return self._weighted_average_predict(predictions)

        # 构建元特征(保持顺序一致)
        model_order = ["lightgbm", "xgboost", "lstm"]
        meta_features = []
        for model_name in model_order:
            if model_name in predictions:
                meta_features.append(predictions[model_name])

        meta_features = np.column_stack(meta_features)

        # 标准化
        if self.scaler is not None:
            meta_features = self.scaler.transform(meta_features)

        return self.meta_learner.predict(meta_features)

    def predict_with_details(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        预测并返回各模型详细结果

        Args:
            data: 数据
            feature_cols: 特征列

        Returns:
            包含各模型预测和集成预测的字典
        """
        X = data[feature_cols]
        results = {}

        # 获取各模型预测
        if "lightgbm" in self.models:
            results["lightgbm"] = self.lgb_model.predict(X)

        if "xgboost" in self.models:
            results["xgboost"] = self.xgb_model.predict(X)

        if "lstm" in self.models:
            pred_lstm, dates, codes = self.lstm_model.predict_with_dates(data)
            if len(pred_lstm) == len(X):
                results["lstm"] = pred_lstm

        # 集成预测
        results["ensemble"] = self.predict(data, feature_cols)

        return results

    def get_model_weights(self) -> Dict[str, float]:
        """
        获取模型权重

        Returns:
            权重字典
        """
        if self.method == "stacking" and self.meta_learner is not None:
            model_order = ["lightgbm", "xgboost", "lstm"]
            available = [m for m in model_order if m in self.models]
            return dict(zip(available, self.meta_learner.coef_))
        else:
            return self.weights

    def save(self, name: str = "ensemble"):
        """
        保存集成模型

        Args:
            name: 模型名称
        """
        # 保存子模型
        if "lightgbm" in self.models:
            self.lgb_model.save(f"{name}_lgb")

        if "xgboost" in self.models:
            self.xgb_model.save(f"{name}_xgb")

        if "lstm" in self.models:
            self.lstm_model.save(f"{name}_lstm")

        # 保存集成配置
        meta_path = self.model_dir / f"{name}_ensemble_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "weights": self.weights,
                "available_models": list(self.models.keys()),
                "meta_learner": self.meta_learner,
                "scaler": self.scaler
            }, f)

        logger.info(f"Ensemble model saved with prefix: {name}")

    def load(self, name: str = "ensemble"):
        """
        加载集成模型

        Args:
            name: 模型名称
        """
        # 加载集成配置
        meta_path = self.model_dir / f"{name}_ensemble_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble meta not found: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.method = meta["method"]
        self.weights = meta["weights"]
        self.meta_learner = meta.get("meta_learner")
        self.scaler = meta.get("scaler")
        available_models = meta["available_models"]

        # 加载子模型
        if "lightgbm" in available_models:
            self.lgb_model.load(f"{name}_lgb")
            self.models["lightgbm"] = self.lgb_model

        if "xgboost" in available_models:
            self.xgb_model.load(f"{name}_xgb")
            self.models["xgboost"] = self.xgb_model

        if "lstm" in available_models:
            self.lstm_model.load(f"{name}_lstm")
            self.models["lstm"] = self.lstm_model

        logger.info(f"Ensemble model loaded. Available models: {available_models}")

    def evaluate_models(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "return_1d_rank"
    ) -> pd.DataFrame:
        """
        评估各模型性能

        Args:
            data: 数据
            feature_cols: 特征列
            label_col: 标签列

        Returns:
            评估结果DataFrame
        """
        X = data[feature_cols]
        y = data[label_col]

        results = []

        # LightGBM
        if "lightgbm" in self.models:
            pred = self.lgb_model.predict(X)
            ic = np.corrcoef(pred, y)[0, 1]
            results.append({"model": "LightGBM", "IC": ic})

        # XGBoost
        if "xgboost" in self.models:
            pred = self.xgb_model.predict(X)
            ic = np.corrcoef(pred, y)[0, 1]
            results.append({"model": "XGBoost", "IC": ic})

        # LSTM
        if "lstm" in self.models:
            pred, _, _ = self.lstm_model.predict_with_dates(data)
            if len(pred) == len(y):
                ic = np.corrcoef(pred, y)[0, 1]
                results.append({"model": "LSTM", "IC": ic})

        # Ensemble
        pred = self.predict(data, feature_cols)
        if len(pred) == len(y):
            ic = np.corrcoef(pred, y)[0, 1]
            results.append({"model": "Ensemble", "IC": ic})

        return pd.DataFrame(results)
