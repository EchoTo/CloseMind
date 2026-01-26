"""
LightGBM模型
支持排序学习和回归任务
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from loguru import logger


class LightGBMModel:
    """LightGBM模型封装"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        model_config = config.get("models", {}).get("lightgbm", {})

        self.enabled = model_config.get("enabled", True)
        self.params = model_config.get("params", {})
        self.model = None
        self.feature_names = None

        # 设置默认参数
        self._set_default_params()

        # 模型保存路径
        paths = config.get("paths", {})
        self.model_dir = Path(paths.get("model_dir", "./checkpoints"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("LightGBMModel initialized")

    def _set_default_params(self):
        """设置默认参数"""
        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 100,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "max_depth": 8,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            "verbose": -1,
            "ndcg_eval_at": [1, 3, 5, 10],
        }

        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: pd.Series
    ) -> Tuple[lgb.Dataset, np.ndarray]:
        """
        准备LightGBM数据集

        Args:
            X: 特征
            y: 标签
            group: 分组(日期)

        Returns:
            LightGBM Dataset和group数组
        """
        # 计算每个group的样本数
        group_counts = group.value_counts().sort_index().values

        # 创建Dataset
        dataset = lgb.Dataset(
            X,
            label=y,
            group=group_counts,
            feature_name=list(X.columns) if hasattr(X, 'columns') else None
        )

        return dataset, group_counts

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        group_valid: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            group_train: 训练分组
            X_valid: 验证特征
            y_valid: 验证标签
            group_valid: 验证分组

        Returns:
            训练结果字典
        """
        logger.info(f"Training LightGBM model, train size: {len(X_train)}")

        self.feature_names = list(X_train.columns)

        # 准备训练数据
        train_data, train_groups = self._prepare_data(X_train, y_train, group_train)

        # 准备验证数据
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None and group_valid is not None:
            valid_data, valid_groups = self._prepare_data(X_valid, y_valid, group_valid)
            valid_sets.append(valid_data)
            valid_names.append("valid")
            logger.info(f"Valid size: {len(X_valid)}")

        # 提取训练参数
        n_estimators = self.params.pop("n_estimators", 500)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # 训练
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds)
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # 恢复参数
        self.params["n_estimators"] = n_estimators
        self.params["early_stopping_rounds"] = early_stopping_rounds

        # 获取最佳迭代次数
        best_iteration = self.model.best_iteration

        logger.info(f"Training completed. Best iteration: {best_iteration}")

        return {
            "best_iteration": best_iteration,
            "feature_importance": self.get_feature_importance()
        }

    def train_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        训练回归模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签

        Returns:
            训练结果字典
        """
        logger.info(f"Training LightGBM regression model, train size: {len(X_train)}")

        self.feature_names = list(X_train.columns)

        # 修改参数为回归任务
        params = self.params.copy()
        params["objective"] = "regression"
        params["metric"] = "rmse"

        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        # 提取训练参数
        n_estimators = params.pop("n_estimators", 500)
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        # 训练
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds)
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")

        return {
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.get_feature_importance()
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            X: 特征

        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")
        feature_names = self.feature_names or [f"f{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, name: str = "lgb_model"):
        """
        保存模型

        Args:
            name: 模型名称
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = self.model_dir / f"{name}.txt"
        self.model.save_model(str(model_path))

        # 保存特征名
        meta_path = self.model_dir / f"{name}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({"feature_names": self.feature_names}, f)

        logger.info(f"Model saved to {model_path}")

    def load(self, name: str = "lgb_model"):
        """
        加载模型

        Args:
            name: 模型名称
        """
        model_path = self.model_dir / f"{name}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = lgb.Booster(model_file=str(model_path))

        # 加载特征名
        meta_path = self.model_dir / f"{name}_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_names = meta.get("feature_names")

        logger.info(f"Model loaded from {model_path}")

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        交叉验证

        Args:
            X: 特征
            y: 标签
            group: 分组
            n_splits: 折数

        Returns:
            交叉验证结果
        """
        logger.info(f"Running {n_splits}-fold cross validation...")

        # 使用GroupKFold确保同一天的数据在同一折
        unique_dates = group.unique()
        kfold = GroupKFold(n_splits=n_splits)

        results = []
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y, group)):
            logger.info(f"Fold {fold + 1}/{n_splits}")

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            group_train = group.iloc[train_idx]

            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]
            group_valid = group.iloc[valid_idx]

            # 训练
            fold_result = self.train(
                X_train, y_train, group_train,
                X_valid, y_valid, group_valid
            )

            # 预测
            pred_valid = self.predict(X_valid)

            # 计算IC
            ic = np.corrcoef(pred_valid, y_valid)[0, 1]

            results.append({
                "fold": fold + 1,
                "best_iteration": fold_result["best_iteration"],
                "ic": ic
            })

        # 汇总结果
        mean_ic = np.mean([r["ic"] for r in results])
        std_ic = np.std([r["ic"] for r in results])

        logger.info(f"CV Results: IC={mean_ic:.4f}±{std_ic:.4f}")

        return {
            "fold_results": results,
            "mean_ic": mean_ic,
            "std_ic": std_ic
        }
