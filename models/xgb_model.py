"""
XGBoost模型
支持排序学习和回归任务
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from loguru import logger


class XGBoostModel:
    """XGBoost模型封装"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        model_config = config.get("models", {}).get("xgboost", {})

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

        logger.info("XGBoostModel initialized")

    def _set_default_params(self):
        """设置默认参数"""
        default_params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "booster": "gbtree",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 100,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            "tree_method": "hist",  # 默认使用hist,如有GPU可改为gpu_hist
            "verbosity": 1,
        }

        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

    def _prepare_group(self, group: pd.Series) -> np.ndarray:
        """
        准备group数组

        Args:
            group: 分组Series

        Returns:
            每个group的大小数组
        """
        return group.value_counts().sort_index().values

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
        训练排序模型

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
        logger.info(f"Training XGBoost model, train size: {len(X_train)}")

        self.feature_names = list(X_train.columns)

        # 准备训练数据
        train_group = self._prepare_group(group_train)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_group)

        # 准备验证数据
        evals = [(dtrain, "train")]

        if X_valid is not None and y_valid is not None and group_valid is not None:
            valid_group = self._prepare_group(group_valid)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            dvalid.set_group(valid_group)
            evals.append((dvalid, "valid"))
            logger.info(f"Valid size: {len(X_valid)}")

        # 提取训练参数
        params = {k: v for k, v in self.params.items()
                  if k not in ["n_estimators", "early_stopping_rounds"]}
        n_estimators = self.params.get("n_estimators", 500)
        early_stopping_rounds = self.params.get("early_stopping_rounds", 50)

        # 训练
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )

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
        logger.info(f"Training XGBoost regression model, train size: {len(X_train)}")

        self.feature_names = list(X_train.columns)

        # 修改参数为回归任务
        params = {k: v for k, v in self.params.items()
                  if k not in ["n_estimators", "early_stopping_rounds"]}
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

        # 创建数据集
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        n_estimators = self.params.get("n_estimators", 500)
        early_stopping_rounds = self.params.get("early_stopping_rounds", 50)

        # 训练
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
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

        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration))

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.get_score(importance_type="gain")

        # 确保所有特征都有值
        feature_names = self.feature_names or list(importance.keys())
        importance_values = [importance.get(f, 0) for f in feature_names]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, name: str = "xgb_model"):
        """
        保存模型

        Args:
            name: 模型名称
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = self.model_dir / f"{name}.json"
        self.model.save_model(str(model_path))

        # 保存特征名
        meta_path = self.model_dir / f"{name}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({"feature_names": self.feature_names}, f)

        logger.info(f"Model saved to {model_path}")

    def load(self, name: str = "xgb_model"):
        """
        加载模型

        Args:
            name: 模型名称
        """
        model_path = self.model_dir / f"{name}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

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
