#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练脚本
训练LightGBM、XGBoost、LSTM模型
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import pandas as pd
from loguru import logger

from data import DataProcessor
from data.qlib_converter import SimpleDataLoader
from features import FeatureEngine
from models import LightGBMModel, XGBoostModel, LSTMModel, EnsembleModel
from backtest import BacktestEvaluator
from report import ReportGenerator


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "train_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO"
    )


def prepare_features(config: dict) -> pd.DataFrame:
    """准备特征数据"""
    logger.info("Loading and preparing features...")

    # 加载处理后的数据
    loader = SimpleDataLoader(config)
    df = loader.load_data()

    # 加载指数数据
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
    index_path = processed_dir / "index_processed.parquet"
    index_df = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)

    # 计算特征
    feature_engine = FeatureEngine(config)
    df = feature_engine.compute_all_features(df, index_df)

    logger.info(f"Features prepared. Shape: {df.shape}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列"""
    # 排除非特征列
    exclude_cols = [
        "date", "code", "name", "industry",
        "open", "high", "low", "close", "volume", "amount",
        "turnover", "pct_change", "change", "amplitude",
        "year", "month", "day", "weekday", "quarter",
        "is_month_start", "is_month_end",
        "is_suspended", "is_limit_up", "is_limit_down", "is_abnormal",
        "return_1d", "return_5d", "return_1d_rank", "return_5d_rank",
        "label_binary_1d", "label_binary_5d"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 过滤掉全为NaN的列
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.5:  # 至少50%非空
            valid_cols.append(col)

    logger.info(f"Feature columns: {len(valid_cols)}")

    return valid_cols


def main():
    parser = argparse.ArgumentParser(description="训练量化模型")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgb", "xgb", "lstm", "ensemble", "all"],
        default="ensemble",
        help="训练的模型类型"
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["return_1d_rank", "return_5d_rank", "label_binary_1d"],
        default="return_1d_rank",
        help="预测标签"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="训练后进行评估"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = project_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设置日志
    log_dir = Path(config.get("paths", {}).get("log_dir", "./logs"))
    setup_logging(log_dir)

    logger.info("=" * 50)
    logger.info("模型训练脚本")
    logger.info(f"Model: {args.model}, Label: {args.label}")
    logger.info("=" * 50)

    try:
        # 1. 准备特征
        df = prepare_features(config)

        # 2. 获取特征列
        feature_cols = get_feature_columns(df)

        # 3. 数据划分
        processor = DataProcessor(config)
        train_df, valid_df, test_df = processor.get_train_valid_test_split(df)

        logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

        # 处理缺失值
        train_df = train_df.dropna(subset=feature_cols + [args.label])
        valid_df = valid_df.dropna(subset=feature_cols + [args.label])
        test_df = test_df.dropna(subset=feature_cols + [args.label])

        # 填充剩余缺失值
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        valid_df[feature_cols] = valid_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)

        # 4. 训练模型
        results = {}

        if args.model in ["lgb", "all"]:
            logger.info("Training LightGBM...")
            lgb_model = LightGBMModel(config)
            lgb_result = lgb_model.train(
                train_df[feature_cols], train_df[args.label], train_df["date"],
                valid_df[feature_cols], valid_df[args.label], valid_df["date"]
            )
            lgb_model.save("lgb_model")
            results["lightgbm"] = lgb_result

        if args.model in ["xgb", "all"]:
            logger.info("Training XGBoost...")
            xgb_model = XGBoostModel(config)
            xgb_result = xgb_model.train(
                train_df[feature_cols], train_df[args.label], train_df["date"],
                valid_df[feature_cols], valid_df[args.label], valid_df["date"]
            )
            xgb_model.save("xgb_model")
            results["xgboost"] = xgb_result

        if args.model in ["lstm", "all"]:
            logger.info("Training LSTM...")
            lstm_model = LSTMModel(config)
            lstm_result = lstm_model.train(
                train_df, valid_df, feature_cols, args.label
            )
            lstm_model.save("lstm_model")
            results["lstm"] = lstm_result

        if args.model == "ensemble":
            logger.info("Training Ensemble...")
            ensemble = EnsembleModel(config)
            ensemble_result = ensemble.train(
                train_df, valid_df, feature_cols, args.label
            )
            ensemble.save("ensemble")
            results["ensemble"] = ensemble_result

        # 5. 评估
        if args.evaluate:
            logger.info("Evaluating on test set...")
            evaluator = BacktestEvaluator(config)

            # 获取预测
            if args.model == "ensemble":
                predictions = ensemble.predict(test_df, feature_cols)
            elif args.model == "lgb":
                predictions = lgb_model.predict(test_df[feature_cols])
            elif args.model == "xgb":
                predictions = xgb_model.predict(test_df[feature_cols])
            else:
                # 使用第一个可用模型
                if "lightgbm" in results:
                    predictions = lgb_model.predict(test_df[feature_cols])
                elif "xgboost" in results:
                    predictions = xgb_model.predict(test_df[feature_cols])
                else:
                    predictions = None

            if predictions is not None:
                # 构建预测DataFrame
                pred_df = test_df[["date", "code"]].copy()
                pred_df["prediction"] = predictions

                # 生成评估报告
                report = evaluator.generate_report(
                    pred_df,
                    test_df[["date", "code", "return_1d"]]
                )

                # 生成可视化
                reporter = ReportGenerator(config)
                reporter.generate_full_report(
                    report,
                    feature_importance=results.get("lightgbm", {}).get("feature_importance")
                                       or results.get("xgboost", {}).get("feature_importance")
                )

        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
