#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化预测系统主入口
CloseMind - A-Share Quantitative Prediction System

使用方法:
    1. 下载数据: python main.py download
    2. 训练模型: python main.py train
    3. 生成预测: python main.py predict

详细参数请查看: python main.py <command> --help
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger


def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get("logging", {})
    log_dir = Path(config.get("paths", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "closemind_{time}.log",
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "30 days"),
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    )


def cmd_download(args, config):
    """下载数据命令"""
    from data import DataDownloader, DataProcessor, QlibConverter

    logger.info("Starting data download...")

    # 下载
    downloader = DataDownloader(config)
    if args.mode == "full":
        downloader.download_all()
    else:
        downloader.update_data()

    # 处理
    if not args.skip_process:
        processor = DataProcessor(config)
        processor.process()

    # 转换
    if not args.skip_qlib:
        converter = QlibConverter(config)
        converter.convert()

    logger.info("Data download completed!")


def cmd_train(args, config):
    """训练模型命令"""
    import pandas as pd
    from data import DataProcessor
    from data.qlib_converter import SimpleDataLoader
    from features import FeatureEngine
    from models import (
        EnsembleModel, LightGBMModel, XGBoostModel, LSTMModel,
        PatchTSTModel, iTransformerModel, MambaModel, MoEModel
    )
    from backtest import BacktestEvaluator
    from report import ReportGenerator

    logger.info("Starting model training...")

    # 加载数据
    loader = SimpleDataLoader(config)
    df = loader.load_data()

    # 加载指数
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
    index_path = processed_dir / "index_processed.parquet"
    index_df = pd.read_parquet(index_path) if index_path.exists() else None

    # 计算特征
    feature_engine = FeatureEngine(config)
    df = feature_engine.compute_all_features(df, index_df)

    # 获取特征列
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
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].notna().sum() > len(df) * 0.5]

    # 数据划分
    processor = DataProcessor(config)
    train_df, valid_df, test_df = processor.get_train_valid_test_split(df)

    # 处理缺失值
    label_col = args.label
    for data in [train_df, valid_df, test_df]:
        data.dropna(subset=feature_cols + [label_col], inplace=True)
        data[feature_cols] = data[feature_cols].fillna(0)

    # 训练集成模型
    ensemble = EnsembleModel(config)
    result = ensemble.train(train_df, valid_df, feature_cols, label_col)
    ensemble.save("ensemble")

    # 评估
    if args.evaluate:
        evaluator = BacktestEvaluator(config)
        predictions = ensemble.predict(test_df, feature_cols)

        pred_df = test_df[["date", "code"]].copy()
        pred_df["prediction"] = predictions

        report = evaluator.generate_report(
            pred_df,
            test_df[["date", "code", "return_1d"]]
        )

        reporter = ReportGenerator(config)
        reporter.generate_full_report(report)

    logger.info("Model training completed!")


def cmd_predict(args, config):
    """生成预测命令"""
    from datetime import datetime, timedelta
    import pandas as pd
    from data.qlib_converter import SimpleDataLoader
    from features import FeatureEngine
    from models import EnsembleModel
    from strategy import SignalGenerator
    from report import ReportGenerator

    logger.info("Starting prediction generation...")

    # 加载数据
    loader = SimpleDataLoader(config)
    end_date = args.date or datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    df = loader.load_data(start_date=start_date, end_date=end_date)

    # 加载指数
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
    index_path = processed_dir / "index_processed.parquet"
    index_df = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)
        index_df = index_df[(index_df["date"] >= start_date) & (index_df["date"] <= end_date)]

    # 计算特征
    feature_engine = FeatureEngine(config)
    df = feature_engine.compute_all_features(df, index_df)

    # 获取最新数据
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    # 获取特征列
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
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].notna().sum() > len(df) * 0.5]

    latest_df[feature_cols] = latest_df[feature_cols].fillna(0)

    # 加载模型
    ensemble = EnsembleModel(config)
    ensemble.load(args.model)

    # 预测
    predictions = ensemble.predict(latest_df, feature_cols)

    # 生成信号
    result_df = latest_df[["date", "code", "close"]].copy()
    result_df["prediction"] = predictions
    result_df["pred_daily"] = predictions

    signal_generator = SignalGenerator(config)
    market_data = latest_df[["date", "code", "is_limit_up", "is_limit_down", "is_suspended", "close"]].copy()
    signals = signal_generator.generate_signals(result_df, market_data)

    # 输出Top N
    top_signals = signal_generator.get_top_signals(
        signals, str(latest_date)[:10], top_n=args.top_n
    )

    logger.info(f"\nTop {args.top_n} Stocks for {latest_date}:")
    for i, (_, row) in enumerate(top_signals.iterrows(), 1):
        logger.info(f"{i:2d}. {row['code']} | Score: {row['combined_score']:.4f} | Signal: {row['signal']}")

    # 保存
    report_dir = Path(config.get("paths", {}).get("report_dir", "./reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"predictions_{latest_date}.csv"
    top_signals.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"\nResults saved to {output_path}")
    logger.info("Prediction completed!")


def main():
    parser = argparse.ArgumentParser(
        description="CloseMind - A股量化预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 下载命令
    download_parser = subparsers.add_parser("download", help="下载数据")
    download_parser.add_argument(
        "--mode",
        choices=["full", "update"],
        default="full",
        help="下载模式"
    )
    download_parser.add_argument("--skip-process", action="store_true", help="跳过处理")
    download_parser.add_argument("--skip-qlib", action="store_true", help="跳过Qlib转换")

    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--label",
        choices=["return_1d_rank", "return_5d_rank"],
        default="return_1d_rank",
        help="预测标签"
    )
    train_parser.add_argument("--evaluate", action="store_true", help="训练后评估")

    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="生成预测")
    predict_parser.add_argument("--date", type=str, help="预测日期")
    predict_parser.add_argument("--model", default="ensemble", help="模型名称")
    predict_parser.add_argument("--top-n", type=int, default=50, help="Top N股票")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # 加载配置
    project_root = Path(__file__).parent
    config_path = project_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设置日志
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("CloseMind - A股量化预测系统")
    logger.info("=" * 60)

    # 执行命令
    try:
        if args.command == "download":
            cmd_download(args, config)
        elif args.command == "train":
            cmd_train(args, config)
        elif args.command == "predict":
            cmd_predict(args, config)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
