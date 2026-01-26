#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测脚本
生成股票预测和交易信号
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import pandas as pd
from loguru import logger

from data.qlib_converter import SimpleDataLoader
from features import FeatureEngine
from models import EnsembleModel
from strategy import SignalGenerator
from report import ReportGenerator


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "predict_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO"
    )


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列"""
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
    valid_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]

    return valid_cols


def main():
    parser = argparse.ArgumentParser(description="生成股票预测")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="预测日期(YYYY-MM-DD),默认为最新日期"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        help="使用的模型名称"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="输出Top N股票"
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
    logger.info("股票预测脚本")
    logger.info("=" * 50)

    try:
        # 1. 加载数据
        logger.info("Loading data...")
        loader = SimpleDataLoader(config)

        # 确定日期范围(预测需要历史数据计算特征)
        if args.date:
            end_date = args.date
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 加载最近60天数据用于计算特征
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
        df = loader.load_data(start_date=start_date, end_date=end_date)

        logger.info(f"Loaded data: {len(df)} records")

        # 2. 加载指数数据
        processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
        index_path = processed_dir / "index_processed.parquet"
        index_df = None
        if index_path.exists():
            index_df = pd.read_parquet(index_path)
            index_df = index_df[(index_df["date"] >= start_date) & (index_df["date"] <= end_date)]

        # 3. 计算特征
        logger.info("Computing features...")
        feature_engine = FeatureEngine(config)
        df = feature_engine.compute_all_features(df, index_df)

        # 4. 获取最新日期数据
        latest_date = df["date"].max()
        logger.info(f"Latest date in data: {latest_date}")

        latest_df = df[df["date"] == latest_date].copy()

        # 5. 获取特征列
        feature_cols = get_feature_columns(df)

        # 填充缺失值
        latest_df[feature_cols] = latest_df[feature_cols].fillna(0)

        # 6. 加载模型并预测
        logger.info(f"Loading model: {args.model}")
        ensemble = EnsembleModel(config)
        ensemble.load(args.model)

        # 预测
        logger.info("Generating predictions...")
        predictions = ensemble.predict(latest_df, feature_cols)

        # 7. 构建预测结果
        result_df = latest_df[["date", "code", "name", "industry", "close"]].copy()
        result_df["prediction"] = predictions
        result_df["pred_rank"] = result_df["prediction"].rank(pct=True)

        # 8. 生成信号
        logger.info("Generating signals...")
        signal_generator = SignalGenerator(config)

        # 准备市场数据
        market_cols = ["date", "code", "is_limit_up", "is_limit_down", "is_suspended", "close"]
        market_data = latest_df[[c for c in market_cols if c in latest_df.columns]].copy()

        # 生成信号需要pred_daily列
        result_df["pred_daily"] = result_df["prediction"]
        signals = signal_generator.generate_signals(result_df, market_data)

        # 9. 获取Top N
        top_signals = signal_generator.get_top_signals(
            signals,
            str(latest_date)[:10],
            top_n=args.top_n,
            signal_types=["strong_buy", "buy"]
        )

        # 10. 输出结果
        logger.info("=" * 50)
        logger.info(f"Top {args.top_n} Stocks for {latest_date}")
        logger.info("=" * 50)

        for i, (_, row) in enumerate(top_signals.iterrows(), 1):
            logger.info(
                f"{i:2d}. {row['code']} | {row.get('name', 'N/A'):10s} | "
                f"Score: {row['combined_score']:.4f} | Signal: {row['signal']}"
            )

        # 保存结果
        if args.output:
            output_path = Path(args.output)
        else:
            report_dir = Path(config.get("paths", {}).get("report_dir", "./reports"))
            report_dir.mkdir(parents=True, exist_ok=True)
            output_path = report_dir / f"predictions_{latest_date}.csv"

        top_signals.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Results saved to {output_path}")

        # 生成可视化
        reporter = ReportGenerator(config)
        reporter.plot_prediction_distribution(
            signals,
            str(latest_date)[:10],
            pred_col="combined_score",
            save_name=f"prediction_{latest_date}.png"
        )

        logger.info("=" * 50)
        logger.info("Prediction completed!")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
