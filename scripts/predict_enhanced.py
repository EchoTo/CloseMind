#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版预测脚本
包含持仓跟踪、预期收益估算和信号成功率分析
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
import numpy as np
from loguru import logger

from data.qlib_converter import SimpleDataLoader
from features import FeatureEngine
from models import EnsembleModel
from strategy import SignalGenerator, PositionTracker, SignalSuccessAnalyzer
from report import ReportGenerator


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "predict_enhanced_{time}.log",
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


def print_colored(text: str, color: str = "white"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[0m",
        "bold": "\033[1m",
    }
    end = "\033[0m"
    print(f"{colors.get(color, '')}{text}{end}")


def format_report(
    tracked_signals: pd.DataFrame,
    date: str,
    accuracy_stats: pd.DataFrame = None
) -> str:
    """格式化输出报告"""
    lines = []

    lines.append("\n" + "=" * 80)
    lines.append(f"  A股量化预测报告 - {date}")
    lines.append("=" * 80)

    # 买入/持有建议
    buy_hold = tracked_signals[tracked_signals["signal"].isin(["strong_buy", "buy", "hold"])]
    buy_hold = buy_hold.sort_values("combined_score", ascending=False)

    if len(buy_hold) > 0:
        lines.append("\n【买入/持有建议】Top 30")
        lines.append("-" * 80)
        lines.append(f"{'序号':<4} {'代码':<8} {'信号':<10} {'分数':<8} {'持有天数':<8} "
                     f"{'当前收益':<10} {'预期收益':<10} {'预期天数':<8} {'趋势':<6}")
        lines.append("-" * 80)

        for i, (_, row) in enumerate(buy_hold.head(30).iterrows(), 1):
            code = row["code"]
            signal = row["signal"]
            score = row.get("combined_score", 0)

            hold_days = row.get("holding_days", 0)
            hold_days = int(hold_days) if pd.notna(hold_days) else 0

            curr_gain = row.get("current_gain")
            curr_str = f"{curr_gain:.2%}" if pd.notna(curr_gain) else "首次"

            exp_gain = row.get("expected_gain")
            exp_str = f"{exp_gain:.2%}" if pd.notna(exp_gain) else "N/A"

            exp_days = row.get("expected_holding_days")
            exp_days_str = str(int(exp_days)) if pd.notna(exp_days) else "N/A"

            trend = row.get("trend_strength", 0)
            if trend > 0.3:
                trend_str = "↑强"
            elif trend > 0:
                trend_str = "↑"
            elif trend < -0.3:
                trend_str = "↓强"
            elif trend < 0:
                trend_str = "↓"
            else:
                trend_str = "→"

            # 信号颜色标记
            signal_mark = ""
            if signal == "strong_buy":
                signal_mark = "★★"
            elif signal == "buy":
                signal_mark = "★"

            lines.append(
                f"{i:<4} {code:<8} {signal:<10} {score:.4f}  {hold_days:<8} "
                f"{curr_str:<10} {exp_str:<10} {exp_days_str:<8} {trend_str:<6} {signal_mark}"
            )

    # 卖出建议
    sell = tracked_signals[tracked_signals["signal"].isin(["sell", "strong_sell"])]
    sell = sell.sort_values("combined_score", ascending=True)

    if len(sell) > 0:
        lines.append("\n【卖出建议】Top 20")
        lines.append("-" * 80)
        lines.append(f"{'序号':<4} {'代码':<8} {'信号':<12} {'上次买入':<12} "
                     f"{'持有天数':<8} {'本轮收益':<10}")
        lines.append("-" * 80)

        for i, (_, row) in enumerate(sell.head(20).iterrows(), 1):
            code = row["code"]
            signal = row["signal"]

            last_buy = row.get("last_buy_date")
            last_buy_str = str(last_buy)[:10] if pd.notna(last_buy) else "N/A"

            hold_days = row.get("holding_days", 0)
            hold_days = int(hold_days) if pd.notna(hold_days) else 0

            curr_gain = row.get("current_gain")
            curr_str = f"{curr_gain:.2%}" if pd.notna(curr_gain) else "N/A"

            lines.append(
                f"{i:<4} {code:<8} {signal:<12} {last_buy_str:<12} "
                f"{hold_days:<8} {curr_str:<10}"
            )

    # 信号统计
    lines.append("\n【信号统计】")
    lines.append("-" * 40)
    signal_counts = tracked_signals["signal"].value_counts()
    for signal, count in signal_counts.items():
        lines.append(f"  {signal:<15}: {count}")

    # 准确率统计(如果有)
    if accuracy_stats is not None and len(accuracy_stats) > 0:
        lines.append("\n【历史信号准确率】(基于过去数据)")
        lines.append("-" * 60)

        for signal in ["strong_buy", "buy"]:
            sig_stats = accuracy_stats[
                (accuracy_stats["signal"] == signal) &
                (accuracy_stats["forward_days"] == 5)
            ]
            if len(sig_stats) > 0:
                acc = sig_stats["accuracy"].values[0]
                win = sig_stats["win_rate"].values[0]
                avg_ret = sig_stats["avg_return"].values[0]
                lines.append(
                    f"  {signal:<12}: 准确率={acc:.1%}, 胜率={win:.1%}, 平均收益={avg_ret:.2%}"
                )

    lines.append("\n" + "=" * 80)
    lines.append("说明:")
    lines.append("  - 持有天数: 从首次买入信号到现在的天数")
    lines.append("  - 当前收益: 相对入场价的收益率")
    lines.append("  - 预期收益: 基于历史数据估算的本轮预期收益")
    lines.append("  - 预期天数: 建议继续持有的天数")
    lines.append("  - 趋势: ↑上涨趋势, ↓下跌趋势, →震荡")
    lines.append("  - ★★强烈买入, ★买入")
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="增强版股票预测")
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
        help="预测日期(YYYY-MM-DD)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        help="模型名称"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N股票"
    )
    parser.add_argument(
        "--analyze-accuracy",
        action="store_true",
        help="分析历史信号准确率"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = project_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设置日志
    log_dir = Path(config.get("paths", {}).get("log_dir", "./logs"))
    setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("增强版股票预测脚本")
    logger.info("=" * 60)

    try:
        # 1. 加载数据
        logger.info("Loading data...")
        loader = SimpleDataLoader(config)

        if args.date:
            end_date = args.date
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 加载历史数据(用于计算特征和分析准确率)
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
        df = loader.load_data(start_date=start_date, end_date=end_date)

        logger.info(f"Loaded {len(df)} records")

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

        # 4. 获取特征列
        feature_cols = get_feature_columns(df)
        logger.info(f"Using {len(feature_cols)} features")

        # 5. 加载模型
        logger.info(f"Loading model: {args.model}")
        ensemble = EnsembleModel(config)
        ensemble.load(args.model)

        # 6. 对所有日期进行预测(用于历史分析)
        all_predictions = []
        unique_dates = sorted(df["date"].unique())

        for date in unique_dates[-60:]:  # 最近60天
            day_df = df[df["date"] == date].copy()
            day_df[feature_cols] = day_df[feature_cols].fillna(0)

            if len(day_df) > 0:
                pred = ensemble.predict(day_df, feature_cols)
                day_df["prediction"] = pred
                day_df["pred_daily"] = pred
                all_predictions.append(day_df[["date", "code", "prediction", "pred_daily", "close"]])

        all_pred_df = pd.concat(all_predictions, ignore_index=True)

        # 7. 获取最新数据
        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date].copy()
        latest_df[feature_cols] = latest_df[feature_cols].fillna(0)

        predictions = ensemble.predict(latest_df, feature_cols)
        latest_df["prediction"] = predictions
        latest_df["pred_daily"] = predictions

        # 8. 生成信号
        logger.info("Generating signals...")
        signal_generator = SignalGenerator(config)

        market_cols = ["date", "code", "is_limit_up", "is_limit_down", "is_suspended", "close"]
        market_data = latest_df[[c for c in market_cols if c in latest_df.columns]].copy()

        signals = signal_generator.generate_signals(latest_df, market_data)

        # 9. 持仓跟踪
        logger.info("Tracking positions...")
        position_tracker = PositionTracker(config)

        # 更新历史信号
        for date in unique_dates[-30:]:  # 用最近30天建立历史
            day_signals = all_pred_df[all_pred_df["date"] == date].copy()
            if len(day_signals) > 0:
                day_signals["pred_daily"] = day_signals["prediction"]
                day_market = df[df["date"] == date][["date", "code", "close", "is_limit_up", "is_limit_down", "is_suspended"]].copy()

                if len(day_market) > 0:
                    day_full_signals = signal_generator.generate_signals(day_signals, day_market)
                    prices = day_market.set_index("code")["close"].to_dict()
                    position_tracker.update_signals(day_full_signals, str(date)[:10], prices)

        # 处理今日信号并添加跟踪信息
        prices = market_data.set_index("code")["close"].to_dict()
        tracked_signals = position_tracker.process_signals(signals, str(latest_date)[:10], prices)

        # 10. 分析历史准确率(可选)
        accuracy_stats = None
        if args.analyze_accuracy:
            logger.info("Analyzing signal accuracy...")
            analyzer = SignalSuccessAnalyzer(config)

            # 生成历史信号
            historical_signals = []
            for date in unique_dates[-60:-5]:  # 排除最近5天(没有结果)
                day_pred = all_pred_df[all_pred_df["date"] == date].copy()
                if len(day_pred) > 0:
                    day_pred["pred_daily"] = day_pred["prediction"]
                    day_market = df[df["date"] == date][["date", "code", "close", "is_limit_up", "is_limit_down", "is_suspended"]].copy()
                    if len(day_market) > 0:
                        day_signals = signal_generator.generate_signals(day_pred, day_market)
                        historical_signals.append(day_signals)

            if historical_signals:
                hist_signals_df = pd.concat(historical_signals, ignore_index=True)
                accuracy_stats = analyzer.analyze_signal_accuracy(
                    hist_signals_df,
                    df[["date", "code", "close"]],
                    forward_days=[1, 3, 5, 10]
                )

                # 打印准确率报告
                report_text = analyzer.generate_accuracy_report(accuracy_stats)
                logger.info(report_text)

        # 11. 生成报告
        report = format_report(tracked_signals, str(latest_date)[:10], accuracy_stats)
        print(report)

        # 12. 保存结果
        report_dir = Path(config.get("paths", {}).get("report_dir", "./reports"))
        report_dir.mkdir(parents=True, exist_ok=True)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = report_dir / f"prediction_enhanced_{latest_date}.csv"

        # 保存详细结果
        save_cols = [
            "code", "signal", "combined_score", "confidence",
            "holding_days", "entry_date", "entry_price", "current_gain",
            "expected_gain", "expected_holding_days", "trend_strength",
            "last_buy_date", "last_buy_price"
        ]
        save_cols = [c for c in save_cols if c in tracked_signals.columns]

        tracked_signals[save_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Results saved to {output_path}")

        # 保存报告
        report_txt_path = report_dir / f"report_{latest_date}.txt"
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {report_txt_path}")

        logger.info("=" * 60)
        logger.info("Prediction completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
