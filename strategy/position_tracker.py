"""
持仓跟踪模块
跟踪信号历史、持仓天数、预期收益等
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger


class PositionTracker:
    """持仓跟踪器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        paths = config.get("paths", {})
        self.data_dir = Path(paths.get("data_dir", "./data_storage"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 持仓历史文件
        self.history_file = self.data_dir / "position_history.json"

        # 加载历史
        self.position_history = self._load_history()

        # 信号历史 {code: [(date, signal, price, score), ...]}
        self.signal_history = defaultdict(list)

        # 当前持仓 {code: {entry_date, entry_price, holding_days, ...}}
        self.current_positions = {}

        logger.info("PositionTracker initialized")

    def _load_history(self) -> Dict:
        """加载历史数据"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
        return {"positions": {}, "signals": {}, "trades": []}

    def _save_history(self):
        """保存历史数据"""
        try:
            # 转换为可序列化格式
            save_data = {
                "positions": self.current_positions,
                "signals": {k: list(v) for k, v in self.signal_history.items()},
                "trades": self.position_history.get("trades", [])
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def update_signals(
        self,
        signals: pd.DataFrame,
        date: str,
        prices: Dict[str, float]
    ):
        """
        更新信号历史

        Args:
            signals: 当日信号DataFrame
            date: 日期
            prices: 当日收盘价 {code: price}
        """
        for _, row in signals.iterrows():
            code = row["code"]
            signal = row["signal"]
            score = row.get("combined_score", row.get("prediction", 0))
            price = prices.get(code, row.get("close", 0))

            # 记录信号历史
            self.signal_history[code].append({
                "date": str(date),
                "signal": signal,
                "price": float(price),
                "score": float(score)
            })

            # 保留最近100条记录
            if len(self.signal_history[code]) > 100:
                self.signal_history[code] = self.signal_history[code][-100:]

    def process_signals(
        self,
        signals: pd.DataFrame,
        date: str,
        prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        处理信号并添加跟踪信息

        Args:
            signals: 当日信号DataFrame
            date: 日期
            prices: 当日收盘价

        Returns:
            带跟踪信息的信号DataFrame
        """
        # 更新信号历史
        self.update_signals(signals, date, prices)

        # 复制DataFrame
        result = signals.copy()

        # 添加跟踪列
        result["holding_days"] = 0
        result["entry_date"] = None
        result["entry_price"] = None
        result["current_gain"] = None
        result["expected_gain"] = None
        result["expected_holding_days"] = None
        result["last_buy_date"] = None
        result["last_buy_price"] = None
        result["signal_consistency"] = 0
        result["trend_strength"] = 0

        for idx, row in result.iterrows():
            code = row["code"]
            signal = row["signal"]
            current_price = prices.get(code, row.get("close", 0))

            # 获取信号历史
            history = self.signal_history.get(code, [])

            if signal in ["strong_buy", "buy", "hold"]:
                # 买入/持有信号

                # 找到本轮开始买入的时间
                entry_info = self._find_entry_point(history)
                if entry_info:
                    result.at[idx, "entry_date"] = entry_info["date"]
                    result.at[idx, "entry_price"] = entry_info["price"]
                    result.at[idx, "holding_days"] = entry_info["holding_days"]

                    # 当前收益
                    if entry_info["price"] > 0:
                        result.at[idx, "current_gain"] = (
                            (current_price - entry_info["price"]) / entry_info["price"]
                        )

                # 计算预期收益和持有天数
                expected = self._calculate_expected_return(history, signal)
                result.at[idx, "expected_gain"] = expected["expected_return"]
                result.at[idx, "expected_holding_days"] = expected["expected_days"]

                # 信号一致性(近期信号方向一致性)
                result.at[idx, "signal_consistency"] = self._calculate_signal_consistency(history)

                # 趋势强度
                result.at[idx, "trend_strength"] = self._calculate_trend_strength(history)

            elif signal in ["sell", "strong_sell"]:
                # 卖出信号

                # 找到上次买入时间
                last_buy = self._find_last_buy(history)
                if last_buy:
                    result.at[idx, "last_buy_date"] = last_buy["date"]
                    result.at[idx, "last_buy_price"] = last_buy["price"]

                    # 本轮收益
                    if last_buy["price"] > 0:
                        result.at[idx, "current_gain"] = (
                            (current_price - last_buy["price"]) / last_buy["price"]
                        )
                    result.at[idx, "holding_days"] = last_buy.get("holding_days", 0)

        # 保存历史
        self._save_history()

        return result

    def _find_entry_point(self, history: List[Dict]) -> Optional[Dict]:
        """
        找到本轮买入起点

        Args:
            history: 信号历史

        Returns:
            入场信息
        """
        if not history:
            return None

        # 从最近的信号往前找
        entry_date = None
        entry_price = None
        holding_days = 0

        for i in range(len(history) - 1, -1, -1):
            record = history[i]
            signal = record["signal"]

            if signal in ["strong_buy", "buy"]:
                entry_date = record["date"]
                entry_price = record["price"]
                holding_days = len(history) - i
            elif signal in ["sell", "strong_sell"]:
                # 遇到卖出信号,本轮结束
                break

        if entry_date:
            return {
                "date": entry_date,
                "price": entry_price,
                "holding_days": holding_days
            }
        return None

    def _find_last_buy(self, history: List[Dict]) -> Optional[Dict]:
        """
        找到上次买入信号

        Args:
            history: 信号历史

        Returns:
            上次买入信息
        """
        if not history:
            return None

        # 从最近往前找买入信号
        for i in range(len(history) - 1, -1, -1):
            record = history[i]
            if record["signal"] in ["strong_buy", "buy"]:
                return {
                    "date": record["date"],
                    "price": record["price"],
                    "holding_days": len(history) - i
                }

        return None

    def _calculate_expected_return(
        self,
        history: List[Dict],
        current_signal: str
    ) -> Dict[str, float]:
        """
        计算预期收益

        Args:
            history: 信号历史
            current_signal: 当前信号

        Returns:
            预期收益信息
        """
        if len(history) < 5:
            return {"expected_return": None, "expected_days": None}

        # 基于历史信号表现估计
        # 找到所有完整的买入-卖出周期
        cycles = []
        cycle_start = None
        cycle_price = None

        for record in history:
            signal = record["signal"]
            price = record["price"]

            if signal in ["strong_buy", "buy"] and cycle_start is None:
                cycle_start = record["date"]
                cycle_price = price
            elif signal in ["sell", "strong_sell"] and cycle_start is not None:
                if cycle_price and cycle_price > 0:
                    cycle_return = (price - cycle_price) / cycle_price
                    cycle_days = len([r for r in history
                                     if r["date"] >= cycle_start and r["date"] <= record["date"]])
                    cycles.append({
                        "return": cycle_return,
                        "days": cycle_days
                    })
                cycle_start = None
                cycle_price = None

        if not cycles:
            # 没有完整周期,使用简单估计
            # 强买入预期5-10天,普通买入3-5天
            if current_signal == "strong_buy":
                return {"expected_return": 0.05, "expected_days": 7}
            elif current_signal == "buy":
                return {"expected_return": 0.03, "expected_days": 5}
            else:
                return {"expected_return": 0.02, "expected_days": 3}

        # 基于历史周期计算
        avg_return = np.mean([c["return"] for c in cycles])
        avg_days = np.mean([c["days"] for c in cycles])

        # 根据当前信号强度调整
        if current_signal == "strong_buy":
            avg_return *= 1.2
        elif current_signal == "hold":
            avg_return *= 0.5

        return {
            "expected_return": avg_return,
            "expected_days": int(avg_days)
        }

    def _calculate_signal_consistency(self, history: List[Dict], window: int = 5) -> float:
        """
        计算信号一致性

        Args:
            history: 信号历史
            window: 窗口大小

        Returns:
            一致性分数 (-1到1)
        """
        if len(history) < window:
            return 0

        recent = history[-window:]

        # 统计买入/卖出信号数量
        buy_count = sum(1 for r in recent if r["signal"] in ["strong_buy", "buy"])
        sell_count = sum(1 for r in recent if r["signal"] in ["sell", "strong_sell"])

        total = buy_count + sell_count
        if total == 0:
            return 0

        return (buy_count - sell_count) / total

    def _calculate_trend_strength(self, history: List[Dict], window: int = 10) -> float:
        """
        计算趋势强度

        Args:
            history: 信号历史
            window: 窗口大小

        Returns:
            趋势强度 (-1到1)
        """
        if len(history) < window:
            return 0

        recent = history[-window:]
        scores = [r["score"] for r in recent if "score" in r]

        if len(scores) < 2:
            return 0

        # 计算分数趋势
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        # 归一化到-1到1
        return np.clip(slope * 10, -1, 1)

    def get_position_summary(
        self,
        signals: pd.DataFrame,
        date: str
    ) -> pd.DataFrame:
        """
        获取持仓汇总

        Args:
            signals: 信号DataFrame
            date: 日期

        Returns:
            持仓汇总DataFrame
        """
        # 筛选持仓(买入/持有信号)
        holding = signals[signals["signal"].isin(["strong_buy", "buy", "hold"])].copy()

        if len(holding) == 0:
            return pd.DataFrame()

        # 添加额外信息
        summary = holding[[
            "code", "signal", "combined_score", "confidence",
            "holding_days", "entry_date", "entry_price", "current_gain",
            "expected_gain", "expected_holding_days",
            "signal_consistency", "trend_strength"
        ]].copy()

        # 计算预期目标价
        if "close" in signals.columns:
            price_dict = signals.set_index("code")["close"].to_dict()
            summary["current_price"] = summary["code"].map(price_dict)
            summary["target_price"] = summary.apply(
                lambda x: x["current_price"] * (1 + x["expected_gain"])
                if pd.notna(x["expected_gain"]) and x["current_price"] > 0
                else None,
                axis=1
            )

        # 排序
        summary = summary.sort_values("combined_score", ascending=False)

        return summary

    def get_sell_recommendations(
        self,
        signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取卖出建议

        Args:
            signals: 信号DataFrame

        Returns:
            卖出建议DataFrame
        """
        # 筛选卖出信号
        selling = signals[signals["signal"].isin(["sell", "strong_sell"])].copy()

        if len(selling) == 0:
            return pd.DataFrame()

        summary = selling[[
            "code", "signal", "combined_score",
            "last_buy_date", "last_buy_price", "current_gain", "holding_days"
        ]].copy()

        # 添加当前价格
        if "close" in signals.columns:
            summary["current_price"] = selling["close"]

        # 排序(最应该卖出的在前)
        summary = summary.sort_values("combined_score", ascending=True)

        return summary

    def generate_trade_report(
        self,
        signals: pd.DataFrame,
        date: str
    ) -> str:
        """
        生成交易报告

        Args:
            signals: 信号DataFrame
            date: 日期

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"交易信号报告 - {date}")
        report_lines.append("=" * 60)

        # 持仓建议
        holdings = self.get_position_summary(signals, date)
        if len(holdings) > 0:
            report_lines.append("\n【买入/持有建议】")
            report_lines.append("-" * 60)
            report_lines.append(f"{'代码':<10} {'信号':<12} {'持有天数':<8} {'当前收益':<10} {'预期收益':<10} {'预期天数':<8}")
            report_lines.append("-" * 60)

            for _, row in holdings.head(20).iterrows():
                code = row["code"]
                signal = row["signal"]
                hold_days = row.get("holding_days", 0) or 0
                curr_gain = row.get("current_gain")
                exp_gain = row.get("expected_gain")
                exp_days = row.get("expected_holding_days")

                curr_gain_str = f"{curr_gain:.2%}" if pd.notna(curr_gain) else "N/A"
                exp_gain_str = f"{exp_gain:.2%}" if pd.notna(exp_gain) else "N/A"
                exp_days_str = str(int(exp_days)) if pd.notna(exp_days) else "N/A"

                report_lines.append(
                    f"{code:<10} {signal:<12} {hold_days:<8} {curr_gain_str:<10} {exp_gain_str:<10} {exp_days_str:<8}"
                )

        # 卖出建议
        sells = self.get_sell_recommendations(signals)
        if len(sells) > 0:
            report_lines.append("\n【卖出建议】")
            report_lines.append("-" * 60)
            report_lines.append(f"{'代码':<10} {'信号':<12} {'买入日期':<12} {'持有天数':<8} {'本轮收益':<10}")
            report_lines.append("-" * 60)

            for _, row in sells.head(20).iterrows():
                code = row["code"]
                signal = row["signal"]
                buy_date = row.get("last_buy_date", "N/A") or "N/A"
                hold_days = row.get("holding_days", 0) or 0
                curr_gain = row.get("current_gain")

                curr_gain_str = f"{curr_gain:.2%}" if pd.notna(curr_gain) else "N/A"
                buy_date_str = str(buy_date)[:10] if buy_date != "N/A" else "N/A"

                report_lines.append(
                    f"{code:<10} {signal:<12} {buy_date_str:<12} {hold_days:<8} {curr_gain_str:<10}"
                )

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)


class SignalSuccessAnalyzer:
    """信号成功率分析器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config

    def analyze_signal_accuracy(
        self,
        signals_history: pd.DataFrame,
        returns_data: pd.DataFrame,
        forward_days: List[int] = [1, 3, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        分析信号准确率

        Args:
            signals_history: 历史信号
            returns_data: 收益数据
            forward_days: 前瞻天数列表

        Returns:
            准确率分析结果
        """
        results = []

        for days in forward_days:
            return_col = f"return_{days}d"

            # 计算未来收益
            if return_col not in returns_data.columns:
                returns_data[return_col] = returns_data.groupby("code")["close"].pct_change(days).shift(-days)

            # 合并数据
            merged = signals_history.merge(
                returns_data[["date", "code", return_col]],
                on=["date", "code"],
                how="left"
            )

            # 按信号类型统计
            for signal in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
                signal_data = merged[merged["signal"] == signal]

                if len(signal_data) == 0:
                    continue

                # 计算准确率
                if signal in ["strong_buy", "buy"]:
                    # 买入信号:预期上涨,实际上涨为正确
                    accuracy = (signal_data[return_col] > 0).mean()
                    avg_return = signal_data[return_col].mean()
                elif signal in ["sell", "strong_sell"]:
                    # 卖出信号:预期下跌,实际下跌为正确
                    accuracy = (signal_data[return_col] < 0).mean()
                    avg_return = signal_data[return_col].mean()
                else:
                    # 持有信号:小幅波动为正确
                    accuracy = (signal_data[return_col].abs() < 0.05).mean()
                    avg_return = signal_data[return_col].mean()

                results.append({
                    "signal": signal,
                    "forward_days": days,
                    "count": len(signal_data),
                    "accuracy": accuracy,
                    "avg_return": avg_return,
                    "win_rate": (signal_data[return_col] > 0).mean(),
                    "avg_win": signal_data[signal_data[return_col] > 0][return_col].mean(),
                    "avg_loss": signal_data[signal_data[return_col] < 0][return_col].mean(),
                    "profit_factor": (
                        abs(signal_data[signal_data[return_col] > 0][return_col].sum()) /
                        (abs(signal_data[signal_data[return_col] < 0][return_col].sum()) + 1e-10)
                    )
                })

        return pd.DataFrame(results)

    def analyze_by_confidence(
        self,
        signals_history: pd.DataFrame,
        returns_data: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.DataFrame:
        """
        按置信度分析准确率

        Args:
            signals_history: 历史信号
            returns_data: 收益数据
            forward_days: 前瞻天数

        Returns:
            按置信度分组的准确率
        """
        return_col = f"return_{forward_days}d"

        if return_col not in returns_data.columns:
            returns_data[return_col] = returns_data.groupby("code")["close"].pct_change(forward_days).shift(-forward_days)

        merged = signals_history.merge(
            returns_data[["date", "code", return_col]],
            on=["date", "code"],
            how="left"
        )

        # 按置信度分组
        if "confidence" not in merged.columns:
            merged["confidence"] = 0.5

        merged["confidence_group"] = pd.cut(
            merged["confidence"],
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )

        results = []
        for group in merged["confidence_group"].unique():
            group_data = merged[merged["confidence_group"] == group]

            if len(group_data) == 0:
                continue

            # 买入信号准确率
            buy_data = group_data[group_data["signal"].isin(["strong_buy", "buy"])]
            if len(buy_data) > 0:
                results.append({
                    "confidence_group": str(group),
                    "signal_type": "buy",
                    "count": len(buy_data),
                    "accuracy": (buy_data[return_col] > 0).mean(),
                    "avg_return": buy_data[return_col].mean()
                })

            # 卖出信号准确率
            sell_data = group_data[group_data["signal"].isin(["sell", "strong_sell"])]
            if len(sell_data) > 0:
                results.append({
                    "confidence_group": str(group),
                    "signal_type": "sell",
                    "count": len(sell_data),
                    "accuracy": (sell_data[return_col] < 0).mean(),
                    "avg_return": sell_data[return_col].mean()
                })

        return pd.DataFrame(results)

    def generate_accuracy_report(
        self,
        accuracy_df: pd.DataFrame
    ) -> str:
        """
        生成准确率报告

        Args:
            accuracy_df: 准确率分析结果

        Returns:
            报告文本
        """
        lines = []
        lines.append("=" * 70)
        lines.append("信号准确率分析报告")
        lines.append("=" * 70)

        for signal in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
            signal_data = accuracy_df[accuracy_df["signal"] == signal]

            if len(signal_data) == 0:
                continue

            lines.append(f"\n【{signal.upper()}信号】")
            lines.append("-" * 70)
            lines.append(f"{'前瞻天数':<10} {'样本数':<10} {'准确率':<10} {'平均收益':<12} {'胜率':<10} {'盈亏比':<10}")
            lines.append("-" * 70)

            for _, row in signal_data.iterrows():
                lines.append(
                    f"{row['forward_days']:<10} {row['count']:<10} "
                    f"{row['accuracy']:.2%}     {row['avg_return']:.2%}      "
                    f"{row['win_rate']:.2%}     {row['profit_factor']:.2f}"
                )

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
