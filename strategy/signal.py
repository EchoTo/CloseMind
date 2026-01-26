"""
信号生成模块
综合日度和周度预测生成交易信号
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class SignalGenerator:
    """信号生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        signal_config = config.get("signal", {})

        # 信号组合配置
        self.combine_method = signal_config.get("combine_method", "weighted")
        self.daily_weight = signal_config.get("daily_weight", 0.6)
        self.weekly_weight = signal_config.get("weekly_weight", 0.4)

        # 信号阈值
        threshold_config = signal_config.get("threshold", {})
        self.threshold_strong_buy = threshold_config.get("strong_buy", 0.8)
        self.threshold_buy = threshold_config.get("buy", 0.6)
        self.threshold_hold = threshold_config.get("hold", 0.4)
        self.threshold_sell = threshold_config.get("sell", 0.2)

        # 交易限制
        stock_filter = config.get("data", {}).get("stock_filter", {})
        self.exclude_st = stock_filter.get("exclude_st", True)
        self.exclude_suspend = stock_filter.get("exclude_suspend", True)

        logger.info("SignalGenerator initialized")

    def generate_signals(
        self,
        predictions: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            predictions: 预测结果DataFrame,包含columns: date, code, pred_daily, pred_weekly
            market_data: 市场数据DataFrame,包含涨跌停、停牌等信息

        Returns:
            带信号的DataFrame
        """
        logger.info("Generating trading signals...")

        # 合并预测和市场数据
        df = predictions.merge(
            market_data[["date", "code", "is_limit_up", "is_limit_down", "is_suspended", "close"]],
            on=["date", "code"],
            how="left"
        )

        # 综合预测分数
        df["combined_score"] = self._combine_predictions(
            df.get("pred_daily", df.get("pred_1d")),
            df.get("pred_weekly", df.get("pred_5d"))
        )

        # 每日排名
        df["score_rank"] = df.groupby("date")["combined_score"].rank(pct=True)

        # 生成信号
        df["signal"] = self._generate_signal_labels(df["score_rank"])

        # 过滤不可交易股票
        df = self._filter_tradable(df)

        # 添加置信度
        df["confidence"] = self._calculate_confidence(df)

        logger.info(f"Signals generated. Shape: {df.shape}")

        return df

    def _combine_predictions(
        self,
        pred_daily: pd.Series,
        pred_weekly: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        综合日度和周度预测

        Args:
            pred_daily: 日度预测
            pred_weekly: 周度预测

        Returns:
            综合分数
        """
        if pred_weekly is None or pred_weekly.isna().all():
            return pred_daily

        if self.combine_method == "weighted":
            # 加权平均
            combined = (
                self.daily_weight * pred_daily.fillna(0) +
                self.weekly_weight * pred_weekly.fillna(0)
            )
        elif self.combine_method == "rank_fusion":
            # 排名融合
            rank_daily = pred_daily.rank(pct=True)
            rank_weekly = pred_weekly.rank(pct=True)
            combined = (
                self.daily_weight * rank_daily.fillna(0.5) +
                self.weekly_weight * rank_weekly.fillna(0.5)
            )
        else:
            combined = pred_daily

        return combined

    def _generate_signal_labels(self, score_rank: pd.Series) -> pd.Series:
        """
        根据分数排名生成信号标签

        Args:
            score_rank: 分数排名(0-1)

        Returns:
            信号标签
        """
        conditions = [
            score_rank >= self.threshold_strong_buy,
            score_rank >= self.threshold_buy,
            score_rank >= self.threshold_hold,
            score_rank >= self.threshold_sell,
        ]

        choices = ["strong_buy", "buy", "hold", "sell"]

        return pd.Series(
            np.select(conditions, choices, default="strong_sell"),
            index=score_rank.index
        )

    def _filter_tradable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤不可交易股票

        Args:
            df: 数据DataFrame

        Returns:
            过滤后的数据
        """
        original_count = len(df)

        # 排除停牌
        if self.exclude_suspend and "is_suspended" in df.columns:
            df = df[~df["is_suspended"].fillna(False)]

        # 排除涨停(买入受限)
        if "is_limit_up" in df.columns:
            # 涨停股不能买入
            df.loc[df["is_limit_up"].fillna(False), "signal"] = "hold"

        # 排除跌停(卖出受限)
        if "is_limit_down" in df.columns:
            # 跌停股不能卖出(但仍可标记)
            df.loc[df["is_limit_down"].fillna(False), "can_sell"] = False

        logger.info(f"Filtered tradable: {original_count} -> {len(df)}")

        return df

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        计算信号置信度

        Args:
            df: 数据DataFrame

        Returns:
            置信度(0-1)
        """
        # 基于排名的置信度
        confidence = np.abs(df["score_rank"] - 0.5) * 2

        # 极端排名具有更高置信度
        confidence = np.where(
            df["score_rank"] >= 0.9,
            confidence * 1.2,
            confidence
        )
        confidence = np.where(
            df["score_rank"] <= 0.1,
            confidence * 1.2,
            confidence
        )

        return np.clip(confidence, 0, 1)

    def get_top_signals(
        self,
        signals: pd.DataFrame,
        date: str,
        top_n: int = 50,
        signal_types: List[str] = None
    ) -> pd.DataFrame:
        """
        获取指定日期的Top信号

        Args:
            signals: 信号DataFrame
            date: 日期
            top_n: 返回数量
            signal_types: 信号类型过滤

        Returns:
            Top信号DataFrame
        """
        df = signals[signals["date"] == date].copy()

        if signal_types:
            df = df[df["signal"].isin(signal_types)]

        # 按分数排序
        df = df.sort_values("combined_score", ascending=False).head(top_n)

        return df

    def get_signal_summary(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        获取信号汇总统计

        Args:
            signals: 信号DataFrame

        Returns:
            汇总DataFrame
        """
        summary = signals.groupby(["date", "signal"]).size().unstack(fill_value=0)
        summary["total"] = summary.sum(axis=1)

        return summary

    def generate_position_changes(
        self,
        signals: pd.DataFrame,
        current_positions: Dict[str, float],
        max_positions: int = 50
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        生成持仓变化(买入/卖出列表)

        Args:
            signals: 当日信号
            current_positions: 当前持仓 {code: weight}
            max_positions: 最大持仓数

        Returns:
            (买入列表, 卖出列表)
        """
        # 获取买入候选
        buy_candidates = signals[
            signals["signal"].isin(["strong_buy", "buy"])
        ].sort_values("combined_score", ascending=False)

        # 获取卖出候选
        sell_candidates = signals[
            signals["signal"].isin(["strong_sell", "sell"])
        ]

        # 当前持仓中需要卖出的
        to_sell = []
        for code, weight in current_positions.items():
            if code in sell_candidates["code"].values:
                to_sell.append({
                    "code": code,
                    "weight": weight,
                    "reason": "signal_sell"
                })

        # 新建仓位
        current_count = len(current_positions) - len(to_sell)
        available_slots = max_positions - current_count

        to_buy = []
        for _, row in buy_candidates.iterrows():
            if row["code"] not in current_positions and len(to_buy) < available_slots:
                to_buy.append({
                    "code": row["code"],
                    "score": row["combined_score"],
                    "confidence": row["confidence"],
                    "signal": row["signal"]
                })

        return to_buy, to_sell


class SignalAnalyzer:
    """信号分析器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config

    def analyze_signal_performance(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        forward_days: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        分析信号表现

        Args:
            signals: 信号DataFrame
            returns: 收益率DataFrame
            forward_days: 分析的前瞻天数

        Returns:
            信号表现DataFrame
        """
        results = []

        for days in forward_days:
            return_col = f"return_{days}d"
            if return_col not in returns.columns:
                continue

            # 合并数据
            df = signals.merge(
                returns[["date", "code", return_col]],
                on=["date", "code"],
                how="left"
            )

            # 按信号分组统计收益
            for signal_type in df["signal"].unique():
                signal_data = df[df["signal"] == signal_type]

                results.append({
                    "signal": signal_type,
                    "forward_days": days,
                    "count": len(signal_data),
                    "mean_return": signal_data[return_col].mean(),
                    "median_return": signal_data[return_col].median(),
                    "std_return": signal_data[return_col].std(),
                    "win_rate": (signal_data[return_col] > 0).mean(),
                    "sharpe": (
                        signal_data[return_col].mean() /
                        (signal_data[return_col].std() + 1e-10) *
                        np.sqrt(252 / days)
                    )
                })

        return pd.DataFrame(results)

    def analyze_ic_by_signal(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        按信号类型分析IC

        Args:
            signals: 信号DataFrame
            returns: 收益率DataFrame

        Returns:
            IC分析DataFrame
        """
        df = signals.merge(
            returns[["date", "code", "return_1d"]],
            on=["date", "code"],
            how="left"
        )

        # 计算每日IC
        daily_ic = df.groupby("date").apply(
            lambda x: np.corrcoef(x["combined_score"], x["return_1d"])[0, 1]
            if len(x) > 10 else np.nan
        )

        return pd.DataFrame({
            "ic_mean": [daily_ic.mean()],
            "ic_std": [daily_ic.std()],
            "icir": [daily_ic.mean() / (daily_ic.std() + 1e-10)],
            "ic_positive_rate": [(daily_ic > 0).mean()]
        })
