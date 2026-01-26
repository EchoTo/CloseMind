"""
市场特征
包含相对强弱、行业动量、市场情绪等特征
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


class MarketFeatures:
    """市场特征计算器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        feature_config = config.get("features", {}).get("market", {})

        # 基准指数
        self.benchmark = feature_config.get("benchmark", "000300")

        # 行业收益周期
        self.industry_return_periods = feature_config.get("industry_return_periods", [1, 5, 20])

        logger.info("MarketFeatures initialized")

    def _compute_benchmark_features(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算基准指数相关特征

        Args:
            df: 股票数据
            index_df: 指数数据

        Returns:
            带基准特征的数据
        """
        if index_df is None or len(index_df) == 0:
            logger.warning("No index data available for benchmark features")
            return df

        # 获取基准指数数据
        benchmark_data = index_df[index_df["code"] == self.benchmark].copy()

        if len(benchmark_data) == 0:
            logger.warning(f"Benchmark {self.benchmark} not found in index data")
            return df

        benchmark_data = benchmark_data.sort_values("date")

        # 计算基准收益率
        benchmark_data["benchmark_return_1d"] = benchmark_data["close"].pct_change(1)
        benchmark_data["benchmark_return_5d"] = benchmark_data["close"].pct_change(5)
        benchmark_data["benchmark_return_20d"] = benchmark_data["close"].pct_change(20)

        # 基准波动率
        benchmark_data["benchmark_volatility"] = benchmark_data["close"].pct_change().rolling(
            20, min_periods=5
        ).std() * np.sqrt(252)

        # 基准动量
        benchmark_data["benchmark_ma_5"] = benchmark_data["close"].rolling(5, min_periods=1).mean()
        benchmark_data["benchmark_ma_20"] = benchmark_data["close"].rolling(20, min_periods=1).mean()
        benchmark_data["benchmark_trend"] = benchmark_data["benchmark_ma_5"] / benchmark_data["benchmark_ma_20"] - 1

        # 选择需要合并的列
        merge_cols = [
            "date", "benchmark_return_1d", "benchmark_return_5d", "benchmark_return_20d",
            "benchmark_volatility", "benchmark_trend"
        ]
        benchmark_features = benchmark_data[merge_cols].copy()

        # 合并到股票数据
        df = df.merge(benchmark_features, on="date", how="left")

        return df

    def _compute_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算相对强弱特征

        Args:
            df: 股票数据

        Returns:
            带相对强弱特征的数据
        """
        # 相对基准的超额收益
        if "return_1d" in df.columns and "benchmark_return_1d" in df.columns:
            df["excess_return_1d"] = df["return_1d"] - df["benchmark_return_1d"]
            df["excess_return_5d"] = df["return_5d"] - df["benchmark_return_5d"]
            df["excess_return_20d"] = df["return_20d"] - df["benchmark_return_20d"]

        # 相对强弱指标
        if "close" in df.columns:
            # RS: 个股涨幅/基准涨幅
            for period in [5, 20]:
                stock_return = df.groupby("code")["close"].pct_change(period)
                benchmark_return = df.get(f"benchmark_return_{period}d", 0)

                df[f"relative_strength_{period}d"] = (1 + stock_return) / (1 + benchmark_return + 1e-10)

        return df

    def _compute_industry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业特征

        Args:
            df: 股票数据

        Returns:
            带行业特征的数据
        """
        if "industry" not in df.columns:
            logger.warning("No industry column found")
            return df

        # 计算每日收益率(如果没有的话)
        if "return_1d" not in df.columns:
            df["return_1d"] = df.groupby("code")["close"].pct_change(1)

        # 行业平均收益率
        for period in self.industry_return_periods:
            col_name = f"return_{period}d" if period > 1 else "return_1d"
            if col_name in df.columns:
                industry_return = df.groupby(["date", "industry"])[col_name].transform("mean")
                df[f"industry_return_{period}d"] = industry_return

                # 个股相对行业的超额收益
                df[f"excess_industry_{period}d"] = df[col_name] - industry_return

        # 行业内排名
        df["rank_in_industry"] = df.groupby(["date", "industry"])["return_1d"].rank(pct=True)

        # 行业动量
        industry_momentum = df.groupby(["date", "industry"])["return_1d"].mean().reset_index()
        industry_momentum = industry_momentum.rename(columns={"return_1d": "industry_momentum"})
        industry_momentum["industry_momentum_rank"] = industry_momentum.groupby("date")[
            "industry_momentum"
        ].rank(pct=True)

        df = df.merge(
            industry_momentum[["date", "industry", "industry_momentum_rank"]],
            on=["date", "industry"],
            how="left"
        )

        return df

    def _compute_market_breadth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场广度特征

        Args:
            df: 股票数据

        Returns:
            带市场广度特征的数据
        """
        # 计算每日统计
        if "return_1d" not in df.columns:
            df["return_1d"] = df.groupby("code")["close"].pct_change(1)

        daily_stats = df.groupby("date").agg({
            "return_1d": ["mean", "std", "count"],
            "code": "count"
        }).reset_index()
        daily_stats.columns = ["date", "market_return", "market_std", "return_count", "stock_count"]

        # 上涨/下跌股票数
        up_count = df[df["return_1d"] > 0].groupby("date")["code"].count().reset_index()
        up_count.columns = ["date", "up_count"]

        down_count = df[df["return_1d"] < 0].groupby("date")["code"].count().reset_index()
        down_count.columns = ["date", "down_count"]

        # 涨停/跌停数
        if "is_limit_up" in df.columns:
            limit_up = df[df["is_limit_up"]].groupby("date")["code"].count().reset_index()
            limit_up.columns = ["date", "limit_up_count"]
        else:
            limit_up = pd.DataFrame({"date": df["date"].unique(), "limit_up_count": 0})

        if "is_limit_down" in df.columns:
            limit_down = df[df["is_limit_down"]].groupby("date")["code"].count().reset_index()
            limit_down.columns = ["date", "limit_down_count"]
        else:
            limit_down = pd.DataFrame({"date": df["date"].unique(), "limit_down_count": 0})

        # 合并统计数据
        daily_stats = daily_stats.merge(up_count, on="date", how="left")
        daily_stats = daily_stats.merge(down_count, on="date", how="left")
        daily_stats = daily_stats.merge(limit_up, on="date", how="left")
        daily_stats = daily_stats.merge(limit_down, on="date", how="left")

        daily_stats = daily_stats.fillna(0)

        # 计算指标
        daily_stats["advance_decline_ratio"] = (
            daily_stats["up_count"] / (daily_stats["down_count"] + 1)
        )
        daily_stats["advance_decline_diff"] = daily_stats["up_count"] - daily_stats["down_count"]
        daily_stats["up_ratio"] = daily_stats["up_count"] / (daily_stats["stock_count"] + 1)
        daily_stats["limit_up_ratio"] = daily_stats["limit_up_count"] / (daily_stats["stock_count"] + 1)

        # 合并到主数据
        merge_cols = [
            "date", "market_return", "market_std",
            "advance_decline_ratio", "advance_decline_diff",
            "up_ratio", "limit_up_ratio", "limit_up_count", "limit_down_count"
        ]
        df = df.merge(daily_stats[merge_cols], on="date", how="left")

        return df

    def _compute_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场状态特征

        Args:
            df: 股票数据

        Returns:
            带市场状态特征的数据
        """
        # 使用基准指数判断市场状态
        if "benchmark_return_20d" not in df.columns:
            return df

        # 市场趋势状态
        conditions = [
            df["benchmark_return_20d"] > 0.05,   # 强势上涨
            df["benchmark_return_20d"] > 0,      # 温和上涨
            df["benchmark_return_20d"] > -0.05,  # 温和下跌
        ]
        choices = [2, 1, -1]
        df["market_regime"] = np.select(conditions, choices, default=-2)

        # 市场波动状态
        if "benchmark_volatility" in df.columns:
            vol_median = df.groupby("date")["benchmark_volatility"].transform("first")
            df["high_vol_regime"] = (df["benchmark_volatility"] > 0.2).astype(int)

        return df

    def _compute_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算截面特征

        Args:
            df: 股票数据

        Returns:
            带截面特征的数据
        """
        # 每日收益率排名
        if "return_1d" in df.columns:
            df["return_rank"] = df.groupby("date")["return_1d"].rank(pct=True)

        # 每日成交量排名
        if "volume" in df.columns:
            df["volume_rank"] = df.groupby("date")["volume"].rank(pct=True)

        # 每日成交额排名
        if "amount" in df.columns:
            df["amount_rank"] = df.groupby("date")["amount"].rank(pct=True)

        # 每日波动率排名
        if "volatility_20d" in df.columns:
            df["volatility_rank"] = df.groupby("date")["volatility_20d"].rank(pct=True)

        return df

    def compute(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        计算所有市场特征

        Args:
            df: 股票数据DataFrame
            index_df: 指数数据DataFrame

        Returns:
            带市场特征的DataFrame
        """
        logger.info("Computing market features...")

        # 基准指数特征
        df = self._compute_benchmark_features(df, index_df)

        # 相对强弱
        df = self._compute_relative_strength(df)

        # 行业特征
        df = self._compute_industry_features(df)

        # 市场广度
        df = self._compute_market_breadth(df)

        # 市场状态
        df = self._compute_market_regime(df)

        # 截面特征
        df = self._compute_cross_sectional_features(df)

        logger.info(f"Market features computed. Shape: {df.shape}")

        return df

    def get_feature_names(self) -> List[str]:
        """
        获取所有市场特征名

        Returns:
            特征名列表
        """
        features = []

        # 基准特征
        features.extend([
            "benchmark_return_1d", "benchmark_return_5d", "benchmark_return_20d",
            "benchmark_volatility", "benchmark_trend"
        ])

        # 相对强弱
        features.extend([
            "excess_return_1d", "excess_return_5d", "excess_return_20d",
            "relative_strength_5d", "relative_strength_20d"
        ])

        # 行业特征
        for period in self.industry_return_periods:
            features.extend([
                f"industry_return_{period}d",
                f"excess_industry_{period}d"
            ])
        features.extend(["rank_in_industry", "industry_momentum_rank"])

        # 市场广度
        features.extend([
            "market_return", "market_std",
            "advance_decline_ratio", "advance_decline_diff",
            "up_ratio", "limit_up_ratio",
            "limit_up_count", "limit_down_count"
        ])

        # 市场状态
        features.extend(["market_regime", "high_vol_regime"])

        # 截面特征
        features.extend([
            "return_rank", "volume_rank", "amount_rank", "volatility_rank"
        ])

        return features
