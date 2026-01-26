"""
量价特征
包含收益率、波动率、资金流等特征
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


class VolumePriceFeatures:
    """量价特征计算器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        feature_config = config.get("features", {}).get("volume_price", {})

        # 收益率周期
        self.return_periods = feature_config.get("return_periods", [1, 2, 3, 5, 10, 20, 60])

        # 波动率周期
        self.volatility_periods = feature_config.get("volatility_periods", [5, 10, 20, 60])

        # 成交量均线周期
        self.volume_ma_periods = feature_config.get("volume_ma_periods", [5, 10, 20])

        logger.info("VolumePriceFeatures initialized")

    def _compute_returns(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率特征

        Args:
            group: 单只股票数据

        Returns:
            带收益率特征的数据
        """
        # 对数收益率
        group["log_return"] = np.log(group["close"] / group["close"].shift(1))

        # 不同周期收益率
        for period in self.return_periods:
            # 简单收益率
            group[f"return_{period}d"] = group["close"].pct_change(period)

            # 对数收益率
            group[f"log_return_{period}d"] = np.log(group["close"] / group["close"].shift(period))

        # 收益率排名(用于后续截面标准化)
        # 这里先计算原始值,截面标准化在后续处理

        return group

    def _compute_volatility(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率特征

        Args:
            group: 单只股票数据

        Returns:
            带波动率特征的数据
        """
        # 日内波动率
        group["intraday_vol"] = (group["high"] - group["low"]) / group["open"]

        # 真实波动幅度(True Range)
        high_low = group["high"] - group["low"]
        high_close = abs(group["high"] - group["close"].shift(1))
        low_close = abs(group["low"] - group["close"].shift(1))
        group["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 历史波动率(不同周期)
        for period in self.volatility_periods:
            # 收益率标准差
            group[f"volatility_{period}d"] = group["log_return"].rolling(
                window=period, min_periods=1
            ).std() * np.sqrt(252)  # 年化

            # 波动率变化
            if period >= 5:
                group[f"volatility_{period}d_change"] = group[f"volatility_{period}d"].pct_change(5)

        # 振幅
        group["amplitude"] = (group["high"] - group["low"]) / group["close"].shift(1)

        # 波动率比率
        if "volatility_5d" in group.columns and "volatility_20d" in group.columns:
            group["vol_ratio_5_20"] = group["volatility_5d"] / (group["volatility_20d"] + 1e-10)

        return group

    def _compute_volume_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量相关特征

        Args:
            group: 单只股票数据

        Returns:
            带成交量特征的数据
        """
        # 成交量变化率
        for period in [1, 5, 10, 20]:
            group[f"volume_change_{period}d"] = group["volume"].pct_change(period)

        # 成交量移动平均
        for period in self.volume_ma_periods:
            group[f"volume_ma_{period}"] = group["volume"].rolling(
                window=period, min_periods=1
            ).mean()

        # 相对成交量
        group["relative_volume_5"] = group["volume"] / (group["volume_ma_5"] + 1)
        group["relative_volume_20"] = group["volume"] / (group["volume_ma_20"] + 1)

        # 成交量趋势
        group["volume_trend"] = group["volume_ma_5"] / (group["volume_ma_20"] + 1)

        # 成交量波动率
        group["volume_volatility"] = group["volume"].rolling(
            window=20, min_periods=1
        ).std() / (group["volume"].rolling(window=20, min_periods=1).mean() + 1)

        return group

    def _compute_amount_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交额相关特征

        Args:
            group: 单只股票数据

        Returns:
            带成交额特征的数据
        """
        if "amount" not in group.columns:
            return group

        # 成交额移动平均
        for period in [5, 10, 20]:
            group[f"amount_ma_{period}"] = group["amount"].rolling(
                window=period, min_periods=1
            ).mean()

        # 成交额变化率
        group["amount_change_1d"] = group["amount"].pct_change(1)
        group["amount_change_5d"] = group["amount"].pct_change(5)

        # 相对成交额
        group["relative_amount"] = group["amount"] / (group["amount_ma_20"] + 1)

        # 平均成交价
        group["avg_price"] = group["amount"] / (group["volume"] + 1)

        # 平均成交价与收盘价比较
        group["avg_price_ratio"] = group["avg_price"] / (group["close"] + 1e-10)

        return group

    def _compute_vwap(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算VWAP相关特征

        Args:
            group: 单只股票数据

        Returns:
            带VWAP特征的数据
        """
        # 典型价格
        group["typical_price"] = (group["high"] + group["low"] + group["close"]) / 3

        # VWAP
        if "amount" in group.columns:
            # 使用成交额计算
            cumulative_amount = group["amount"].rolling(window=20, min_periods=1).sum()
            cumulative_volume = group["volume"].rolling(window=20, min_periods=1).sum()
            group["vwap_20"] = cumulative_amount / (cumulative_volume + 1)
        else:
            # 使用典型价格和成交量近似
            group["vwap_20"] = (
                group["typical_price"] * group["volume"]
            ).rolling(window=20, min_periods=1).sum() / (
                group["volume"].rolling(window=20, min_periods=1).sum() + 1
            )

        # 价格与VWAP偏离度
        group["vwap_bias"] = (group["close"] - group["vwap_20"]) / (group["vwap_20"] + 1e-10)

        return group

    def _compute_price_position(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格位置特征

        Args:
            group: 单只股票数据

        Returns:
            带价格位置特征的数据
        """
        # 不同周期的最高最低价
        for period in [5, 10, 20, 60]:
            high_max = group["high"].rolling(window=period, min_periods=1).max()
            low_min = group["low"].rolling(window=period, min_periods=1).min()

            # 价格在范围中的位置(0-1)
            group[f"price_position_{period}d"] = (group["close"] - low_min) / (
                high_max - low_min + 1e-10
            )

            # 距离最高点的回撤
            group[f"drawdown_{period}d"] = (group["close"] - high_max) / (high_max + 1e-10)

            # 距离最低点的反弹
            group[f"rally_{period}d"] = (group["close"] - low_min) / (low_min + 1e-10)

        # 创新高/新低
        group["new_high_20"] = (group["close"] >= group["high"].rolling(20, min_periods=1).max()).astype(int)
        group["new_low_20"] = (group["close"] <= group["low"].rolling(20, min_periods=1).min()).astype(int)

        return group

    def _compute_momentum_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量特征

        Args:
            group: 单只股票数据

        Returns:
            带动量特征的数据
        """
        # 价格动量
        for period in [5, 10, 20]:
            group[f"price_momentum_{period}d"] = group["close"] / group["close"].shift(period) - 1

        # 成交量动量
        for period in [5, 10, 20]:
            group[f"volume_momentum_{period}d"] = group["volume"] / (
                group["volume"].shift(period) + 1
            ) - 1

        # 量价配合度
        group["vol_price_corr_10"] = group["close"].rolling(10, min_periods=5).corr(group["volume"])
        group["vol_price_corr_20"] = group["close"].rolling(20, min_periods=10).corr(group["volume"])

        # 资金流向估计(简化版)
        # 上涨时的成交量 vs 下跌时的成交量
        group["price_direction"] = np.sign(group["close"].diff())
        group["up_volume"] = group["volume"] * (group["price_direction"] > 0)
        group["down_volume"] = group["volume"] * (group["price_direction"] < 0)

        group["money_flow_10"] = (
            group["up_volume"].rolling(10, min_periods=1).sum() /
            (group["down_volume"].rolling(10, min_periods=1).sum() + 1)
        )

        # 清理临时列
        group = group.drop(columns=["price_direction", "up_volume", "down_volume"])

        return group

    def _compute_turnover_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算换手率相关特征

        Args:
            group: 单只股票数据

        Returns:
            带换手率特征的数据
        """
        if "turnover" not in group.columns:
            return group

        # 换手率移动平均
        for period in [5, 10, 20]:
            group[f"turnover_ma_{period}"] = group["turnover"].rolling(
                window=period, min_periods=1
            ).mean()

        # 相对换手率
        group["relative_turnover"] = group["turnover"] / (group["turnover_ma_20"] + 1e-10)

        # 换手率变化
        group["turnover_change"] = group["turnover"].pct_change()

        # 累计换手率
        group["cumulative_turnover_5"] = group["turnover"].rolling(5, min_periods=1).sum()
        group["cumulative_turnover_20"] = group["turnover"].rolling(20, min_periods=1).sum()

        return group

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有量价特征

        Args:
            df: 股票数据DataFrame

        Returns:
            带量价特征的DataFrame
        """
        logger.info("Computing volume-price features...")

        # 确保按股票和日期排序
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 对每只股票分别计算
        result_groups = []
        for code, group in df.groupby("code"):
            group = group.copy()

            # 计算各类量价特征
            group = self._compute_returns(group)
            group = self._compute_volatility(group)
            group = self._compute_volume_features(group)
            group = self._compute_amount_features(group)
            group = self._compute_vwap(group)
            group = self._compute_price_position(group)
            group = self._compute_momentum_features(group)
            group = self._compute_turnover_features(group)

            result_groups.append(group)

        result = pd.concat(result_groups, ignore_index=True)

        # 截面标准化(可选)
        # result = self._cross_sectional_standardize(result)

        logger.info(f"Volume-price features computed. Shape: {result.shape}")

        return result

    def _cross_sectional_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        截面标准化(每日z-score)

        Args:
            df: 数据DataFrame

        Returns:
            标准化后的数据
        """
        feature_cols = self.get_feature_names()
        existing_cols = [c for c in feature_cols if c in df.columns]

        for col in existing_cols:
            df[f"{col}_zscore"] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )

        return df

    def get_feature_names(self) -> List[str]:
        """
        获取所有量价特征名

        Returns:
            特征名列表
        """
        features = ["log_return"]

        # 收益率
        for period in self.return_periods:
            features.extend([f"return_{period}d", f"log_return_{period}d"])

        # 波动率
        features.extend(["intraday_vol", "true_range", "amplitude"])
        for period in self.volatility_periods:
            features.append(f"volatility_{period}d")
            if period >= 5:
                features.append(f"volatility_{period}d_change")
        features.append("vol_ratio_5_20")

        # 成交量
        for period in [1, 5, 10, 20]:
            features.append(f"volume_change_{period}d")
        for period in self.volume_ma_periods:
            features.append(f"volume_ma_{period}")
        features.extend([
            "relative_volume_5", "relative_volume_20",
            "volume_trend", "volume_volatility"
        ])

        # 成交额
        for period in [5, 10, 20]:
            features.append(f"amount_ma_{period}")
        features.extend([
            "amount_change_1d", "amount_change_5d",
            "relative_amount", "avg_price", "avg_price_ratio"
        ])

        # VWAP
        features.extend(["typical_price", "vwap_20", "vwap_bias"])

        # 价格位置
        for period in [5, 10, 20, 60]:
            features.extend([
                f"price_position_{period}d",
                f"drawdown_{period}d",
                f"rally_{period}d"
            ])
        features.extend(["new_high_20", "new_low_20"])

        # 动量
        for period in [5, 10, 20]:
            features.extend([
                f"price_momentum_{period}d",
                f"volume_momentum_{period}d"
            ])
        features.extend(["vol_price_corr_10", "vol_price_corr_20", "money_flow_10"])

        # 换手率
        for period in [5, 10, 20]:
            features.append(f"turnover_ma_{period}")
        features.extend([
            "relative_turnover", "turnover_change",
            "cumulative_turnover_5", "cumulative_turnover_20"
        ])

        return features
