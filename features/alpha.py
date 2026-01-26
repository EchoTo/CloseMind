"""
Alpha因子
基于WorldQuant 101 Alpha实现,针对A股优化
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


class AlphaFeatures:
    """Alpha因子计算器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        alpha_config = config.get("features", {}).get("alpha", {})

        self.enabled = alpha_config.get("enabled", True)
        self.factors = alpha_config.get("factors", "all")

        logger.info("AlphaFeatures initialized")

    # ==================== 辅助函数 ====================

    def _ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        """时序排名"""
        return series.rolling(window, min_periods=1).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )

    def _ts_max(self, series: pd.Series, window: int) -> pd.Series:
        """时序最大值"""
        return series.rolling(window, min_periods=1).max()

    def _ts_min(self, series: pd.Series, window: int) -> pd.Series:
        """时序最小值"""
        return series.rolling(window, min_periods=1).min()

    def _ts_argmax(self, series: pd.Series, window: int) -> pd.Series:
        """时序最大值位置"""
        return series.rolling(window, min_periods=1).apply(np.argmax, raw=True) + 1

    def _ts_argmin(self, series: pd.Series, window: int) -> pd.Series:
        """时序最小值位置"""
        return series.rolling(window, min_periods=1).apply(np.argmin, raw=True) + 1

    def _ts_sum(self, series: pd.Series, window: int) -> pd.Series:
        """时序求和"""
        return series.rolling(window, min_periods=1).sum()

    def _ts_mean(self, series: pd.Series, window: int) -> pd.Series:
        """时序均值"""
        return series.rolling(window, min_periods=1).mean()

    def _ts_std(self, series: pd.Series, window: int) -> pd.Series:
        """时序标准差"""
        return series.rolling(window, min_periods=1).std()

    def _ts_corr(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """时序相关性"""
        return x.rolling(window, min_periods=1).corr(y)

    def _ts_cov(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """时序协方差"""
        return x.rolling(window, min_periods=1).cov(y)

    def _delay(self, series: pd.Series, periods: int) -> pd.Series:
        """延迟"""
        return series.shift(periods)

    def _delta(self, series: pd.Series, periods: int) -> pd.Series:
        """差分"""
        return series.diff(periods)

    def _decay_linear(self, series: pd.Series, window: int) -> pd.Series:
        """线性衰减加权"""
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()

        def weighted_mean(x):
            if len(x) < window:
                w = np.arange(1, len(x) + 1, dtype=float)
                w = w / w.sum()
                return np.dot(x, w)
            return np.dot(x, weights)

        return series.rolling(window, min_periods=1).apply(weighted_mean, raw=True)

    def _scale(self, series: pd.Series) -> pd.Series:
        """标准化到和为1"""
        abs_sum = series.abs().sum()
        if abs_sum == 0:
            return series
        return series / abs_sum

    def _sign(self, series: pd.Series) -> pd.Series:
        """符号函数"""
        return np.sign(series)

    # ==================== Alpha因子 ====================

    def _alpha001(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        简化版: 近期低点的位置
        """
        close = group["close"]
        returns = close.pct_change()
        std = self._ts_std(returns, 20)

        inner = np.where(returns < 0, std, close)
        inner_squared = inner ** 2

        return self._ts_argmax(pd.Series(inner_squared, index=close.index), 5) - 0.5

    def _alpha002(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        """
        volume = group["volume"]
        close = group["close"]
        open_ = group["open"]

        delta_log_vol = self._delta(np.log(volume + 1), 2)
        price_change = (close - open_) / (open_ + 1e-10)

        return -1 * self._ts_corr(delta_log_vol, price_change, 6)

    def _alpha003(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        """
        return -1 * self._ts_corr(group["open"], group["volume"], 10)

    def _alpha004(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        """
        return -1 * self._ts_rank(group["low"], 9)

    def _alpha005(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        简化版使用typical price代替vwap
        """
        typical_price = (group["high"] + group["low"] + group["close"]) / 3
        vwap = typical_price  # 简化

        part1 = group["open"] - self._ts_mean(vwap, 10)
        part2 = -1 * abs(group["close"] - vwap)

        return part1 * part2

    def _alpha006(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#6: (-1 * correlation(open, volume, 10))
        """
        return -1 * self._ts_corr(group["open"], group["volume"], 10)

    def _alpha007(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        """
        adv20 = self._ts_mean(group["volume"], 20)
        delta_close = self._delta(group["close"], 7)

        condition = adv20 < group["volume"]
        true_val = -1 * self._ts_rank(abs(delta_close), 60) * self._sign(delta_close)

        return np.where(condition, true_val, -1)

    def _alpha008(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        """
        returns = group["close"].pct_change()
        sum_open = self._ts_sum(group["open"], 5)
        sum_ret = self._ts_sum(returns, 5)
        product = sum_open * sum_ret

        return -1 * (product - self._delay(product, 10))

    def _alpha009(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        """
        delta_close = self._delta(group["close"], 1)
        ts_min_delta = self._ts_min(delta_close, 5)
        ts_max_delta = self._ts_max(delta_close, 5)

        condition1 = ts_min_delta > 0
        condition2 = ts_max_delta < 0

        return np.where(condition1, delta_close,
                       np.where(condition2, delta_close, -1 * delta_close))

    def _alpha010(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        """
        delta_close = self._delta(group["close"], 1)
        ts_min_delta = self._ts_min(delta_close, 4)
        ts_max_delta = self._ts_max(delta_close, 4)

        condition1 = ts_min_delta > 0
        condition2 = ts_max_delta < 0

        return np.where(condition1, delta_close,
                       np.where(condition2, delta_close, -1 * delta_close))

    def _alpha012(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        """
        return self._sign(self._delta(group["volume"], 1)) * (-1 * self._delta(group["close"], 1))

    def _alpha013(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        """
        return -1 * self._ts_cov(group["close"], group["volume"], 5)

    def _alpha014(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        """
        returns = group["close"].pct_change()
        return -1 * self._delta(returns, 3) * self._ts_corr(group["open"], group["volume"], 10)

    def _alpha015(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """
        corr = self._ts_corr(group["high"], group["volume"], 3)
        return -1 * self._ts_sum(corr, 3)

    def _alpha016(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        """
        return -1 * self._ts_cov(group["high"], group["volume"], 5)

    def _alpha017(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        """
        adv20 = self._ts_mean(group["volume"], 20)
        ts_rank_close = self._ts_rank(group["close"], 10)
        delta_delta = self._delta(self._delta(group["close"], 1), 1)
        vol_ratio = group["volume"] / (adv20 + 1)

        return -1 * ts_rank_close * delta_delta * self._ts_rank(vol_ratio, 5)

    def _alpha018(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        """
        close_open = group["close"] - group["open"]
        std_abs = self._ts_std(abs(close_open), 5)
        corr = self._ts_corr(group["close"], group["open"], 10)

        return -1 * (std_abs + close_open + corr)

    def _alpha019(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        """
        returns = group["close"].pct_change()
        price_change = group["close"] - self._delay(group["close"], 7) + self._delta(group["close"], 7)

        return -1 * self._sign(price_change) * (1 + self._ts_sum(returns, 250))

    def _alpha020(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        """
        part1 = -1 * (group["open"] - self._delay(group["high"], 1))
        part2 = group["open"] - self._delay(group["close"], 1)
        part3 = group["open"] - self._delay(group["low"], 1)

        return part1 * part2 * part3

    def _alpha021(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#21: 均线趋势
        """
        close = group["close"]
        ma8 = self._ts_mean(close, 8)
        ma2 = self._ts_mean(close, 2)

        condition1 = (self._ts_mean(close, 8) + self._ts_std(close, 8)) < self._ts_mean(close, 2)
        condition2 = self._ts_mean(close, 2) < (self._ts_mean(close, 8) - self._ts_std(close, 8))

        return np.where(condition1, -1, np.where(condition2, 1, 0))

    def _alpha023(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#23: 高点突破
        """
        high = group["high"]
        ma_high = self._ts_mean(high, 20)

        condition = high > ma_high
        return np.where(condition, -1 * self._delta(high, 2), 0)

    def _alpha024(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#24: 价格动量
        """
        close = group["close"]
        ma_close = self._ts_mean(close, 100)

        delta_ma = self._delta(ma_close, 100) / 100
        condition = delta_ma < (close - ma_close)

        return np.where(condition, -1 * (close - self._delay(close, 3)), 0)

    def _alpha026(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """
        ts_rank_vol = self._ts_rank(group["volume"], 5)
        ts_rank_high = self._ts_rank(group["high"], 5)
        corr = self._ts_corr(ts_rank_vol, ts_rank_high, 5)

        return -1 * self._ts_max(corr, 3)

    def _alpha028(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#28: 量价相关性
        """
        adv20 = self._ts_mean(group["volume"], 20)
        high_low_corr = self._ts_corr(adv20, group["low"], 5)
        mean_result = (high_low_corr + (group["high"] + group["low"]) / 2 - group["close"])

        return self._scale(mean_result)

    def _alpha029(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#29: 反转因子
        """
        returns = group["close"].pct_change()
        log_sum = self._ts_sum(np.log(1 + returns + 1e-10), 5)

        return self._ts_min(log_sum, 5)

    def _alpha030(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#30: 成交量变化
        """
        close = group["close"]
        volume = group["volume"]

        sign_delta_close = self._sign(self._delta(close, 1))
        sign_delta_vol = self._sign(self._delta(volume, 1))
        sign_delta_close_delay = self._sign(self._delta(self._delay(close, 1), 1))

        return ((1 - (close - self._delay(close, 5)) /
                (self._delay(close, 5) + 1e-10)) *
               (1 - (volume - self._delay(volume, 5)) /
                (self._delay(volume, 5) + 1)))

    def _alpha031(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#31: 价格偏离
        """
        close = group["close"]
        low = group["low"]
        adv20 = self._ts_mean(group["volume"], 20)

        part1 = self._decay_linear(close - self._ts_mean(close, 12), 10)
        part2 = self._ts_corr(group["volume"], low, 12)
        part3 = self._sign(self._delta(close, 3))

        return part1 + part2 * part3

    def _alpha033(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#33: 开盘价位置
        """
        open_ = group["open"]
        close = group["close"]

        return -(1 - (open_ / (close + 1e-10))) ** 2

    def _alpha034(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#34: 收益率反转
        """
        returns = group["close"].pct_change()
        std_ret = self._ts_std(returns, 2) / (self._ts_std(returns, 5) + 1e-10)

        return (1 - std_ret) * (1 - (returns - self._ts_mean(returns, 5)))

    def _alpha035(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#35: 量价排名
        """
        volume = group["volume"]
        close = group["close"]

        ts_rank_vol = self._ts_rank(volume, 32)
        ts_rank_close = self._ts_rank(close, 32)

        return ts_rank_vol * (1 - ts_rank_close) * (1 - self._ts_rank(close.pct_change(), 16))

    def _alpha037(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#37: 开盘收盘相关
        """
        return (
            self._ts_rank(self._ts_corr(self._delay(group["open"], 1), group["close"], 200), 5) +
            (group["open"] - group["close"])
        )

    def _alpha038(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#38: 高点反转
        """
        return (
            -1 * self._ts_rank(group["close"], 10) *
            (self._delta(group["close"], 1) / (group["close"] + 1e-10))
        )

    def _alpha039(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#39: 动量衰减
        """
        returns = group["close"].pct_change()
        adv20 = self._ts_mean(group["volume"], 20)

        return (
            -1 * (self._delta(group["close"], 7) *
                  (1 - self._ts_rank(self._decay_linear(group["volume"] / adv20, 9), 5))) *
            (1 + self._ts_rank(self._ts_sum(returns, 250), 250))
        )

    def _alpha040(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#40: 高点成交量
        """
        high = group["high"]
        volume = group["volume"]

        return -1 * self._ts_std(high, 10) * self._ts_corr(high, volume, 10)

    def _alpha041(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#41: 最高价最低价差
        """
        return (group["high"] - group["low"]) ** 0.5 - group["volume"] / (group["close"] + 1e-10)

    def _alpha042(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#42: 量价背离
        """
        typical_price = (group["high"] + group["low"] + group["close"]) / 3
        return (typical_price - group["close"]) / (group["high"] - group["low"] + 1e-10)

    def _alpha043(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#43: 成交量趋势
        """
        adv20 = self._ts_mean(group["volume"], 20)
        return self._ts_rank(group["volume"] / adv20, 20) * self._ts_rank(-1 * self._delta(group["close"], 7), 8)

    def _alpha044(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#44: 高点回落
        """
        high = group["high"]
        volume = group["volume"]

        return -1 * self._ts_corr(high, self._ts_rank(volume, 5), 5)

    def _alpha045(self, group: pd.DataFrame) -> pd.Series:
        """
        Alpha#45: 收盘价延迟
        """
        close = group["close"]
        return -1 * (self._ts_mean(self._delay(close, 5), 20) *
                     self._ts_corr(close, group["volume"], 2) *
                     self._ts_rank(self._ts_corr(close, group["high"], 5), 3))

    def _compute_all_alphas(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有Alpha因子

        Args:
            group: 单只股票数据

        Returns:
            带Alpha因子的数据
        """
        # 定义所有Alpha函数
        alpha_funcs = {
            "alpha_001": self._alpha001,
            "alpha_002": self._alpha002,
            "alpha_003": self._alpha003,
            "alpha_004": self._alpha004,
            "alpha_005": self._alpha005,
            "alpha_006": self._alpha006,
            "alpha_007": self._alpha007,
            "alpha_008": self._alpha008,
            "alpha_009": self._alpha009,
            "alpha_010": self._alpha010,
            "alpha_012": self._alpha012,
            "alpha_013": self._alpha013,
            "alpha_014": self._alpha014,
            "alpha_015": self._alpha015,
            "alpha_016": self._alpha016,
            "alpha_017": self._alpha017,
            "alpha_018": self._alpha018,
            "alpha_019": self._alpha019,
            "alpha_020": self._alpha020,
            "alpha_021": self._alpha021,
            "alpha_023": self._alpha023,
            "alpha_024": self._alpha024,
            "alpha_026": self._alpha026,
            "alpha_028": self._alpha028,
            "alpha_029": self._alpha029,
            "alpha_030": self._alpha030,
            "alpha_031": self._alpha031,
            "alpha_033": self._alpha033,
            "alpha_034": self._alpha034,
            "alpha_035": self._alpha035,
            "alpha_037": self._alpha037,
            "alpha_038": self._alpha038,
            "alpha_039": self._alpha039,
            "alpha_040": self._alpha040,
            "alpha_041": self._alpha041,
            "alpha_042": self._alpha042,
            "alpha_043": self._alpha043,
            "alpha_044": self._alpha044,
            "alpha_045": self._alpha045,
        }

        # 计算每个Alpha
        for name, func in alpha_funcs.items():
            try:
                group[name] = func(group)
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                group[name] = np.nan

        return group

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有Alpha因子

        Args:
            df: 股票数据DataFrame

        Returns:
            带Alpha因子的DataFrame
        """
        if not self.enabled:
            logger.info("Alpha features disabled")
            return df

        logger.info("Computing Alpha features...")

        # 确保按股票和日期排序
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 对每只股票分别计算
        result_groups = []
        for code, group in df.groupby("code"):
            group = group.copy()
            group = self._compute_all_alphas(group)
            result_groups.append(group)

        result = pd.concat(result_groups, ignore_index=True)

        # 处理无穷值
        alpha_cols = [c for c in result.columns if c.startswith("alpha_")]
        for col in alpha_cols:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        logger.info(f"Alpha features computed. Shape: {result.shape}")

        return result

    def get_feature_names(self) -> List[str]:
        """
        获取所有Alpha特征名

        Returns:
            特征名列表
        """
        return [
            "alpha_001", "alpha_002", "alpha_003", "alpha_004", "alpha_005",
            "alpha_006", "alpha_007", "alpha_008", "alpha_009", "alpha_010",
            "alpha_012", "alpha_013", "alpha_014", "alpha_015", "alpha_016",
            "alpha_017", "alpha_018", "alpha_019", "alpha_020", "alpha_021",
            "alpha_023", "alpha_024", "alpha_026", "alpha_028", "alpha_029",
            "alpha_030", "alpha_031", "alpha_033", "alpha_034", "alpha_035",
            "alpha_037", "alpha_038", "alpha_039", "alpha_040", "alpha_041",
            "alpha_042", "alpha_043", "alpha_044", "alpha_045",
        ]
