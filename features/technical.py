"""
技术指标特征
包含MA、MACD、RSI、KDJ、CCI、ATR、Bollinger Bands等
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


class TechnicalFeatures:
    """技术指标特征计算器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        feature_config = config.get("features", {}).get("technical", {})

        # 均线参数
        self.ma_periods = feature_config.get("ma_periods", [5, 10, 20, 30, 60, 120, 250])
        self.ema_periods = feature_config.get("ema_periods", [5, 10, 20, 60])

        # MACD参数
        self.macd_fast = feature_config.get("macd_fast", 12)
        self.macd_slow = feature_config.get("macd_slow", 26)
        self.macd_signal = feature_config.get("macd_signal", 9)

        # RSI参数
        self.rsi_periods = feature_config.get("rsi_periods", [6, 12, 24])

        # KDJ参数
        self.kdj_period = feature_config.get("kdj_period", 9)
        self.kdj_signal = feature_config.get("kdj_signal", 3)

        # CCI参数
        self.cci_period = feature_config.get("cci_period", 20)

        # ATR参数
        self.atr_period = feature_config.get("atr_period", 14)

        # Bollinger Bands参数
        self.boll_period = feature_config.get("boll_period", 20)
        self.boll_std = feature_config.get("boll_std", 2)

        # ROC参数
        self.roc_periods = feature_config.get("roc_periods", [1, 5, 10, 20])

        logger.info("TechnicalFeatures initialized")

    def _compute_ma(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算简单移动平均线

        Args:
            group: 单只股票数据

        Returns:
            带MA特征的数据
        """
        for period in self.ma_periods:
            group[f"ma_{period}"] = group["close"].rolling(window=period, min_periods=1).mean()
            # MA偏离度
            group[f"ma_{period}_bias"] = (group["close"] - group[f"ma_{period}"]) / group[f"ma_{period}"]

        return group

    def _compute_ema(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数移动平均线

        Args:
            group: 单只股票数据

        Returns:
            带EMA特征的数据
        """
        for period in self.ema_periods:
            group[f"ema_{period}"] = group["close"].ewm(span=period, adjust=False).mean()
            # EMA偏离度
            group[f"ema_{period}_bias"] = (group["close"] - group[f"ema_{period}"]) / group[f"ema_{period}"]

        return group

    def _compute_macd(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标

        Args:
            group: 单只股票数据

        Returns:
            带MACD特征的数据
        """
        # 计算快慢EMA
        ema_fast = group["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = group["close"].ewm(span=self.macd_slow, adjust=False).mean()

        # DIF
        group["macd_dif"] = ema_fast - ema_slow

        # DEA (Signal Line)
        group["macd_dea"] = group["macd_dif"].ewm(span=self.macd_signal, adjust=False).mean()

        # MACD Histogram
        group["macd_hist"] = 2 * (group["macd_dif"] - group["macd_dea"])

        # MACD金叉死叉信号
        group["macd_cross"] = np.sign(group["macd_dif"] - group["macd_dea"])
        group["macd_cross_change"] = group["macd_cross"].diff()

        return group

    def _compute_rsi(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标

        Args:
            group: 单只股票数据

        Returns:
            带RSI特征的数据
        """
        delta = group["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        for period in self.rsi_periods:
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            group[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return group

    def _compute_kdj(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            group: 单只股票数据

        Returns:
            带KDJ特征的数据
        """
        # 计算RSV
        low_min = group["low"].rolling(window=self.kdj_period, min_periods=1).min()
        high_max = group["high"].rolling(window=self.kdj_period, min_periods=1).max()

        rsv = (group["close"] - low_min) / (high_max - low_min + 1e-10) * 100

        # 计算K, D, J
        group["kdj_k"] = rsv.ewm(com=self.kdj_signal - 1, adjust=False).mean()
        group["kdj_d"] = group["kdj_k"].ewm(com=self.kdj_signal - 1, adjust=False).mean()
        group["kdj_j"] = 3 * group["kdj_k"] - 2 * group["kdj_d"]

        # KDJ金叉死叉
        group["kdj_cross"] = np.sign(group["kdj_k"] - group["kdj_d"])

        return group

    def _compute_cci(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算CCI指标

        Args:
            group: 单只股票数据

        Returns:
            带CCI特征的数据
        """
        tp = (group["high"] + group["low"] + group["close"]) / 3
        ma_tp = tp.rolling(window=self.cci_period, min_periods=1).mean()
        md_tp = tp.rolling(window=self.cci_period, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        group["cci"] = (tp - ma_tp) / (0.015 * md_tp + 1e-10)

        return group

    def _compute_atr(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算ATR指标

        Args:
            group: 单只股票数据

        Returns:
            带ATR特征的数据
        """
        high = group["high"]
        low = group["low"]
        close = group["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        group["atr"] = tr.rolling(window=self.atr_period, min_periods=1).mean()

        # ATR相对值
        group["atr_ratio"] = group["atr"] / close

        return group

    def _compute_bollinger(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标

        Args:
            group: 单只股票数据

        Returns:
            带布林带特征的数据
        """
        ma = group["close"].rolling(window=self.boll_period, min_periods=1).mean()
        std = group["close"].rolling(window=self.boll_period, min_periods=1).std()

        group["boll_mid"] = ma
        group["boll_upper"] = ma + self.boll_std * std
        group["boll_lower"] = ma - self.boll_std * std

        # 布林带宽度
        group["boll_width"] = (group["boll_upper"] - group["boll_lower"]) / group["boll_mid"]

        # 价格在布林带中的位置
        group["boll_pct"] = (group["close"] - group["boll_lower"]) / (
            group["boll_upper"] - group["boll_lower"] + 1e-10
        )

        return group

    def _compute_roc(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算ROC指标(变化率)

        Args:
            group: 单只股票数据

        Returns:
            带ROC特征的数据
        """
        for period in self.roc_periods:
            group[f"roc_{period}"] = group["close"].pct_change(period)

        return group

    def _compute_obv(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算OBV指标(能量潮)

        Args:
            group: 单只股票数据

        Returns:
            带OBV特征的数据
        """
        direction = np.sign(group["close"].diff())
        direction.iloc[0] = 0

        group["obv"] = (direction * group["volume"]).cumsum()

        # OBV变化率
        group["obv_change"] = group["obv"].pct_change(5)

        return group

    def _compute_williams(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标

        Args:
            group: 单只股票数据

        Returns:
            带威廉指标的数据
        """
        period = 14
        high_max = group["high"].rolling(window=period, min_periods=1).max()
        low_min = group["low"].rolling(window=period, min_periods=1).min()

        group["williams_r"] = (high_max - group["close"]) / (high_max - low_min + 1e-10) * -100

        return group

    def _compute_momentum(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标

        Args:
            group: 单只股票数据

        Returns:
            带动量指标的数据
        """
        for period in [5, 10, 20]:
            group[f"momentum_{period}"] = group["close"] - group["close"].shift(period)
            group[f"momentum_{period}_pct"] = group["close"].pct_change(period)

        return group

    def _compute_volume_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量相关技术指标

        Args:
            group: 单只股票数据

        Returns:
            带成交量指标的数据
        """
        # 成交量移动平均
        for period in [5, 10, 20]:
            group[f"vol_ma_{period}"] = group["volume"].rolling(window=period, min_periods=1).mean()

        # 量比
        group["vol_ratio"] = group["volume"] / group["vol_ma_5"]

        # 成交量变化率
        group["vol_change"] = group["volume"].pct_change()

        return group

    def _compute_price_patterns(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格形态特征

        Args:
            group: 单只股票数据

        Returns:
            带价格形态的数据
        """
        # 实体大小
        group["body"] = abs(group["close"] - group["open"])
        group["body_ratio"] = group["body"] / (group["high"] - group["low"] + 1e-10)

        # 上下影线
        group["upper_shadow"] = group["high"] - group[["open", "close"]].max(axis=1)
        group["lower_shadow"] = group[["open", "close"]].min(axis=1) - group["low"]

        # 影线比例
        total_range = group["high"] - group["low"] + 1e-10
        group["upper_shadow_ratio"] = group["upper_shadow"] / total_range
        group["lower_shadow_ratio"] = group["lower_shadow"] / total_range

        # 阳线阴线
        group["is_red"] = (group["close"] > group["open"]).astype(int)

        # 连续涨跌天数
        group["consecutive_up"] = group["is_red"].groupby(
            (group["is_red"] != group["is_red"].shift()).cumsum()
        ).cumsum() * group["is_red"]

        group["consecutive_down"] = (1 - group["is_red"]).groupby(
            ((1 - group["is_red"]) != (1 - group["is_red"]).shift()).cumsum()
        ).cumsum() * (1 - group["is_red"])

        return group

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标

        Args:
            df: 股票数据DataFrame

        Returns:
            带技术指标的DataFrame
        """
        logger.info("Computing technical features...")

        # 确保按股票和日期排序
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 对每只股票分别计算
        result_groups = []
        for code, group in df.groupby("code"):
            group = group.copy()

            # 计算各类技术指标
            group = self._compute_ma(group)
            group = self._compute_ema(group)
            group = self._compute_macd(group)
            group = self._compute_rsi(group)
            group = self._compute_kdj(group)
            group = self._compute_cci(group)
            group = self._compute_atr(group)
            group = self._compute_bollinger(group)
            group = self._compute_roc(group)
            group = self._compute_obv(group)
            group = self._compute_williams(group)
            group = self._compute_momentum(group)
            group = self._compute_volume_features(group)
            group = self._compute_price_patterns(group)

            result_groups.append(group)

        result = pd.concat(result_groups, ignore_index=True)
        logger.info(f"Technical features computed. Shape: {result.shape}")

        return result

    def get_feature_names(self) -> List[str]:
        """
        获取所有技术指标特征名

        Returns:
            特征名列表
        """
        features = []

        # MA
        for period in self.ma_periods:
            features.extend([f"ma_{period}", f"ma_{period}_bias"])

        # EMA
        for period in self.ema_periods:
            features.extend([f"ema_{period}", f"ema_{period}_bias"])

        # MACD
        features.extend(["macd_dif", "macd_dea", "macd_hist", "macd_cross", "macd_cross_change"])

        # RSI
        for period in self.rsi_periods:
            features.append(f"rsi_{period}")

        # KDJ
        features.extend(["kdj_k", "kdj_d", "kdj_j", "kdj_cross"])

        # CCI
        features.append("cci")

        # ATR
        features.extend(["atr", "atr_ratio"])

        # Bollinger
        features.extend(["boll_mid", "boll_upper", "boll_lower", "boll_width", "boll_pct"])

        # ROC
        for period in self.roc_periods:
            features.append(f"roc_{period}")

        # OBV
        features.extend(["obv", "obv_change"])

        # Williams
        features.append("williams_r")

        # Momentum
        for period in [5, 10, 20]:
            features.extend([f"momentum_{period}", f"momentum_{period}_pct"])

        # Volume
        for period in [5, 10, 20]:
            features.append(f"vol_ma_{period}")
        features.extend(["vol_ratio", "vol_change"])

        # Price patterns
        features.extend([
            "body", "body_ratio", "upper_shadow", "lower_shadow",
            "upper_shadow_ratio", "lower_shadow_ratio", "is_red",
            "consecutive_up", "consecutive_down"
        ])

        return features
