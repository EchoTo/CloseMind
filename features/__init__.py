"""
特征工程模块
包含技术指标、量价特征、市场特征、Alpha因子和舆情特征
"""

from .technical import TechnicalFeatures
from .volume_price import VolumePriceFeatures
from .market import MarketFeatures
from .alpha import AlphaFeatures
from .sentiment import SentimentFeatures

__all__ = [
    "TechnicalFeatures",
    "VolumePriceFeatures",
    "MarketFeatures",
    "AlphaFeatures",
    "SentimentFeatures",
]


class FeatureEngine:
    """特征工程引擎,整合所有特征计算"""

    def __init__(self, config: dict):
        """
        初始化特征引擎

        Args:
            config: 配置字典
        """
        self.config = config
        self.technical = TechnicalFeatures(config)
        self.volume_price = VolumePriceFeatures(config)
        self.market = MarketFeatures(config)
        self.alpha = AlphaFeatures(config)
        self.sentiment = SentimentFeatures(config)

        # 是否启用舆情特征
        self.use_sentiment = config.get("features", {}).get("sentiment", {}).get("enabled", False)

    def compute_all_features(self, df, index_df=None, include_sentiment=None):
        """
        计算所有特征

        Args:
            df: 股票数据DataFrame
            index_df: 指数数据DataFrame
            include_sentiment: 是否包含舆情特征(None则使用配置)

        Returns:
            带所有特征的DataFrame
        """
        # 技术指标
        df = self.technical.compute(df)

        # 量价特征
        df = self.volume_price.compute(df)

        # 市场特征
        df = self.market.compute(df, index_df)

        # Alpha因子
        df = self.alpha.compute(df)

        # 舆情特征(可选)
        if include_sentiment or (include_sentiment is None and self.use_sentiment):
            df = self.sentiment.compute(df)

        return df
