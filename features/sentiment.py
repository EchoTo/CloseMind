"""
舆情特征模块
获取和处理新闻、公告、舆论等数据
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False


class SentimentFeatures:
    """舆情特征计算器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        paths = config.get("paths", {})
        self.data_dir = Path(paths.get("raw_data_dir", "./data_storage/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 情感词典(简化版)
        self.positive_words = [
            "上涨", "涨停", "突破", "新高", "利好", "增长", "盈利", "超预期",
            "增持", "回购", "分红", "业绩预增", "订单", "中标", "合作",
            "收购", "并购", "重组", "获批", "通过", "成功", "首次"
        ]
        self.negative_words = [
            "下跌", "跌停", "破位", "新低", "利空", "下滑", "亏损", "不及预期",
            "减持", "质押", "ST", "退市", "处罚", "违规", "诉讼",
            "调查", "风险", "终止", "失败", "暂停", "延期"
        ]

        logger.info("SentimentFeatures initialized")

    def fetch_stock_news(
        self,
        stock_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取个股新闻

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            新闻DataFrame
        """
        if not HAS_AKSHARE:
            logger.warning("akshare not installed, returning empty DataFrame")
            return pd.DataFrame()

        try:
            # 获取个股新闻
            news_df = ak.stock_news_em(symbol=stock_code)

            if news_df is None or len(news_df) == 0:
                return pd.DataFrame()

            # 标准化列名
            news_df = news_df.rename(columns={
                "新闻标题": "title",
                "新闻内容": "content",
                "发布时间": "publish_time",
                "文章来源": "source"
            })

            news_df["code"] = stock_code
            news_df["publish_time"] = pd.to_datetime(news_df["publish_time"])
            news_df["date"] = news_df["publish_time"].dt.date

            # 日期过滤
            if start_date:
                news_df = news_df[news_df["date"] >= pd.to_datetime(start_date).date()]
            if end_date:
                news_df = news_df[news_df["date"] <= pd.to_datetime(end_date).date()]

            return news_df

        except Exception as e:
            logger.debug(f"Failed to fetch news for {stock_code}: {e}")
            return pd.DataFrame()

    def fetch_stock_announcements(
        self,
        stock_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取公司公告

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            公告DataFrame
        """
        if not HAS_AKSHARE:
            return pd.DataFrame()

        try:
            # 获取公告
            ann_df = ak.stock_notice_report(symbol=stock_code)

            if ann_df is None or len(ann_df) == 0:
                return pd.DataFrame()

            # 标准化
            ann_df = ann_df.rename(columns={
                "公告标题": "title",
                "公告时间": "publish_time",
                "公告类型": "type"
            })

            ann_df["code"] = stock_code
            ann_df["publish_time"] = pd.to_datetime(ann_df["publish_time"])
            ann_df["date"] = ann_df["publish_time"].dt.date

            return ann_df

        except Exception as e:
            logger.debug(f"Failed to fetch announcements for {stock_code}: {e}")
            return pd.DataFrame()

    def fetch_market_sentiment(self, date: str = None) -> Dict[str, float]:
        """
        获取市场情绪指标

        Args:
            date: 日期

        Returns:
            市场情绪指标字典
        """
        if not HAS_AKSHARE:
            return {}

        try:
            # 融资融券余额(市场杠杆情绪)
            margin_df = ak.stock_margin_sse(start_date=date, end_date=date)

            # 北向资金(外资情绪)
            north_df = ak.stock_hsgt_north_net_flow_in_em(symbol="沪股通")

            sentiment = {}

            if margin_df is not None and len(margin_df) > 0:
                sentiment["margin_balance"] = margin_df["融资余额"].iloc[-1]
                sentiment["margin_buy"] = margin_df["融资买入额"].iloc[-1]

            if north_df is not None and len(north_df) > 0:
                sentiment["north_flow"] = north_df["当日净流入"].iloc[-1]

            return sentiment

        except Exception as e:
            logger.debug(f"Failed to fetch market sentiment: {e}")
            return {}

    def calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        计算文本情感分数

        Args:
            text: 文本内容

        Returns:
            情感分数字典
        """
        if not text or pd.isna(text):
            return {"positive": 0, "negative": 0, "score": 0}

        text = str(text)

        # 统计情感词
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)

        total = positive_count + negative_count
        if total == 0:
            score = 0
        else:
            score = (positive_count - negative_count) / total

        return {
            "positive": positive_count,
            "negative": negative_count,
            "score": score
        }

    def compute_news_features(
        self,
        news_df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """
        计算新闻特征

        Args:
            news_df: 新闻数据
            stock_code: 股票代码

        Returns:
            新闻特征DataFrame
        """
        if news_df is None or len(news_df) == 0:
            return pd.DataFrame()

        # 计算每条新闻的情感
        sentiments = news_df["title"].apply(self.calculate_sentiment_score)
        news_df["sentiment_score"] = [s["score"] for s in sentiments]
        news_df["positive_words"] = [s["positive"] for s in sentiments]
        news_df["negative_words"] = [s["negative"] for s in sentiments]

        # 按日期聚合
        daily_features = news_df.groupby("date").agg({
            "title": "count",  # 新闻数量
            "sentiment_score": ["mean", "sum", "std"],
            "positive_words": "sum",
            "negative_words": "sum"
        }).reset_index()

        daily_features.columns = [
            "date", "news_count", "sentiment_mean", "sentiment_sum",
            "sentiment_std", "positive_total", "negative_total"
        ]

        daily_features["code"] = stock_code

        return daily_features

    def compute_announcement_features(
        self,
        ann_df: pd.DataFrame,
        stock_code: str
    ) -> pd.DataFrame:
        """
        计算公告特征

        Args:
            ann_df: 公告数据
            stock_code: 股票代码

        Returns:
            公告特征DataFrame
        """
        if ann_df is None or len(ann_df) == 0:
            return pd.DataFrame()

        # 公告类型权重
        type_weights = {
            "业绩预告": 2.0,
            "业绩快报": 2.0,
            "年报": 1.5,
            "季报": 1.5,
            "分红": 1.2,
            "增持": 1.3,
            "减持": -1.3,
            "股权质押": -0.8,
            "风险提示": -1.0,
        }

        # 计算公告情感
        def get_announcement_weight(title, ann_type):
            weight = 0
            # 类型权重
            for key, w in type_weights.items():
                if key in str(ann_type) or key in str(title):
                    weight += w
            # 标题情感
            sentiment = self.calculate_sentiment_score(title)
            weight += sentiment["score"]
            return weight

        ann_df["ann_weight"] = ann_df.apply(
            lambda x: get_announcement_weight(x.get("title", ""), x.get("type", "")),
            axis=1
        )

        # 按日期聚合
        daily_features = ann_df.groupby("date").agg({
            "title": "count",
            "ann_weight": ["sum", "mean"]
        }).reset_index()

        daily_features.columns = [
            "date", "ann_count", "ann_weight_sum", "ann_weight_mean"
        ]

        daily_features["code"] = stock_code

        return daily_features

    def fetch_all_sentiment_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        批量获取舆情数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            舆情特征DataFrame
        """
        logger.info(f"Fetching sentiment data for {len(stock_codes)} stocks...")

        all_features = []

        for i, code in enumerate(stock_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(stock_codes)}")

            try:
                # 获取新闻
                news = self.fetch_stock_news(code, start_date, end_date)
                if len(news) > 0:
                    news_features = self.compute_news_features(news, code)
                    if len(news_features) > 0:
                        all_features.append(news_features)

                # 获取公告
                ann = self.fetch_stock_announcements(code, start_date, end_date)
                if len(ann) > 0:
                    ann_features = self.compute_announcement_features(ann, code)
                    if len(ann_features) > 0:
                        all_features.append(ann_features)

                time.sleep(0.1)  # 避免请求过快

            except Exception as e:
                logger.debug(f"Error processing {code}: {e}")
                continue

        if all_features:
            result = pd.concat(all_features, ignore_index=True)
            # 按股票和日期聚合(合并新闻和公告特征)
            result = result.groupby(["date", "code"]).first().reset_index()
            logger.info(f"Sentiment data fetched: {len(result)} records")
            return result

        return pd.DataFrame()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算舆情特征(主入口)

        Args:
            df: 股票数据DataFrame

        Returns:
            带舆情特征的DataFrame
        """
        logger.info("Computing sentiment features...")

        # 获取股票列表和日期范围
        stock_codes = df["code"].unique().tolist()
        start_date = df["date"].min()
        end_date = df["date"].max()

        # 尝试从缓存加载
        cache_path = self.data_dir / "sentiment_cache.parquet"
        if cache_path.exists():
            logger.info("Loading sentiment data from cache...")
            sentiment_df = pd.read_parquet(cache_path)
        else:
            # 获取舆情数据
            sentiment_df = self.fetch_all_sentiment_data(
                stock_codes[:100],  # 限制数量,避免请求过多
                str(start_date)[:10],
                str(end_date)[:10]
            )
            if len(sentiment_df) > 0:
                sentiment_df.to_parquet(cache_path, index=False)

        if len(sentiment_df) == 0:
            logger.warning("No sentiment data available, using default values")
            # 添加默认舆情特征
            df["news_count"] = 0
            df["sentiment_mean"] = 0
            df["sentiment_sum"] = 0
            df["ann_count"] = 0
            df["ann_weight_sum"] = 0
            return df

        # 合并舆情数据
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        df = df.merge(sentiment_df, on=["date", "code"], how="left")

        # 填充缺失值
        sentiment_cols = [
            "news_count", "sentiment_mean", "sentiment_sum", "sentiment_std",
            "positive_total", "negative_total", "ann_count",
            "ann_weight_sum", "ann_weight_mean"
        ]
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 计算滚动舆情特征
        df = df.sort_values(["code", "date"])
        for window in [5, 10, 20]:
            df[f"sentiment_ma_{window}"] = df.groupby("code")["sentiment_mean"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"news_count_ma_{window}"] = df.groupby("code")["news_count"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # 舆情动量
        df["sentiment_momentum"] = df.groupby("code")["sentiment_mean"].diff(5)

        # 舆情异常(相对历史)
        df["sentiment_zscore"] = df.groupby("code")["sentiment_mean"].transform(
            lambda x: (x - x.rolling(20, min_periods=5).mean()) /
                      (x.rolling(20, min_periods=5).std() + 1e-10)
        )

        logger.info(f"Sentiment features computed. Shape: {df.shape}")

        return df

    def get_feature_names(self) -> List[str]:
        """
        获取舆情特征名列表

        Returns:
            特征名列表
        """
        return [
            "news_count", "sentiment_mean", "sentiment_sum", "sentiment_std",
            "positive_total", "negative_total",
            "ann_count", "ann_weight_sum", "ann_weight_mean",
            "sentiment_ma_5", "sentiment_ma_10", "sentiment_ma_20",
            "news_count_ma_5", "news_count_ma_10", "news_count_ma_20",
            "sentiment_momentum", "sentiment_zscore"
        ]
