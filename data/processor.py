"""
数据处理模块
负责数据清洗、异常值处理和标准化
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataProcessor:
    """A股数据处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.paths = config.get("paths", {})
        self.data_config = config.get("data", {})
        self.stock_filter = self.data_config.get("stock_filter", {})

        # 路径设置
        self.raw_data_dir = Path(self.paths.get("raw_data_dir", "./data_storage/raw"))
        self.processed_data_dir = Path(self.paths.get("processed_data_dir", "./data_storage/processed"))
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DataProcessor initialized")

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载原始数据

        Returns:
            原始数据字典
        """
        data = {}

        # 加载股票日K线
        stock_daily_path = self.raw_data_dir / "stock_daily.parquet"
        if stock_daily_path.exists():
            data["stock_daily"] = pd.read_parquet(stock_daily_path)
            logger.info(f"Loaded stock daily data: {len(data['stock_daily'])} records")

        # 加载股票列表
        stock_list_path = self.raw_data_dir / "stock_list.csv"
        if stock_list_path.exists():
            data["stock_list"] = pd.read_csv(stock_list_path)
            logger.info(f"Loaded stock list: {len(data['stock_list'])} stocks")

        # 加载行业分类
        industry_path = self.raw_data_dir / "stock_industry.csv"
        if industry_path.exists():
            data["industry"] = pd.read_csv(industry_path)
            logger.info(f"Loaded industry data: {len(data['industry'])} records")

        # 加载指数数据
        index_path = self.raw_data_dir / "index_daily.parquet"
        if index_path.exists():
            data["index_daily"] = pd.read_parquet(index_path)
            logger.info(f"Loaded index data: {len(data['index_daily'])} records")

        # 加载交易日历
        calendar_path = self.raw_data_dir / "trade_calendar.csv"
        if calendar_path.exists():
            data["calendar"] = pd.read_csv(calendar_path, parse_dates=["date"])
            logger.info(f"Loaded trade calendar: {len(data['calendar'])} days")

        return data

    def filter_stocks(self, df: pd.DataFrame, stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        股票筛选

        Args:
            df: 股票数据
            stock_list: 股票列表(含名称)

        Returns:
            筛选后的数据
        """
        original_count = df["code"].nunique()
        logger.info(f"Filtering stocks, original count: {original_count}")

        # 合并股票名称
        if "name" not in df.columns and stock_list is not None:
            stock_name_map = stock_list.set_index("code")["name"].to_dict()
            df["name"] = df["code"].map(stock_name_map)

        # 排除ST股
        if self.stock_filter.get("exclude_st", True) and "name" in df.columns:
            st_mask = df["name"].str.contains("ST|退", na=False, case=False)
            df = df[~st_mask]
            logger.info(f"After excluding ST stocks: {df['code'].nunique()}")

        # 排除新股(上市不满N天)
        exclude_days = self.stock_filter.get("exclude_new_stock_days", 60)
        if exclude_days > 0:
            # 计算每只股票的首个交易日
            first_trade_date = df.groupby("code")["date"].min().reset_index()
            first_trade_date.columns = ["code", "first_date"]

            # 计算每条记录距离上市的天数
            df = df.merge(first_trade_date, on="code", how="left")
            df["days_since_ipo"] = (df["date"] - df["first_date"]).dt.days

            # 保留上市满N天的记录
            df = df[df["days_since_ipo"] >= exclude_days]
            df = df.drop(columns=["first_date", "days_since_ipo"])
            logger.info(f"After excluding new stocks: {df['code'].nunique()}")

        logger.info(f"Final stock count: {df['code'].nunique()}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            df: 数据DataFrame

        Returns:
            处理后的数据
        """
        logger.info("Handling missing values...")

        # 检查缺失情况
        missing_stats = df.isnull().sum()
        if missing_stats.any():
            logger.info(f"Missing values:\n{missing_stats[missing_stats > 0]}")

        # 价格相关列:前向填充
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df.groupby("code")[col].ffill()

        # 成交量/成交额:填充0(停牌)
        volume_cols = ["volume", "amount"]
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 其他列:前向填充
        other_cols = ["turnover", "amplitude", "pct_change", "change"]
        for col in other_cols:
            if col in df.columns:
                df[col] = df.groupby("code")[col].ffill()
                df[col] = df[col].fillna(0)

        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值

        Args:
            df: 数据DataFrame

        Returns:
            处理后的数据
        """
        logger.info("Handling outliers...")

        # 涨跌幅异常(超过20%可能是除权除息)
        if "pct_change" in df.columns:
            # 标记异常涨跌幅
            df["is_abnormal"] = (df["pct_change"].abs() > 20) | (df["pct_change"].isna())

        # 价格为0或负数的记录
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan
                df[col] = df.groupby("code")[col].ffill()

        # 成交量异常(负数)
        if "volume" in df.columns:
            df.loc[df["volume"] < 0, "volume"] = 0

        return df

    def mark_trading_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记交易状态

        Args:
            df: 数据DataFrame

        Returns:
            带交易状态的数据
        """
        logger.info("Marking trading status...")

        # 停牌:成交量为0
        df["is_suspended"] = df["volume"] == 0

        # 涨停:涨幅接近10%或20%(创业板/科创板)
        if "pct_change" in df.columns:
            # 主板涨停
            df["is_limit_up"] = (df["pct_change"] >= 9.9) & (df["pct_change"] <= 10.1)
            # 创业板/科创板涨停(20%)
            cb_codes = df["code"].str.startswith(("300", "301", "688", "689"))
            df.loc[cb_codes & (df["pct_change"] >= 19.9) & (df["pct_change"] <= 20.1), "is_limit_up"] = True

        # 跌停
        if "pct_change" in df.columns:
            df["is_limit_down"] = (df["pct_change"] <= -9.9) & (df["pct_change"] >= -10.1)
            cb_codes = df["code"].str.startswith(("300", "301", "688", "689"))
            df.loc[cb_codes & (df["pct_change"] <= -19.9) & (df["pct_change"] >= -20.1), "is_limit_down"] = True

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征

        Args:
            df: 数据DataFrame

        Returns:
            带时间特征的数据
        """
        logger.info("Adding time features...")

        df["date"] = pd.to_datetime(df["date"])

        # 年/月/日/星期
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.weekday  # 0=周一

        # 是否月初/月末
        df["is_month_start"] = df["date"].dt.is_month_start
        df["is_month_end"] = df["date"].dt.is_month_end

        # 季度
        df["quarter"] = df["date"].dt.quarter

        return df

    def merge_industry(self, df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并行业数据

        Args:
            df: 股票数据
            industry_df: 行业分类数据

        Returns:
            合并后的数据
        """
        if industry_df is None or len(industry_df) == 0:
            logger.warning("No industry data available")
            return df

        logger.info("Merging industry data...")

        # 去重(一只股票可能属于多个行业,取第一个)
        industry_map = industry_df.drop_duplicates(subset=["code"]).set_index("code")["industry"]
        df["industry"] = df["code"].map(industry_map)

        # 缺失行业填充"未知"
        df["industry"] = df["industry"].fillna("未知")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建预测标签

        Args:
            df: 数据DataFrame

        Returns:
            带标签的数据
        """
        logger.info("Creating labels...")

        labels_config = self.config.get("labels", {})
        daily_config = labels_config.get("daily", {})
        weekly_config = labels_config.get("weekly", {})

        # 按股票和日期排序
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 日度标签:下一日收益率
        df["return_1d"] = df.groupby("code")["close"].pct_change(1).shift(-1)

        # 周度标签:未来5日累计收益率
        forward_days = weekly_config.get("forward_days", 5)
        df["return_5d"] = df.groupby("code")["close"].pct_change(forward_days).shift(-forward_days)

        # 计算收益率排名(每日截面)
        df["return_1d_rank"] = df.groupby("date")["return_1d"].rank(pct=True)
        df["return_5d_rank"] = df.groupby("date")["return_5d"].rank(pct=True)

        # 二分类标签
        threshold = daily_config.get("threshold", 0.0)
        df["label_binary_1d"] = (df["return_1d"] > threshold).astype(int)
        df["label_binary_5d"] = (df["return_5d"] > threshold).astype(int)

        return df

    def process(self) -> pd.DataFrame:
        """
        执行完整的数据处理流程

        Returns:
            处理后的数据
        """
        logger.info("Starting data processing...")

        # 加载原始数据
        raw_data = self.load_raw_data()

        if "stock_daily" not in raw_data:
            logger.error("No stock daily data found!")
            return pd.DataFrame()

        df = raw_data["stock_daily"]
        stock_list = raw_data.get("stock_list")
        industry_df = raw_data.get("industry")

        # 确保日期格式正确
        df["date"] = pd.to_datetime(df["date"])

        # 股票筛选
        df = self.filter_stocks(df, stock_list)

        # 缺失值处理
        df = self.handle_missing_values(df)

        # 异常值处理
        df = self.handle_outliers(df)

        # 标记交易状态
        df = self.mark_trading_status(df)

        # 添加时间特征
        df = self.add_time_features(df)

        # 合并行业数据
        df = self.merge_industry(df, industry_df)

        # 创建标签
        df = self.create_labels(df)

        # 排序
        df = df.sort_values(["date", "code"]).reset_index(drop=True)

        # 保存处理后的数据
        save_path = self.processed_data_dir / "stock_processed.parquet"
        df.to_parquet(save_path, index=False)
        logger.info(f"Processed data saved to {save_path}")
        logger.info(f"Final shape: {df.shape}")

        # 保存指数数据(简单处理)
        if "index_daily" in raw_data:
            index_df = raw_data["index_daily"]
            index_df["date"] = pd.to_datetime(index_df["date"])
            index_save_path = self.processed_data_dir / "index_processed.parquet"
            index_df.to_parquet(index_save_path, index=False)
            logger.info(f"Index data saved to {index_save_path}")

        return df

    def get_processed_data(self) -> pd.DataFrame:
        """
        获取处理后的数据

        Returns:
            处理后的DataFrame
        """
        processed_path = self.processed_data_dir / "stock_processed.parquet"
        if processed_path.exists():
            return pd.read_parquet(processed_path)
        else:
            logger.warning("No processed data found, running processing...")
            return self.process()

    def get_train_valid_test_split(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间划分训练/验证/测试集

        Args:
            df: 数据DataFrame,None则自动加载

        Returns:
            (train_df, valid_df, test_df)
        """
        if df is None:
            df = self.get_processed_data()

        training_config = self.config.get("training", {})
        train_start = training_config.get("train_start", "2022-01-01")
        train_end = training_config.get("train_end", "2024-06-30")
        valid_start = training_config.get("valid_start", "2024-07-01")
        valid_end = training_config.get("valid_end", "2024-12-31")
        test_start = training_config.get("test_start", "2025-01-01")
        test_end = training_config.get("test_end")

        df["date"] = pd.to_datetime(df["date"])

        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)]

        if test_end:
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        else:
            test_df = df[df["date"] >= test_start]

        logger.info(f"Train: {len(train_df)} ({train_start} to {train_end})")
        logger.info(f"Valid: {len(valid_df)} ({valid_start} to {valid_end})")
        logger.info(f"Test: {len(test_df)} ({test_start} to {test_end or 'latest'})")

        return train_df, valid_df, test_df


if __name__ == "__main__":
    import yaml

    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化处理器
    processor = DataProcessor(config)

    # 执行处理
    processed_data = processor.process()
