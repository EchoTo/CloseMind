"""
Qlib格式转换模块
将处理后的数据转换为Qlib可读取的bin格式
"""

import os
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger


class QlibConverter:
    """Qlib数据格式转换器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化转换器

        Args:
            config: 配置字典
        """
        self.config = config
        self.paths = config.get("paths", {})

        # 路径设置
        self.processed_data_dir = Path(self.paths.get("processed_data_dir", "./data_storage/processed"))
        self.qlib_data_dir = Path(self.paths.get("qlib_data_dir", "./data_storage/qlib"))

        # 创建输出目录
        self.qlib_data_dir.mkdir(parents=True, exist_ok=True)

        # Qlib数据列映射
        self.column_mapping = {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "amount": "$amount",
            "turnover": "$turnover",
            "pct_change": "$change",
        }

        logger.info("QlibConverter initialized")

    def _float_to_bin(self, value: float) -> bytes:
        """
        将浮点数转换为4字节二进制

        Args:
            value: 浮点数

        Returns:
            二进制数据
        """
        if np.isnan(value) or np.isinf(value):
            return struct.pack("f", np.nan)
        return struct.pack("f", float(value))

    def _create_calendar(self, df: pd.DataFrame) -> None:
        """
        创建交易日历文件

        Args:
            df: 数据DataFrame
        """
        calendar_dir = self.qlib_data_dir / "calendars"
        calendar_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有交易日
        dates = sorted(df["date"].unique())
        dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]

        # 保存day日历
        calendar_path = calendar_dir / "day.txt"
        with open(calendar_path, "w") as f:
            f.write("\n".join(dates_str))

        logger.info(f"Calendar created: {len(dates_str)} trading days")

    def _create_instruments(self, df: pd.DataFrame) -> None:
        """
        创建股票列表文件

        Args:
            df: 数据DataFrame
        """
        instruments_dir = self.qlib_data_dir / "instruments"
        instruments_dir.mkdir(parents=True, exist_ok=True)

        # 获取每只股票的起止日期
        stock_dates = df.groupby("code")["date"].agg(["min", "max"]).reset_index()
        stock_dates.columns = ["code", "start_date", "end_date"]

        # 转换为Qlib格式: code start_date end_date
        lines = []
        for _, row in stock_dates.iterrows():
            code = row["code"]
            # 转换为Qlib股票代码格式 (sh600000 或 sz000001)
            if code.startswith(("6", "9")):
                qlib_code = f"sh{code}"
            else:
                qlib_code = f"sz{code}"

            start = pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d")
            end = pd.Timestamp(row["end_date"]).strftime("%Y-%m-%d")
            lines.append(f"{qlib_code}\t{start}\t{end}")

        # 保存all股票列表
        all_path = instruments_dir / "all.txt"
        with open(all_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Instruments created: {len(lines)} stocks")

    def _create_features(self, df: pd.DataFrame) -> None:
        """
        创建特征文件(bin格式)

        Args:
            df: 数据DataFrame
        """
        features_dir = self.qlib_data_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有交易日作为索引
        all_dates = sorted(df["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        num_days = len(all_dates)

        # 按股票处理
        stocks = df["code"].unique()
        logger.info(f"Converting features for {len(stocks)} stocks...")

        for code in tqdm(stocks, desc="Converting features"):
            # 转换股票代码
            if code.startswith(("6", "9")):
                qlib_code = f"sh{code}"
            else:
                qlib_code = f"sz{code}"

            # 创建股票目录
            stock_dir = features_dir / qlib_code
            stock_dir.mkdir(parents=True, exist_ok=True)

            # 获取该股票数据
            stock_data = df[df["code"] == code].copy()
            stock_data = stock_data.set_index("date")

            # 为每个特征创建bin文件
            for col, qlib_name in self.column_mapping.items():
                if col not in stock_data.columns:
                    continue

                # 创建完整时间序列(用nan填充)
                values = np.full(num_days, np.nan, dtype=np.float32)

                for date, row in stock_data.iterrows():
                    if date in date_to_idx:
                        idx = date_to_idx[date]
                        values[idx] = row[col]

                # 保存为bin文件
                feature_name = qlib_name.replace("$", "")
                bin_path = stock_dir / f"{feature_name}.day.bin"
                values.tofile(bin_path)

    def _create_feature_list(self) -> None:
        """创建特征列表文件"""
        features_list_path = self.qlib_data_dir / "features" / "feature_list.txt"
        feature_names = [v.replace("$", "") for v in self.column_mapping.values()]

        with open(features_list_path, "w") as f:
            f.write("\n".join(feature_names))

        logger.info(f"Feature list created: {len(feature_names)} features")

    def convert(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        执行转换

        Args:
            df: 数据DataFrame,None则从processed目录加载
        """
        logger.info("Starting Qlib format conversion...")

        # 加载数据
        if df is None:
            processed_path = self.processed_data_dir / "stock_processed.parquet"
            if not processed_path.exists():
                logger.error("No processed data found!")
                return
            df = pd.read_parquet(processed_path)

        # 确保日期格式
        df["date"] = pd.to_datetime(df["date"])

        # 创建日历
        self._create_calendar(df)

        # 创建股票列表
        self._create_instruments(df)

        # 创建特征文件
        self._create_features(df)

        # 创建特征列表
        self._create_feature_list()

        logger.info(f"Qlib conversion completed! Output: {self.qlib_data_dir}")

    def init_qlib(self) -> None:
        """
        初始化Qlib环境
        """
        try:
            import qlib
            from qlib.config import REG_CN

            qlib.init(
                provider_uri=str(self.qlib_data_dir),
                region=REG_CN,
            )
            logger.info("Qlib initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise

    def load_qlib_data(
        self,
        instruments: str = "all",
        start_time: str = "2022-01-01",
        end_time: str = None,
        fields: List[str] = None
    ) -> pd.DataFrame:
        """
        使用Qlib加载数据

        Args:
            instruments: 股票池
            start_time: 开始时间
            end_time: 结束时间
            fields: 特征列表

        Returns:
            数据DataFrame
        """
        try:
            from qlib.data import D

            if fields is None:
                fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]

            # 加载数据
            df = D.features(
                instruments=instruments,
                fields=fields,
                start_time=start_time,
                end_time=end_time,
            )

            logger.info(f"Loaded Qlib data: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to load Qlib data: {e}")
            return pd.DataFrame()


class SimpleDataLoader:
    """
    简化版数据加载器(不依赖Qlib bin格式)
    直接从parquet文件加载,适用于快速开发和测试
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        self.paths = config.get("paths", {})
        self.processed_data_dir = Path(self.paths.get("processed_data_dir", "./data_storage/processed"))

        self._data_cache = None

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        codes: List[str] = None
    ) -> pd.DataFrame:
        """
        加载数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            codes: 股票代码列表

        Returns:
            数据DataFrame
        """
        # 使用缓存
        if self._data_cache is None:
            data_path = self.processed_data_dir / "stock_processed.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self._data_cache = pd.read_parquet(data_path)
            self._data_cache["date"] = pd.to_datetime(self._data_cache["date"])

        df = self._data_cache.copy()

        # 日期过滤
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        # 股票过滤
        if codes:
            df = df[df["code"].isin(codes)]

        return df

    def get_trade_dates(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> List[str]:
        """
        获取交易日列表

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        df = self.load_data(start_date, end_date)
        dates = sorted(df["date"].unique())
        return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]

    def get_stock_codes(self, date: str = None) -> List[str]:
        """
        获取股票列表

        Args:
            date: 指定日期,None则返回所有股票

        Returns:
            股票代码列表
        """
        df = self.load_data()
        if date:
            df = df[df["date"] == date]
        return sorted(df["code"].unique().tolist())


if __name__ == "__main__":
    import yaml

    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化转换器
    converter = QlibConverter(config)

    # 执行转换
    converter.convert()
