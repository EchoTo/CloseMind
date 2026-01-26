"""
数据下载模块
使用akshare获取A股数据
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak
from tqdm import tqdm
from loguru import logger


class DataDownloader:
    """A股数据下载器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化下载器

        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.paths = config.get("paths", {})

        # 创建数据目录
        self.raw_data_dir = Path(self.paths.get("raw_data_dir", "./data_storage/raw"))
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # 下载配置
        download_config = self.data_config.get("download", {})
        self.retry_times = download_config.get("retry_times", 3)
        self.retry_delay = download_config.get("retry_delay", 5)
        self.batch_size = download_config.get("batch_size", 100)
        self.parallel_workers = download_config.get("parallel_workers", 4)

        # 时间范围
        self.start_date = self.data_config.get("start_date", "2022-01-01")
        self.end_date = self.data_config.get("end_date") or datetime.now().strftime("%Y-%m-%d")

        logger.info(f"DataDownloader initialized. Data range: {self.start_date} to {self.end_date}")

    def _retry_request(self, func, *args, **kwargs) -> Optional[Any]:
        """
        带重试的请求

        Args:
            func: 请求函数
            *args, **kwargs: 函数参数

        Returns:
            请求结果或None
        """
        for attempt in range(self.retry_times):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_times}): {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay)
        return None

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全部A股股票列表

        Returns:
            股票列表DataFrame
        """
        logger.info("Fetching A-share stock list...")

        # 获取A股股票列表
        stock_info = self._retry_request(ak.stock_info_a_code_name)
        if stock_info is None:
            logger.error("Failed to fetch stock list")
            return pd.DataFrame()

        # 重命名列
        stock_info.columns = ["code", "name"]

        # 保存
        save_path = self.raw_data_dir / "stock_list.csv"
        stock_info.to_csv(save_path, index=False, encoding="utf-8-sig")
        logger.info(f"Stock list saved to {save_path}, total: {len(stock_info)} stocks")

        return stock_info

    def get_stock_industry(self) -> pd.DataFrame:
        """
        获取股票行业分类(申万行业)

        Returns:
            行业分类DataFrame
        """
        logger.info("Fetching stock industry classification...")

        try:
            # 获取申万行业分类
            industry_df = self._retry_request(ak.stock_board_industry_name_em)
            if industry_df is None:
                return pd.DataFrame()

            all_industry_stocks = []

            for _, row in tqdm(industry_df.iterrows(), total=len(industry_df), desc="Fetching industry stocks"):
                industry_name = row["板块名称"]
                try:
                    stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
                    if stocks is not None and len(stocks) > 0:
                        stocks["industry"] = industry_name
                        all_industry_stocks.append(stocks[["代码", "名称", "industry"]])
                    time.sleep(0.1)  # 避免请求过快
                except Exception as e:
                    logger.warning(f"Failed to fetch industry {industry_name}: {e}")
                    continue

            if all_industry_stocks:
                result = pd.concat(all_industry_stocks, ignore_index=True)
                result.columns = ["code", "name", "industry"]

                # 保存
                save_path = self.raw_data_dir / "stock_industry.csv"
                result.to_csv(save_path, index=False, encoding="utf-8-sig")
                logger.info(f"Industry data saved to {save_path}")

                return result

        except Exception as e:
            logger.error(f"Failed to fetch industry data: {e}")

        return pd.DataFrame()

    def download_stock_daily(self, stock_code: str, adjust: str = "qfq") -> Optional[pd.DataFrame]:
        """
        下载单只股票的日K线数据

        Args:
            stock_code: 股票代码(6位数字)
            adjust: 复权类型 qfq-前复权, hfq-后复权, 空-不复权

        Returns:
            日K线DataFrame
        """
        try:
            # akshare需要带市场前缀
            if stock_code.startswith(("6", "9")):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"

            # 获取历史数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                adjust=adjust
            )

            if df is None or len(df) == 0:
                return None

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover"
            })

            # 添加股票代码
            df["code"] = stock_code

            # 确保日期格式
            df["date"] = pd.to_datetime(df["date"])

            return df

        except Exception as e:
            logger.debug(f"Failed to download {stock_code}: {e}")
            return None

    def download_all_stocks(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量下载所有股票数据

        Args:
            stock_codes: 股票代码列表,None则下载全部

        Returns:
            合并后的DataFrame
        """
        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list["code"].tolist()

        logger.info(f"Starting to download {len(stock_codes)} stocks...")

        all_data = []
        failed_stocks = []

        # 使用多线程下载
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self.download_stock_daily, code): code
                for code in stock_codes
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                    else:
                        failed_stocks.append(code)
                except Exception as e:
                    logger.error(f"Error downloading {code}: {e}")
                    failed_stocks.append(code)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)

            # 保存
            save_path = self.raw_data_dir / "stock_daily.parquet"
            result.to_parquet(save_path, index=False)
            logger.info(f"Daily data saved to {save_path}, total records: {len(result)}")

            # 保存失败列表
            if failed_stocks:
                failed_path = self.raw_data_dir / "failed_stocks.txt"
                with open(failed_path, "w") as f:
                    f.write("\n".join(failed_stocks))
                logger.warning(f"Failed stocks: {len(failed_stocks)}, saved to {failed_path}")

            return result

        return pd.DataFrame()

    def download_index_daily(self) -> pd.DataFrame:
        """
        下载指数日K线数据

        Returns:
            指数数据DataFrame
        """
        indices = self.data_config.get("indices", [])
        logger.info(f"Downloading {len(indices)} index data...")

        all_data = []

        for idx_info in tqdm(indices, desc="Downloading indices"):
            code = idx_info["code"]
            name = idx_info["name"]

            try:
                # 上证指数
                if code.startswith("0"):
                    df = ak.stock_zh_index_daily(symbol=f"sh{code}")
                else:
                    df = ak.stock_zh_index_daily(symbol=f"sz{code}")

                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        "date": "date",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume"
                    })
                    df["code"] = code
                    df["name"] = name
                    df["date"] = pd.to_datetime(df["date"])

                    # 过滤日期范围
                    df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
                    all_data.append(df)

            except Exception as e:
                logger.error(f"Failed to download index {code}: {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            save_path = self.raw_data_dir / "index_daily.parquet"
            result.to_parquet(save_path, index=False)
            logger.info(f"Index data saved to {save_path}")
            return result

        return pd.DataFrame()

    def download_trade_calendar(self) -> pd.DataFrame:
        """
        下载交易日历

        Returns:
            交易日历DataFrame
        """
        logger.info("Downloading trade calendar...")

        try:
            # 使用上交所交易日历
            df = ak.tool_trade_date_hist_sina()
            if df is not None:
                df.columns = ["date"]
                df["date"] = pd.to_datetime(df["date"])

                # 过滤日期范围
                df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]

                save_path = self.raw_data_dir / "trade_calendar.csv"
                df.to_csv(save_path, index=False)
                logger.info(f"Trade calendar saved to {save_path}")

                return df

        except Exception as e:
            logger.error(f"Failed to download trade calendar: {e}")

        return pd.DataFrame()

    def download_all(self) -> Dict[str, pd.DataFrame]:
        """
        下载所有数据

        Returns:
            包含所有数据的字典
        """
        logger.info("Starting full data download...")

        result = {}

        # 下载交易日历
        result["calendar"] = self.download_trade_calendar()

        # 下载股票列表
        result["stock_list"] = self.get_stock_list()

        # 下载行业分类
        result["industry"] = self.get_stock_industry()

        # 下载股票日K线
        result["stock_daily"] = self.download_all_stocks()

        # 下载指数数据
        result["index_daily"] = self.download_index_daily()

        logger.info("Full data download completed!")
        return result

    def update_data(self) -> Dict[str, pd.DataFrame]:
        """
        增量更新数据

        Returns:
            更新后的数据字典
        """
        logger.info("Starting incremental data update...")

        # 读取现有数据
        stock_daily_path = self.raw_data_dir / "stock_daily.parquet"
        if stock_daily_path.exists():
            existing_data = pd.read_parquet(stock_daily_path)
            last_date = existing_data["date"].max()

            # 设置新的开始日期
            self.start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Updating data from {self.start_date}")

            # 下载新数据
            new_data = self.download_all_stocks(
                existing_data["code"].unique().tolist()
            )

            if len(new_data) > 0:
                # 合并数据
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                updated_data = updated_data.drop_duplicates(subset=["code", "date"], keep="last")
                updated_data.to_parquet(stock_daily_path, index=False)
                logger.info(f"Updated data saved, new records: {len(new_data)}")

                return {"stock_daily": updated_data}

        logger.info("No existing data found, performing full download")
        return self.download_all()


if __name__ == "__main__":
    import yaml

    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化下载器
    downloader = DataDownloader(config)

    # 下载所有数据
    data = downloader.download_all()
