"""
数据下载模块
支持 akshare 和 baostock 数据源
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

# baostock 可选导入
try:
    import baostock as bs
    HAS_BAOSTOCK = True
except ImportError:
    HAS_BAOSTOCK = False


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
        带重试的请求（指数退避）

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
                wait_time = self.retry_delay * (2 ** attempt)  # 指数退避: 5, 10, 20秒
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_times}), waiting {wait_time}s: {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(wait_time)
        return None

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全部A股股票列表

        Returns:
            股票列表DataFrame
        """
        logger.info("Fetching A-share stock list...")

        # 优先使用 baostock
        if HAS_BAOSTOCK:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    rs = bs.query_stock_basic()
                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        data_list.append(rs.get_row_data())
                    bs.logout()

                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        # 过滤A股（排除指数等）
                        df = df[df["type"] == "1"]  # 1=股票
                        df = df[df["status"] == "1"]  # 1=上市
                        # 提取6位代码
                        df["code"] = df["code"].str.split(".").str[1]
                        df = df.rename(columns={"code_name": "name"})
                        stock_info = df[["code", "name"]].copy()

                        save_path = self.raw_data_dir / "stock_list.csv"
                        stock_info.to_csv(save_path, index=False, encoding="utf-8-sig")
                        logger.info(f"Stock list saved to {save_path}, total: {len(stock_info)} stocks (baostock)")
                        return stock_info
            except Exception as e:
                logger.warning(f"baostock failed, fallback to akshare: {e}")

        # 回退到 akshare
        stock_info = self._retry_request(ak.stock_info_a_code_name)
        if stock_info is None:
            logger.error("Failed to fetch stock list")
            return pd.DataFrame()

        stock_info.columns = ["code", "name"]
        save_path = self.raw_data_dir / "stock_list.csv"
        stock_info.to_csv(save_path, index=False, encoding="utf-8-sig")
        logger.info(f"Stock list saved to {save_path}, total: {len(stock_info)} stocks")

        return stock_info

    def get_stock_industry(self) -> pd.DataFrame:
        """
        获取股票行业分类

        Returns:
            行业分类DataFrame
        """
        logger.info("Fetching stock industry classification...")

        # 优先使用 baostock
        if HAS_BAOSTOCK:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    rs = bs.query_stock_industry()
                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        data_list.append(rs.get_row_data())
                    bs.logout()

                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        # 提取6位代码
                        df["code"] = df["code"].str.split(".").str[1]
                        df = df.rename(columns={
                            "code_name": "name",
                            "industry": "industry"
                        })
                        result = df[["code", "name", "industry"]].copy()

                        save_path = self.raw_data_dir / "stock_industry.csv"
                        result.to_csv(save_path, index=False, encoding="utf-8-sig")
                        logger.info(f"Industry data saved to {save_path} (baostock)")
                        return result
            except Exception as e:
                logger.warning(f"baostock failed, fallback to akshare: {e}")

        # 回退到 akshare
        try:
            industry_df = self._retry_request(ak.stock_board_industry_name_em)
            if industry_df is None:
                return pd.DataFrame()

            all_industry_stocks = []

            for _, row in tqdm(industry_df.iterrows(), total=len(industry_df), desc="Fetching industry stocks"):
                industry_name = row["板块名称"]
                stocks = self._retry_request(ak.stock_board_industry_cons_em, symbol=industry_name)
                if stocks is not None and len(stocks) > 0:
                    stocks["industry"] = industry_name
                    all_industry_stocks.append(stocks[["代码", "名称", "industry"]])
                time.sleep(0.5)

            if all_industry_stocks:
                result = pd.concat(all_industry_stocks, ignore_index=True)
                result.columns = ["code", "name", "industry"]

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

            # 获取历史数据（带重试）
            df = self._retry_request(
                ak.stock_zh_a_hist,
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

    def download_all_stocks_baostock(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        使用 baostock 批量下载股票数据（推荐，无限流）

        Args:
            stock_codes: 股票代码列表,None则下载全部

        Returns:
            合并后的DataFrame
        """
        if not HAS_BAOSTOCK:
            logger.error("baostock not installed. Run: pip install baostock")
            return pd.DataFrame()

        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list["code"].tolist()

        # 检查是否有已下载的数据（断点续传）
        save_path = self.raw_data_dir / "stock_daily.parquet"
        partial_path = self.raw_data_dir / "stock_daily_partial.parquet"

        existing_data = None
        downloaded_codes = set()

        # 优先读取部分下载的数据
        if partial_path.exists():
            existing_data = pd.read_parquet(partial_path)
            downloaded_codes = set(existing_data["code"].unique())
            logger.info(f"Found partial download: {len(downloaded_codes)} stocks already downloaded")
        elif save_path.exists():
            existing_data = pd.read_parquet(save_path)
            downloaded_codes = set(existing_data["code"].unique())
            logger.info(f"Found existing data: {len(downloaded_codes)} stocks already downloaded")

        # 过滤掉已下载的股票
        remaining_codes = [c for c in stock_codes if c not in downloaded_codes]

        if not remaining_codes:
            logger.info("All stocks already downloaded!")
            return existing_data if existing_data is not None else pd.DataFrame()

        logger.info(f"Starting to download {len(remaining_codes)}/{len(stock_codes)} stocks via baostock...")

        # 登录 baostock
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"baostock login failed: {lg.error_msg}")
            return existing_data if existing_data is not None else pd.DataFrame()

        all_data = [existing_data] if existing_data is not None else []
        failed_stocks = []
        new_downloaded = 0

        for i, code in enumerate(tqdm(remaining_codes, desc="Downloading (baostock)")):
            # baostock 代码格式: sh.600000 或 sz.000001
            if code.startswith(("6", "9")):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"

            success = False
            for attempt in range(3):  # 重试3次
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,code,open,high,low,close,volume,amount,turn,pctChg",
                        start_date=self.start_date,
                        end_date=self.end_date,
                        frequency="d",
                        adjustflag="2"  # 前复权
                    )

                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        data_list.append(rs.get_row_data())

                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        df = df.rename(columns={
                            "date": "date",
                            "code": "code",
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume",
                            "amount": "amount",
                            "turn": "turnover",
                            "pctChg": "pct_change"
                        })
                        for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        df["date"] = pd.to_datetime(df["date"])
                        df["code"] = code
                        all_data.append(df)
                        new_downloaded += 1
                        success = True
                        break
                    elif attempt < 2:
                        time.sleep(2)
                        continue

                except Exception as e:
                    if attempt < 2:
                        logger.debug(f"Retry {code} (attempt {attempt + 1}): {e}")
                        # 重新登录
                        try:
                            bs.logout()
                            time.sleep(3)
                            bs.login()
                        except:
                            pass
                        continue
                    logger.debug(f"Failed to download {code}: {e}")

            if not success:
                failed_stocks.append(code)

            # 每个请求间隔 0.1 秒
            time.sleep(0.1)

            # 每 500 个保存一次进度，防止中断丢失
            if (i + 1) % 500 == 0 and all_data:
                temp_result = pd.concat(all_data, ignore_index=True)
                temp_path = self.raw_data_dir / "stock_daily_partial.parquet"
                temp_result.to_parquet(temp_path, index=False)
                total_done = len(downloaded_codes) + new_downloaded
                logger.info(f"Progress saved: {total_done}/{len(stock_codes)} stocks ({new_downloaded} new)")

        # 登出
        bs.logout()

        if all_data:
            result = pd.concat(all_data, ignore_index=True)

            # 保存最终结果
            result.to_parquet(save_path, index=False)
            logger.info(f"Daily data saved to {save_path}, total: {len(result['code'].unique())} stocks, {len(result)} records")

            # 删除临时文件
            if partial_path.exists():
                partial_path.unlink()
                logger.info("Partial download file removed")

            if failed_stocks:
                failed_path = self.raw_data_dir / "failed_stocks.txt"
                with open(failed_path, "w") as f:
                    f.write("\n".join(failed_stocks))
                logger.warning(f"Failed stocks: {len(failed_stocks)}, saved to {failed_path}")

            return result

        return existing_data if existing_data is not None else pd.DataFrame()

    def download_all_stocks(self, stock_codes: Optional[List[str]] = None, use_baostock: bool = True) -> pd.DataFrame:
        """
        批量下载所有股票数据

        Args:
            stock_codes: 股票代码列表,None则下载全部
            use_baostock: 是否使用baostock（推荐，无限流）

        Returns:
            合并后的DataFrame
        """
        # 优先使用 baostock
        if use_baostock and HAS_BAOSTOCK:
            return self.download_all_stocks_baostock(stock_codes)

        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list["code"].tolist()

        logger.info(f"Starting to download {len(stock_codes)} stocks via akshare...")

        all_data = []
        failed_stocks = []

        # 串行下载，避免限流
        for i, code in enumerate(tqdm(stock_codes, desc="Downloading")):
            try:
                df = self.download_stock_daily(code)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                else:
                    failed_stocks.append(code)

                # 请求间隔 1 秒
                time.sleep(1.0)

                # 每 50 个请求暂停 30 秒，让服务器冷却
                if (i + 1) % 50 == 0:
                    logger.info(f"Downloaded {i + 1}/{len(stock_codes)}, pausing 30s to avoid rate limit...")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error downloading {code}: {e}")
                failed_stocks.append(code)
                time.sleep(5)  # 出错后额外等待

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

        # 优先使用 baostock
        if HAS_BAOSTOCK:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    for idx_info in tqdm(indices, desc="Downloading indices (baostock)"):
                        code = idx_info["code"]
                        name = idx_info["name"]

                        # baostock 指数代码格式
                        if code.startswith("0") or code.startswith("5"):
                            bs_code = f"sh.{code}"
                        else:
                            bs_code = f"sz.{code}"

                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume",
                            start_date=self.start_date,
                            end_date=self.end_date,
                            frequency="d"
                        )

                        data_list = []
                        while (rs.error_code == '0') and rs.next():
                            data_list.append(rs.get_row_data())

                        if data_list:
                            df = pd.DataFrame(data_list, columns=rs.fields)
                            for col in ["open", "high", "low", "close", "volume"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            df["date"] = pd.to_datetime(df["date"])
                            df["code"] = code
                            df["name"] = name
                            all_data.append(df)

                    bs.logout()

                    if all_data:
                        result = pd.concat(all_data, ignore_index=True)
                        save_path = self.raw_data_dir / "index_daily.parquet"
                        result.to_parquet(save_path, index=False)
                        logger.info(f"Index data saved to {save_path} (baostock)")
                        return result
            except Exception as e:
                logger.warning(f"baostock failed, fallback to akshare: {e}")

        # 回退到 akshare
        for idx_info in tqdm(indices, desc="Downloading indices"):
            code = idx_info["code"]
            name = idx_info["name"]

            try:
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

        # 优先使用 baostock
        if HAS_BAOSTOCK:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    rs = bs.query_trade_dates(
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        row = rs.get_row_data()
                        if row[1] == '1':  # is_trading_day = 1
                            data_list.append(row[0])  # calendar_date
                    bs.logout()

                    if data_list:
                        df = pd.DataFrame({"date": data_list})
                        df["date"] = pd.to_datetime(df["date"])

                        save_path = self.raw_data_dir / "trade_calendar.csv"
                        df.to_csv(save_path, index=False)
                        logger.info(f"Trade calendar saved to {save_path} (baostock)")
                        return df
            except Exception as e:
                logger.warning(f"baostock failed, fallback to akshare: {e}")

        # 回退到 akshare
        try:
            df = ak.tool_trade_date_hist_sina()
            if df is not None:
                df.columns = ["date"]
                df["date"] = pd.to_datetime(df["date"])

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
