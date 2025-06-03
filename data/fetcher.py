import akshare as ak
import pandas as pd
import datetime

def fetch_stock_data(stock_code, start_date, end_date=None):
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date="19900101",  # 给个足够早的时间
        end_date=end_date or pd.Timestamp.today().strftime('%Y%m%d'),
        adjust="qfq"
    )
    df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # 手动筛选
    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    return df