import time
from datetime import datetime, time as dt_time

from data.fetcher import fetch_stock_data
from features.dataset_builder import build_dataset
from utils.scaler import normalize_close
from models.lstm_model import build_lstm_model
from predictor.predictor import train_and_predict
from config import *
from visualization.plotter import plot_kline_with_signals


def is_trading_time():
    """判断当前是否在交易时间段内"""
    now = datetime.now().time()
    return (
        dt_time(9, 30) <= now <= dt_time(11, 30)
        or dt_time(13, 0) <= now <= dt_time(15, 0)
    )


def is_market_closed():
    """判断是否已经收盘"""
    now = datetime.now().time()
    return now >= dt_time(15, 0)


def run_once():
    # 1. 获取数据
    df = fetch_stock_data(STOCK_CODE, START_DATE)
    print(f"数据日期范围: {df['date'].min()} - {df['date'].max()}")

    # 2. 归一化close
    close_scaled, scaler = normalize_close(df.copy())

    # 3. 构建训练数据集
    df_scaled = df.copy()
    df_scaled['close'] = close_scaled
    X, y, y_return, dates = build_dataset(df_scaled, N_PAST_DAYS, N_FUTURE_DAYS)

    # 4. 划分训练测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. 建模训练预测
    model = build_lstm_model((N_PAST_DAYS, 1))
    result = train_and_predict(model, X_train, y_train, X_test, y_test, df_scaled, scaler)

    # 6. 买卖点索引
    buy_index = len(df) - 1 if result['buy'] else None
    sell_index = buy_index + result['sell_in_days'] if buy_index is not None else None
    if sell_index is not None and sell_index >= len(df):
        sell_index = len(df) - 1

    # 7. 收盘后不建议买入
    if is_market_closed():
        print("当前已收盘，不建议今日买入。")
        buy_index = None
        result['buy'] = False

    # 8. 输出买入信息
    if buy_index is not None:
        buy_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        buy_price = df.loc[buy_index, 'close']
        print(f"买入时间：{buy_date_str}, 买入价格：¥{buy_price:.2f}")
    else:
        print("无买入信号")

    # 9. 输出卖出信息
    if sell_index is not None:
        sell_date = df.loc[sell_index, 'date']
        if hasattr(sell_date, 'hour') and sell_date.hour == 0 and sell_date.minute == 0 and sell_date.second == 0:
            sell_datetime = sell_date.replace(hour=15, minute=0, second=0)
        else:
            sell_datetime = sell_date
        sell_price = df.loc[sell_index, 'close']
        print(f"建议卖出时间：{sell_datetime.strftime('%Y-%m-%d %H:%M:%S')}, 卖出价格：¥{sell_price:.2f}")

    print(f"Buy index: {buy_index}, Sell index: {sell_index}")
    print(f"Estimated return: {result.get('estimated_return')}")

    # 10. 绘制K线图（可以注释掉避免每次都弹窗）
    # plot_kline_with_signals(df, buy_index=buy_index, sell_index=sell_index, profit=result['estimated_return'])

    if result['buy']:
        print(f"建议持有 {result['sell_in_days']} 天后卖出，预估涨幅为 {result['estimated_return']:.2%}")
    else:
        print("当前不建议买入。")


def main_loop():
    print("开始实时监控，交易时间内每3秒更新一次，非交易时间等待。")
    while True:
        now = datetime.now()
        if is_market_closed():
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 市场已收盘，等待明日开盘...")
            # 一般隔夜睡眠时间长一点
            time.sleep(60 * 60)  # 1小时

        if is_trading_time():
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 交易时间，开始执行预测流程...")
            try:
                run_once()
            except Exception as e:
                print(f"运行异常: {e}")
            time.sleep(3)
        else:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 非交易时间，等待30秒后继续检测。")
            time.sleep(30)

if __name__ == '__main__':
    main_loop()