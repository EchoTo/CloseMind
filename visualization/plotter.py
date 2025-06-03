import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_kline_with_signals(df, buy_index, sell_index, profit):
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制K线图
    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([row['date'], row['date']], [row['low'], row['high']], color=color)
        ax.plot([row['date'], row['date']], [row['open'], row['close']], color=color, linewidth=5)

    # 买入点
    if buy_index is not None:
        buy_price = df.loc[buy_index, 'close']
        ax.scatter(df.loc[buy_index, 'date'], buy_price, marker='^', color='blue', s=100, label='Buy Signal')
        ax.text(df.loc[buy_index, 'date'], buy_price * 1.01, f"Buy\n¥{buy_price:.2f}", color='blue', fontsize=10, ha='center')

    # 卖出点
    if sell_index is not None and 0 <= sell_index < len(df):
        sell_price = df.loc[sell_index, 'close']
        ax.scatter(df.loc[sell_index, 'date'], sell_price, marker='v', color='black', s=100, label='Sell Signal')
        ax.text(df.loc[sell_index, 'date'], sell_price * 0.99, f"Sell\n¥{sell_price:.2f}", color='black', fontsize=10, ha='center')
        ax.text(df['date'].iloc[-1], df['high'].max(), f"Predicted ROI: {profit:.2%}", fontsize=12, color='purple', ha='right')

    ax.set_title("K-Line Chart with AI Buy/Sell Signal", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.grid(True)
    plt.tight_layout()
    plt.show()