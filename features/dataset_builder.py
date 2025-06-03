import numpy as np

def build_dataset(df, n_past, n_future):
    X, y, y_return, dates = [], [], [], []
    close = df['close'].values
    for i in range(n_past, len(df) - n_future):
        X.append(close[i - n_past:i])
        future_return = (close[i + n_future] - close[i]) / close[i]
        y_return.append(future_return)
        y.append(1 if future_return > 0 else 0)
        dates.append(df['date'].values[i])
    X = np.array(X).reshape(-1, n_past, 1)
    return X, np.array(y), np.array(y_return), np.array(dates)