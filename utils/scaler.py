from sklearn.preprocessing import MinMaxScaler

def normalize_close(df):
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(df[['close']])  # 返回二维数组，shape=(n,1)
    return close_scaled.flatten(), scaler  # flatten成1维数组