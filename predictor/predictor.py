import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from config import N_PAST_DAYS, PREDICT_FORWARD_DAYS

def train_and_predict(model, X_train, y_train, X_test, y_test, df, scaler):
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model accuracy: {acc*100:.2f}%")

    # 实时预测
    last_seq = df['close'].values[-N_PAST_DAYS:].reshape(1, N_PAST_DAYS, 1)
    prob = model.predict(last_seq)[0][0]
    signal = prob > 0.5

    print(f"涨的概率：{prob:.3f}，买入信号：{'是' if signal else '否'}")

    # 预测未来几天的涨幅概率
    future_preds = []
    close_vals = df['close'].values
    for d in range(1, PREDICT_FORWARD_DAYS + 1):
        if len(close_vals) - N_PAST_DAYS - d < 0:
            break
        x = close_vals[-(N_PAST_DAYS + d):-d].reshape(1, N_PAST_DAYS, 1)
        p = model.predict(x)[0][0]
        future_preds.append((d, p))

    best_day, best_prob = max(future_preds, key=lambda x: x[1])
    est_return = best_prob * 0.02

    print(f"建议持有 {best_day} 天后卖出，预估涨幅为 {est_return*100:.2f}%")

    return {
        'buy': signal,
        'probability': prob,
        'sell_in_days': best_day,
        'estimated_return': est_return
    }