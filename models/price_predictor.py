"""
价格预测模块
用于预测未来K线走势
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PricePredictor:
    """股票价格预测器"""

    def __init__(self, config: Dict):
        """
        初始化预测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.lookback_days = 60  # 使用过去60天数据
        self.forecast_days = 10  # 预测未来10天

    def predict_price(
        self,
        df: pd.DataFrame,
        code: str,
        forecast_days: int = None
    ) -> Optional[pd.DataFrame]:
        """
        预测单只股票未来价格

        Args:
            df: 历史数据DataFrame
            code: 股票代码
            forecast_days: 预测天数

        Returns:
            预测结果DataFrame，包含 date, open, high, low, close, volume
        """
        if forecast_days is None:
            forecast_days = self.forecast_days

        # 获取该股票数据
        stock_data = df[df["code"] == code].copy()
        if len(stock_data) < self.lookback_days:
            logger.warning(f"Insufficient data for {code}: {len(stock_data)} days")
            return None

        stock_data = stock_data.sort_values("date").tail(self.lookback_days)

        # 使用多种方法综合预测
        predictions = self._ensemble_predict(stock_data, forecast_days)

        return predictions

    def _ensemble_predict(
        self,
        data: pd.DataFrame,
        forecast_days: int
    ) -> pd.DataFrame:
        """
        集成预测方法

        Args:
            data: 历史数据
            forecast_days: 预测天数

        Returns:
            预测DataFrame
        """
        # 获取最后一个交易日
        last_date = pd.to_datetime(data["date"].iloc[-1])
        last_close = data["close"].iloc[-1]
        last_volume = data["volume"].iloc[-1]

        # 计算历史统计特征
        returns = data["close"].pct_change().dropna()
        volatility = returns.std()
        avg_return = returns.mean()

        # 计算趋势（线性回归斜率）
        prices = data["close"].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_return = slope / prices[-1]  # 转换为收益率

        # 计算动量
        momentum_5 = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        momentum_20 = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0

        # 高低价波动比例
        high_low_ratio = (data["high"] / data["low"]).mean()
        open_close_ratio = abs(data["open"] - data["close"]).mean() / data["close"].mean()

        # 生成预测日期（跳过周末）
        pred_dates = []
        current_date = last_date
        while len(pred_dates) < forecast_days:
            current_date += timedelta(days=1)
            # 跳过周末（简化处理，实际应用交易日历）
            if current_date.weekday() < 5:
                pred_dates.append(current_date)

        # 综合预测收益率
        # 权重：趋势 40%，动量 30%，均值回归 30%
        base_return = (
            0.4 * trend_return +
            0.2 * momentum_5 / 5 +
            0.1 * momentum_20 / 20 +
            0.3 * avg_return
        )

        # 生成预测价格
        predictions = []
        current_close = last_close
        current_volume = last_volume

        for i, date in enumerate(pred_dates):
            # 添加随机波动（基于历史波动率）
            # 预测越远，不确定性越大
            uncertainty_factor = 1 + i * 0.1
            daily_return = base_return + np.random.normal(0, volatility * uncertainty_factor * 0.5)

            # 限制单日涨跌幅（A股限制10%）
            daily_return = np.clip(daily_return, -0.10, 0.10)

            # 计算收盘价
            pred_close = current_close * (1 + daily_return)

            # 计算开高低价
            intraday_vol = open_close_ratio * (1 + np.random.uniform(-0.3, 0.3))
            pred_open = current_close * (1 + np.random.normal(0, volatility * 0.3))

            if daily_return > 0:
                pred_high = pred_close * (1 + abs(np.random.normal(0, intraday_vol)))
                pred_low = min(pred_open, pred_close) * (1 - abs(np.random.normal(0, intraday_vol * 0.5)))
            else:
                pred_high = max(pred_open, pred_close) * (1 + abs(np.random.normal(0, intraday_vol * 0.5)))
                pred_low = pred_close * (1 - abs(np.random.normal(0, intraday_vol)))

            # 确保 high >= max(open, close) 且 low <= min(open, close)
            pred_high = max(pred_high, pred_open, pred_close)
            pred_low = min(pred_low, pred_open, pred_close)

            # 预测成交量（基于趋势）
            volume_change = 1 + np.random.normal(0, 0.2)
            pred_volume = current_volume * volume_change

            predictions.append({
                "date": date,
                "open": round(pred_open, 2),
                "high": round(pred_high, 2),
                "low": round(pred_low, 2),
                "close": round(pred_close, 2),
                "volume": int(pred_volume),
                "is_prediction": True
            })

            current_close = pred_close
            current_volume = pred_volume

        return pd.DataFrame(predictions)

    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        code: str,
        forecast_days: int = None,
        n_simulations: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        带置信区间的预测（蒙特卡洛模拟）

        Args:
            df: 历史数据
            code: 股票代码
            forecast_days: 预测天数
            n_simulations: 模拟次数

        Returns:
            (中位数预测, 下界预测, 上界预测)
        """
        if forecast_days is None:
            forecast_days = self.forecast_days

        all_predictions = []
        for _ in range(n_simulations):
            pred = self.predict_price(df, code, forecast_days)
            if pred is not None:
                all_predictions.append(pred)

        if not all_predictions:
            return None, None, None

        # 合并所有模拟结果
        all_closes = np.array([p["close"].values for p in all_predictions])

        # 计算分位数
        median_pred = all_predictions[0].copy()
        lower_pred = all_predictions[0].copy()
        upper_pred = all_predictions[0].copy()

        median_pred["close"] = np.percentile(all_closes, 50, axis=0)
        lower_pred["close"] = np.percentile(all_closes, 10, axis=0)
        upper_pred["close"] = np.percentile(all_closes, 90, axis=0)

        # 同样处理其他价格
        all_highs = np.array([p["high"].values for p in all_predictions])
        all_lows = np.array([p["low"].values for p in all_predictions])

        median_pred["high"] = np.percentile(all_highs, 50, axis=0)
        median_pred["low"] = np.percentile(all_lows, 50, axis=0)
        lower_pred["high"] = np.percentile(all_highs, 10, axis=0)
        lower_pred["low"] = np.percentile(all_lows, 10, axis=0)
        upper_pred["high"] = np.percentile(all_highs, 90, axis=0)
        upper_pred["low"] = np.percentile(all_lows, 90, axis=0)

        return median_pred, lower_pred, upper_pred

    def get_prediction_summary(
        self,
        df: pd.DataFrame,
        code: str,
        forecast_days: int = None
    ) -> Dict:
        """
        获取预测摘要

        Args:
            df: 历史数据
            code: 股票代码
            forecast_days: 预测天数

        Returns:
            预测摘要字典
        """
        median, lower, upper = self.predict_with_confidence(
            df, code, forecast_days, n_simulations=50
        )

        if median is None:
            return None

        stock_data = df[df["code"] == code]
        current_price = stock_data["close"].iloc[-1]

        pred_price = median["close"].iloc[-1]
        pred_return = (pred_price / current_price - 1) * 100

        lower_price = lower["close"].iloc[-1]
        upper_price = upper["close"].iloc[-1]

        return {
            "current_price": current_price,
            "predicted_price": pred_price,
            "predicted_return": pred_return,
            "lower_bound": lower_price,
            "upper_bound": upper_price,
            "lower_return": (lower_price / current_price - 1) * 100,
            "upper_return": (upper_price / current_price - 1) * 100,
            "forecast_days": forecast_days or self.forecast_days
        }
