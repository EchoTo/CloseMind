"""
组合优化模块
负责根据信号构建和优化投资组合
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        backtest_config = config.get("backtest", {})
        portfolio_config = backtest_config.get("portfolio", {})

        # 持仓配置
        self.top_n = portfolio_config.get("top_n", 50)
        self.rebalance = portfolio_config.get("rebalance", "daily")
        self.max_weight = portfolio_config.get("max_weight", 0.1)
        self.min_weight = 1.0 / self.top_n * 0.5  # 最小权重为等权的一半

        # 交易成本
        costs_config = backtest_config.get("costs", {})
        self.commission = costs_config.get("commission", 0.001)
        self.slippage = costs_config.get("slippage", 0.001)
        self.stamp_duty = costs_config.get("stamp_duty", 0.001)

        # 风险控制
        risk_config = backtest_config.get("risk", {})
        self.max_drawdown = risk_config.get("max_drawdown", 0.2)
        self.stop_loss = risk_config.get("stop_loss", 0.1)

        logger.info(f"PortfolioOptimizer initialized. Top N: {self.top_n}")

    def build_portfolio(
        self,
        signals: pd.DataFrame,
        date: str,
        method: str = "equal_weight"
    ) -> Dict[str, float]:
        """
        构建投资组合

        Args:
            signals: 信号DataFrame
            date: 日期
            method: 权重方法 (equal_weight, score_weight, risk_parity)

        Returns:
            持仓权重 {code: weight}
        """
        # 获取当日买入信号
        day_signals = signals[signals["date"] == date].copy()

        # 只选择买入信号
        buy_signals = day_signals[
            day_signals["signal"].isin(["strong_buy", "buy"])
        ].sort_values("combined_score", ascending=False)

        # 选取Top N
        selected = buy_signals.head(self.top_n)

        if len(selected) == 0:
            return {}

        # 计算权重
        if method == "equal_weight":
            weights = self._equal_weight(selected)
        elif method == "score_weight":
            weights = self._score_weight(selected)
        elif method == "risk_parity":
            weights = self._risk_parity_weight(selected, signals)
        else:
            weights = self._equal_weight(selected)

        return weights

    def _equal_weight(self, selected: pd.DataFrame) -> Dict[str, float]:
        """
        等权重

        Args:
            selected: 选中的股票

        Returns:
            权重字典
        """
        n = len(selected)
        weight = 1.0 / n

        return {row["code"]: weight for _, row in selected.iterrows()}

    def _score_weight(self, selected: pd.DataFrame) -> Dict[str, float]:
        """
        按分数加权

        Args:
            selected: 选中的股票

        Returns:
            权重字典
        """
        # 使用softmax归一化分数
        scores = selected["combined_score"].values
        scores = scores - scores.max()  # 防止溢出
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()

        # 限制最大权重
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()  # 重新归一化

        return dict(zip(selected["code"], weights))

    def _risk_parity_weight(
        self,
        selected: pd.DataFrame,
        historical_data: pd.DataFrame,
        lookback_days: int = 60
    ) -> Dict[str, float]:
        """
        风险平价权重

        Args:
            selected: 选中的股票
            historical_data: 历史数据
            lookback_days: 回看天数

        Returns:
            权重字典
        """
        codes = selected["code"].tolist()

        # 获取历史收益率
        if "return_1d" not in historical_data.columns:
            return self._equal_weight(selected)

        returns = historical_data[
            historical_data["code"].isin(codes)
        ].pivot(index="date", columns="code", values="return_1d")

        # 取最近N天
        returns = returns.tail(lookback_days).dropna(axis=1, how="all")

        if returns.empty or len(returns.columns) < 2:
            return self._equal_weight(selected)

        # 计算波动率
        volatilities = returns.std()

        # 风险平价: 权重与波动率成反比
        inv_vol = 1 / (volatilities + 1e-10)
        weights = inv_vol / inv_vol.sum()

        # 限制权重
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()

        return weights.to_dict()

    def optimize_portfolio(
        self,
        selected: pd.DataFrame,
        returns_cov: pd.DataFrame,
        expected_returns: pd.Series,
        risk_aversion: float = 1.0
    ) -> Dict[str, float]:
        """
        均值-方差优化

        Args:
            selected: 选中的股票
            returns_cov: 收益率协方差矩阵
            expected_returns: 预期收益率
            risk_aversion: 风险厌恶系数

        Returns:
            最优权重
        """
        codes = selected["code"].tolist()
        n = len(codes)

        if n == 0:
            return {}

        # 只保留有数据的股票
        common_codes = list(set(codes) & set(returns_cov.columns) & set(expected_returns.index))
        if len(common_codes) < 2:
            return self._equal_weight(selected[selected["code"].isin(common_codes)])

        cov = returns_cov.loc[common_codes, common_codes].values
        mu = expected_returns.loc[common_codes].values

        # 目标函数: 最大化 收益 - 风险厌恶 * 方差
        def objective(w):
            ret = np.dot(w, mu)
            risk = np.dot(w.T, np.dot(cov, w))
            return -(ret - risk_aversion * risk)

        # 约束: 权重和为1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # 边界: 每只股票权重在 [min_weight, max_weight]
        bounds = [(self.min_weight, self.max_weight)] * len(common_codes)

        # 初始权重
        w0 = np.ones(len(common_codes)) / len(common_codes)

        # 优化
        try:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                weights = dict(zip(common_codes, result.x))
            else:
                logger.warning("Optimization failed, using equal weight")
                weights = self._equal_weight(selected[selected["code"].isin(common_codes)])
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            weights = self._equal_weight(selected[selected["code"].isin(common_codes)])

        return weights

    def calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """
        计算换手率

        Args:
            old_weights: 旧权重
            new_weights: 新权重

        Returns:
            换手率
        """
        all_codes = set(old_weights.keys()) | set(new_weights.keys())

        turnover = 0
        for code in all_codes:
            old_w = old_weights.get(code, 0)
            new_w = new_weights.get(code, 0)
            turnover += abs(new_w - old_w)

        return turnover / 2  # 单边换手率

    def calculate_transaction_costs(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """
        计算交易成本

        Args:
            old_weights: 旧权重
            new_weights: 新权重
            portfolio_value: 组合价值

        Returns:
            交易成本
        """
        all_codes = set(old_weights.keys()) | set(new_weights.keys())

        total_cost = 0
        for code in all_codes:
            old_w = old_weights.get(code, 0)
            new_w = new_weights.get(code, 0)

            trade_value = abs(new_w - old_w) * portfolio_value

            # 买入成本: 佣金 + 滑点
            # 卖出成本: 佣金 + 滑点 + 印花税
            if new_w > old_w:
                # 买入
                cost = trade_value * (self.commission + self.slippage)
            else:
                # 卖出
                cost = trade_value * (self.commission + self.slippage + self.stamp_duty)

            total_cost += cost

        return total_cost

    def apply_risk_control(
        self,
        weights: Dict[str, float],
        current_returns: Dict[str, float],
        cumulative_return: float
    ) -> Dict[str, float]:
        """
        应用风险控制

        Args:
            weights: 当前权重
            current_returns: 当日各股票收益
            cumulative_return: 累计收益

        Returns:
            调整后的权重
        """
        adjusted_weights = weights.copy()

        # 个股止损
        for code, weight in list(adjusted_weights.items()):
            if code in current_returns:
                if current_returns[code] < -self.stop_loss:
                    logger.info(f"Stop loss triggered for {code}")
                    adjusted_weights[code] = 0

        # 组合最大回撤控制
        if cumulative_return < -self.max_drawdown:
            logger.warning(f"Max drawdown reached: {cumulative_return:.2%}")
            # 降低仓位
            for code in adjusted_weights:
                adjusted_weights[code] *= 0.5

        # 重新归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        signals: pd.DataFrame,
        date: str,
        method: str = "equal_weight"
    ) -> Tuple[Dict[str, float], float]:
        """
        组合再平衡

        Args:
            current_weights: 当前权重
            signals: 信号DataFrame
            date: 日期
            method: 权重方法

        Returns:
            (新权重, 换手率)
        """
        # 构建新组合
        new_weights = self.build_portfolio(signals, date, method)

        # 计算换手率
        turnover = self.calculate_turnover(current_weights, new_weights)

        return new_weights, turnover

    def get_portfolio_stats(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        获取组合统计信息

        Args:
            weights: 权重
            returns: 收益率数据

        Returns:
            统计信息
        """
        codes = list(weights.keys())
        w = np.array([weights[c] for c in codes])

        # 获取收益率
        ret_data = returns[returns["code"].isin(codes)].pivot(
            index="date", columns="code", values="return_1d"
        )

        if ret_data.empty:
            return {}

        # 对齐
        ret_data = ret_data[codes].dropna()

        if len(ret_data) < 20:
            return {}

        # 组合收益率
        portfolio_returns = ret_data.values @ w

        # 计算统计量
        stats = {
            "mean_return": portfolio_returns.mean() * 252,  # 年化
            "volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe": portfolio_returns.mean() / (portfolio_returns.std() + 1e-10) * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "num_stocks": len(codes),
            "top_weight": max(weights.values()),
            "concentration": sum([w ** 2 for w in weights.values()])  # HHI
        }

        return stats

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            最大回撤
        """
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
