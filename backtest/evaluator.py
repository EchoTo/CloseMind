"""
回测评估模块
计算IC/ICIR、分组收益、回测指标等
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class BacktestEvaluator:
    """回测评估器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        backtest_config = config.get("backtest", {})
        portfolio_config = backtest_config.get("portfolio", {})

        # 交易成本
        costs_config = backtest_config.get("costs", {})
        self.commission = costs_config.get("commission", 0.001)
        self.slippage = costs_config.get("slippage", 0.001)
        self.stamp_duty = costs_config.get("stamp_duty", 0.001)

        # 持仓配置
        self.top_n = portfolio_config.get("top_n", 50)
        self.n_groups = 10  # 分组数

        logger.info("BacktestEvaluator initialized")

    def calculate_ic(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        pred_col: str = "prediction",
        actual_col: str = "return_1d"
    ) -> pd.DataFrame:
        """
        计算IC(Information Coefficient)

        Args:
            predictions: 预测值DataFrame,包含date, code, prediction
            actuals: 实际值DataFrame,包含date, code, return_1d
            pred_col: 预测列名
            actual_col: 实际列名

        Returns:
            每日IC DataFrame
        """
        # 合并数据
        df = predictions.merge(
            actuals[["date", "code", actual_col]],
            on=["date", "code"],
            how="inner"
        )

        # 计算每日IC
        daily_ic = df.groupby("date").apply(
            lambda x: x[pred_col].corr(x[actual_col])
            if len(x) > 10 else np.nan
        )

        ic_df = pd.DataFrame({
            "date": daily_ic.index,
            "ic": daily_ic.values
        })

        return ic_df

    def calculate_rank_ic(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        pred_col: str = "prediction",
        actual_col: str = "return_1d"
    ) -> pd.DataFrame:
        """
        计算Rank IC(Spearman相关系数)

        Args:
            predictions: 预测值DataFrame
            actuals: 实际值DataFrame
            pred_col: 预测列名
            actual_col: 实际列名

        Returns:
            每日Rank IC DataFrame
        """
        df = predictions.merge(
            actuals[["date", "code", actual_col]],
            on=["date", "code"],
            how="inner"
        )

        # 计算每日Rank IC
        daily_rank_ic = df.groupby("date").apply(
            lambda x: stats.spearmanr(x[pred_col], x[actual_col])[0]
            if len(x) > 10 else np.nan
        )

        ic_df = pd.DataFrame({
            "date": daily_rank_ic.index,
            "rank_ic": daily_rank_ic.values
        })

        return ic_df

    def calculate_ic_summary(self, ic_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算IC汇总统计

        Args:
            ic_df: IC DataFrame

        Returns:
            IC统计字典
        """
        ic_col = "ic" if "ic" in ic_df.columns else "rank_ic"
        ic_values = ic_df[ic_col].dropna()

        if len(ic_values) == 0:
            return {}

        return {
            "ic_mean": ic_values.mean(),
            "ic_std": ic_values.std(),
            "icir": ic_values.mean() / (ic_values.std() + 1e-10),
            "ic_positive_rate": (ic_values > 0).mean(),
            "ic_abs_mean": ic_values.abs().mean(),
            "ic_max": ic_values.max(),
            "ic_min": ic_values.min(),
            "ic_count": len(ic_values)
        }

    def calculate_group_returns(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        pred_col: str = "prediction",
        actual_col: str = "return_1d",
        n_groups: int = None
    ) -> pd.DataFrame:
        """
        计算分组收益

        Args:
            predictions: 预测值DataFrame
            actuals: 实际值DataFrame
            pred_col: 预测列名
            actual_col: 实际列名
            n_groups: 分组数

        Returns:
            分组收益DataFrame
        """
        if n_groups is None:
            n_groups = self.n_groups

        df = predictions.merge(
            actuals[["date", "code", actual_col]],
            on=["date", "code"],
            how="inner"
        )

        # 每日分组
        df["group"] = df.groupby("date")[pred_col].transform(
            lambda x: pd.qcut(x, n_groups, labels=range(1, n_groups + 1), duplicates="drop")
        )

        # 计算每组每日收益
        group_returns = df.groupby(["date", "group"])[actual_col].mean().unstack()

        # 计算累计收益
        cumulative_returns = (1 + group_returns).cumprod()

        # 计算Top组和Bottom组差异
        long_short = group_returns[n_groups] - group_returns[1]

        return pd.DataFrame({
            **{f"group_{i}": group_returns[i] for i in range(1, n_groups + 1)},
            "long_short": long_short
        })

    def calculate_group_summary(
        self,
        group_returns: pd.DataFrame,
        n_groups: int = None
    ) -> pd.DataFrame:
        """
        计算分组收益汇总

        Args:
            group_returns: 分组收益DataFrame
            n_groups: 分组数

        Returns:
            汇总DataFrame
        """
        if n_groups is None:
            n_groups = self.n_groups

        results = []
        for i in range(1, n_groups + 1):
            col = f"group_{i}"
            if col not in group_returns.columns:
                continue

            returns = group_returns[col].dropna()
            results.append({
                "group": i,
                "mean_return": returns.mean() * 252,  # 年化
                "volatility": returns.std() * np.sqrt(252),
                "sharpe": returns.mean() / (returns.std() + 1e-10) * np.sqrt(252),
                "win_rate": (returns > 0).mean(),
                "cumulative_return": (1 + returns).prod() - 1
            })

        # 多空组合
        if "long_short" in group_returns.columns:
            ls_returns = group_returns["long_short"].dropna()
            results.append({
                "group": "Long-Short",
                "mean_return": ls_returns.mean() * 252,
                "volatility": ls_returns.std() * np.sqrt(252),
                "sharpe": ls_returns.mean() / (ls_returns.std() + 1e-10) * np.sqrt(252),
                "win_rate": (ls_returns > 0).mean(),
                "cumulative_return": (1 + ls_returns).prod() - 1
            })

        return pd.DataFrame(results)

    def run_backtest(
        self,
        predictions: pd.DataFrame,
        market_data: pd.DataFrame,
        initial_capital: float = 1000000,
        pred_col: str = "prediction"
    ) -> Dict[str, Any]:
        """
        运行完整回测

        Args:
            predictions: 预测值DataFrame
            market_data: 市场数据DataFrame
            initial_capital: 初始资金
            pred_col: 预测列名

        Returns:
            回测结果
        """
        logger.info("Running backtest...")

        # 确保日期排序
        dates = sorted(predictions["date"].unique())

        # 初始化
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}  # {code: shares}

        # 记录
        portfolio_history = []
        trade_history = []

        for date in dates:
            # 获取当日数据
            day_pred = predictions[predictions["date"] == date]
            day_market = market_data[market_data["date"] == date]

            if len(day_market) == 0:
                continue

            # 更新持仓价值
            portfolio_value, positions = self._update_positions(
                positions, day_market, cash
            )

            # 生成新的目标持仓
            target_positions = self._generate_target_positions(
                day_pred, day_market, portfolio_value, pred_col
            )

            # 执行交易
            trades, cash_change, cost = self._execute_trades(
                positions, target_positions, day_market, portfolio_value
            )

            cash += cash_change - cost
            positions = target_positions

            # 记录
            portfolio_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "num_positions": len(positions),
                "cost": cost
            })

            trade_history.extend(trades)

        # 计算回测指标
        portfolio_df = pd.DataFrame(portfolio_history)
        metrics = self._calculate_backtest_metrics(portfolio_df)

        return {
            "portfolio_history": portfolio_df,
            "trade_history": pd.DataFrame(trade_history),
            "metrics": metrics
        }

    def _update_positions(
        self,
        positions: Dict[str, int],
        market_data: pd.DataFrame,
        cash: float
    ) -> Tuple[float, Dict[str, int]]:
        """
        更新持仓价值

        Args:
            positions: 当前持仓
            market_data: 市场数据
            cash: 现金

        Returns:
            (组合价值, 更新后的持仓)
        """
        price_dict = market_data.set_index("code")["close"].to_dict()

        position_value = 0
        updated_positions = {}

        for code, shares in positions.items():
            if code in price_dict:
                position_value += shares * price_dict[code]
                updated_positions[code] = shares

        return cash + position_value, updated_positions

    def _generate_target_positions(
        self,
        predictions: pd.DataFrame,
        market_data: pd.DataFrame,
        portfolio_value: float,
        pred_col: str
    ) -> Dict[str, int]:
        """
        生成目标持仓

        Args:
            predictions: 预测值
            market_data: 市场数据
            portfolio_value: 组合价值
            pred_col: 预测列名

        Returns:
            目标持仓 {code: shares}
        """
        # 选择Top N股票
        top_stocks = predictions.nlargest(self.top_n, pred_col)

        # 合并价格
        top_stocks = top_stocks.merge(
            market_data[["code", "close", "is_limit_up", "is_suspended"]],
            on="code",
            how="left"
        )

        # 过滤不可交易
        top_stocks = top_stocks[
            ~top_stocks["is_limit_up"].fillna(False) &
            ~top_stocks["is_suspended"].fillna(False)
        ]

        if len(top_stocks) == 0:
            return {}

        # 等权分配
        weight = 1.0 / len(top_stocks)
        target_value = portfolio_value * weight

        positions = {}
        for _, row in top_stocks.iterrows():
            if row["close"] > 0:
                shares = int(target_value / row["close"] / 100) * 100  # 100股为单位
                if shares > 0:
                    positions[row["code"]] = shares

        return positions

    def _execute_trades(
        self,
        current: Dict[str, int],
        target: Dict[str, int],
        market_data: pd.DataFrame,
        portfolio_value: float
    ) -> Tuple[List[Dict], float, float]:
        """
        执行交易

        Args:
            current: 当前持仓
            target: 目标持仓
            market_data: 市场数据
            portfolio_value: 组合价值

        Returns:
            (交易列表, 现金变化, 交易成本)
        """
        price_dict = market_data.set_index("code")["close"].to_dict()

        trades = []
        cash_change = 0
        total_cost = 0

        # 卖出
        for code, shares in current.items():
            if code not in target or target[code] < shares:
                sell_shares = shares - target.get(code, 0)
                if code in price_dict:
                    value = sell_shares * price_dict[code]
                    cost = value * (self.commission + self.slippage + self.stamp_duty)
                    cash_change += value - cost
                    total_cost += cost

                    trades.append({
                        "code": code,
                        "direction": "sell",
                        "shares": sell_shares,
                        "price": price_dict[code],
                        "value": value,
                        "cost": cost
                    })

        # 买入
        for code, shares in target.items():
            if code not in current or current[code] < shares:
                buy_shares = shares - current.get(code, 0)
                if code in price_dict:
                    value = buy_shares * price_dict[code]
                    cost = value * (self.commission + self.slippage)
                    cash_change -= value + cost
                    total_cost += cost

                    trades.append({
                        "code": code,
                        "direction": "buy",
                        "shares": buy_shares,
                        "price": price_dict[code],
                        "value": value,
                        "cost": cost
                    })

        return trades, cash_change, total_cost

    def _calculate_backtest_metrics(
        self,
        portfolio_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        计算回测指标

        Args:
            portfolio_df: 组合历史DataFrame

        Returns:
            指标字典
        """
        values = portfolio_df["portfolio_value"].values
        returns = np.diff(values) / values[:-1]

        # 基本指标
        total_return = (values[-1] - values[0]) / values[0]
        n_days = len(values)
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率(假设无风险利率为0.02)
        risk_free = 0.02
        sharpe = (annual_return - risk_free) / (volatility + 1e-10)

        # 最大回撤
        cumulative = values / values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar比率
        calmar = annual_return / (abs(max_drawdown) + 1e-10)

        # 胜率
        win_rate = (returns > 0).mean()

        # 交易成本
        total_cost = portfolio_df["cost"].sum()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "win_rate": win_rate,
            "total_cost": total_cost,
            "cost_ratio": total_cost / values[0],
            "n_days": n_days
        }

    def generate_report(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        pred_col: str = "prediction"
    ) -> Dict[str, Any]:
        """
        生成完整评估报告

        Args:
            predictions: 预测值DataFrame
            actuals: 实际值DataFrame
            pred_col: 预测列名

        Returns:
            报告字典
        """
        logger.info("Generating evaluation report...")

        report = {}

        # IC分析
        ic_df = self.calculate_ic(predictions, actuals, pred_col)
        rank_ic_df = self.calculate_rank_ic(predictions, actuals, pred_col)

        report["ic_daily"] = ic_df
        report["rank_ic_daily"] = rank_ic_df
        report["ic_summary"] = self.calculate_ic_summary(ic_df)
        report["rank_ic_summary"] = self.calculate_ic_summary(rank_ic_df)

        # 分组收益
        group_returns = self.calculate_group_returns(predictions, actuals, pred_col)
        report["group_returns"] = group_returns
        report["group_summary"] = self.calculate_group_summary(group_returns)

        # 打印摘要
        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict[str, Any]):
        """打印评估摘要"""
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)

        # IC
        ic_summary = report.get("ic_summary", {})
        logger.info(f"IC Mean: {ic_summary.get('ic_mean', 0):.4f}")
        logger.info(f"IC Std: {ic_summary.get('ic_std', 0):.4f}")
        logger.info(f"ICIR: {ic_summary.get('icir', 0):.4f}")
        logger.info(f"IC Positive Rate: {ic_summary.get('ic_positive_rate', 0):.2%}")

        # Rank IC
        rank_ic_summary = report.get("rank_ic_summary", {})
        logger.info(f"Rank IC Mean: {rank_ic_summary.get('ic_mean', 0):.4f}")
        logger.info(f"Rank ICIR: {rank_ic_summary.get('icir', 0):.4f}")

        # 分组收益
        group_summary = report.get("group_summary")
        if group_summary is not None and len(group_summary) > 0:
            logger.info("-" * 50)
            logger.info("Group Returns:")
            for _, row in group_summary.iterrows():
                logger.info(
                    f"  {row['group']}: Annual Return={row['mean_return']:.2%}, "
                    f"Sharpe={row['sharpe']:.2f}"
                )

        logger.info("=" * 50)
