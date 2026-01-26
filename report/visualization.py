"""
可视化模块
生成回测报告和图表
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from loguru import logger

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        paths = config.get("paths", {})
        self.report_dir = Path(paths.get("report_dir", "./reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 图表样式
        self.figsize = (12, 6)
        self.dpi = 100

        logger.info(f"ReportGenerator initialized. Output: {self.report_dir}")

    def plot_ic_analysis(
        self,
        ic_df: pd.DataFrame,
        title: str = "IC Analysis",
        save_name: str = "ic_analysis.png"
    ):
        """
        绘制IC分析图

        Args:
            ic_df: IC DataFrame
            title: 标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ic_col = "ic" if "ic" in ic_df.columns else "rank_ic"
        ic_values = ic_df[ic_col].dropna()
        dates = pd.to_datetime(ic_df["date"])

        # 1. IC时序图
        ax1 = axes[0, 0]
        ax1.bar(dates, ic_values, alpha=0.7, color="steelblue", width=1)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.axhline(y=ic_values.mean(), color="red", linestyle="--", linewidth=1,
                    label=f"Mean: {ic_values.mean():.4f}")
        ax1.set_title("Daily IC")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("IC")
        ax1.legend()
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. IC分布直方图
        ax2 = axes[0, 1]
        ax2.hist(ic_values, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax2.axvline(x=ic_values.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {ic_values.mean():.4f}")
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax2.set_title("IC Distribution")
        ax2.set_xlabel("IC")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        # 3. 累计IC
        ax3 = axes[1, 0]
        cumulative_ic = ic_values.cumsum()
        ax3.plot(dates[:len(cumulative_ic)], cumulative_ic.values, color="steelblue", linewidth=1.5)
        ax3.fill_between(dates[:len(cumulative_ic)], 0, cumulative_ic.values,
                         alpha=0.3, color="steelblue")
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_title("Cumulative IC")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Cumulative IC")
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # 4. IC统计摘要
        ax4 = axes[1, 1]
        ax4.axis("off")
        stats_text = f"""
        IC Statistics
        ─────────────────────
        Mean:           {ic_values.mean():.4f}
        Std:            {ic_values.std():.4f}
        ICIR:           {ic_values.mean() / (ic_values.std() + 1e-10):.4f}
        Positive Rate:  {(ic_values > 0).mean():.2%}
        Max:            {ic_values.max():.4f}
        Min:            {ic_values.min():.4f}
        Count:          {len(ic_values)}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, family="monospace",
                 verticalalignment="center")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"IC analysis plot saved to {save_path}")

    def plot_group_returns(
        self,
        group_returns: pd.DataFrame,
        title: str = "Group Returns Analysis",
        save_name: str = "group_returns.png"
    ):
        """
        绘制分组收益图

        Args:
            group_returns: 分组收益DataFrame
            title: 标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 获取分组列
        group_cols = [c for c in group_returns.columns if c.startswith("group_")]
        n_groups = len(group_cols)

        # 1. 分组累计收益曲线
        ax1 = axes[0, 0]
        colors = plt.cm.RdYlGn(np.linspace(0, 1, n_groups))

        for i, col in enumerate(group_cols):
            cumulative = (1 + group_returns[col].fillna(0)).cumprod()
            ax1.plot(group_returns.index, cumulative, label=f"Group {i+1}",
                     color=colors[i], linewidth=1.5)

        ax1.set_title("Cumulative Returns by Group")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 多空组合收益
        ax2 = axes[0, 1]
        if "long_short" in group_returns.columns:
            ls_cumulative = (1 + group_returns["long_short"].fillna(0)).cumprod()
            ax2.plot(group_returns.index, ls_cumulative, color="steelblue", linewidth=2)
            ax2.fill_between(group_returns.index, 1, ls_cumulative,
                             alpha=0.3, color="steelblue")
        ax2.axhline(y=1, color="black", linestyle="--", linewidth=0.5)
        ax2.set_title("Long-Short Portfolio Cumulative Return")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cumulative Return")
        ax2.grid(True, alpha=0.3)

        # 3. 分组平均收益柱状图
        ax3 = axes[1, 0]
        mean_returns = [group_returns[col].mean() * 252 for col in group_cols]
        bars = ax3.bar(range(1, n_groups + 1), mean_returns, color=colors, edgecolor="white")

        # 添加数值标签
        for bar, val in zip(bars, mean_returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=9)

        ax3.set_title("Annualized Mean Return by Group")
        ax3.set_xlabel("Group")
        ax3.set_ylabel("Annual Return")
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # 4. 分组Sharpe比率
        ax4 = axes[1, 1]
        sharpe_ratios = []
        for col in group_cols:
            returns = group_returns[col].dropna()
            sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
            sharpe_ratios.append(sharpe)

        bars = ax4.bar(range(1, n_groups + 1), sharpe_ratios, color=colors, edgecolor="white")

        for bar, val in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax4.set_title("Sharpe Ratio by Group")
        ax4.set_xlabel("Group")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Group returns plot saved to {save_path}")

    def plot_backtest_results(
        self,
        portfolio_history: pd.DataFrame,
        metrics: Dict[str, float],
        title: str = "Backtest Results",
        save_name: str = "backtest_results.png"
    ):
        """
        绘制回测结果图

        Args:
            portfolio_history: 组合历史
            metrics: 回测指标
            title: 标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        dates = pd.to_datetime(portfolio_history["date"])
        values = portfolio_history["portfolio_value"]
        returns = values.pct_change().dropna()

        # 1. 组合净值曲线
        ax1 = axes[0, 0]
        normalized = values / values.iloc[0]
        ax1.plot(dates, normalized, color="steelblue", linewidth=2)
        ax1.fill_between(dates, 1, normalized, alpha=0.3, color="steelblue")
        ax1.axhline(y=1, color="black", linestyle="--", linewidth=0.5)
        ax1.set_title("Portfolio Net Value")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Net Value")
        ax1.grid(True, alpha=0.3)

        # 2. 回撤曲线
        ax2 = axes[0, 1]
        running_max = normalized.cummax()
        drawdown = (normalized - running_max) / running_max
        ax2.fill_between(dates, 0, drawdown, alpha=0.7, color="tomato")
        ax2.set_title("Drawdown")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)

        # 3. 日收益率分布
        ax3 = axes[1, 0]
        ax3.hist(returns, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax3.axvline(x=returns.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {returns.mean():.4f}")
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax3.set_title("Daily Returns Distribution")
        ax3.set_xlabel("Return")
        ax3.set_ylabel("Frequency")
        ax3.legend()

        # 4. 指标摘要
        ax4 = axes[1, 1]
        ax4.axis("off")
        metrics_text = f"""
        Backtest Metrics
        ─────────────────────────
        Total Return:       {metrics.get('total_return', 0):.2%}
        Annual Return:      {metrics.get('annual_return', 0):.2%}
        Volatility:         {metrics.get('volatility', 0):.2%}
        Sharpe Ratio:       {metrics.get('sharpe', 0):.2f}
        Max Drawdown:       {metrics.get('max_drawdown', 0):.2%}
        Calmar Ratio:       {metrics.get('calmar', 0):.2f}
        Win Rate:           {metrics.get('win_rate', 0):.2%}
        Trading Days:       {metrics.get('n_days', 0)}
        Total Cost:         {metrics.get('cost_ratio', 0):.2%}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, family="monospace",
                 verticalalignment="center")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Backtest results plot saved to {save_path}")

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 30,
        title: str = "Feature Importance",
        save_name: str = "feature_importance.png"
    ):
        """
        绘制特征重要性图

        Args:
            importance_df: 特征重要性DataFrame
            top_n: 显示前N个特征
            title: 标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

        # 取Top N
        top_features = importance_df.head(top_n)

        # 水平条形图
        y_pos = range(len(top_features))
        bars = ax.barh(y_pos, top_features["importance"], color="steelblue", alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(title)

        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}")

    def plot_model_comparison(
        self,
        model_metrics: pd.DataFrame,
        title: str = "Model Comparison",
        save_name: str = "model_comparison.png"
    ):
        """
        绘制模型对比图

        Args:
            model_metrics: 模型指标DataFrame
            title: 标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        models = model_metrics["model"]
        colors = plt.cm.Set2(range(len(models)))

        # 1. IC对比
        ax1 = axes[0]
        bars = ax1.bar(models, model_metrics["IC"], color=colors)
        ax1.set_title("IC Comparison")
        ax1.set_ylabel("IC")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        for bar, val in zip(bars, model_metrics["IC"]):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom")

        # 2. 各指标雷达图(如果有更多指标)
        ax2 = axes[1]
        if len(model_metrics.columns) > 2:
            # 简化为条形图对比
            metrics_to_plot = [c for c in model_metrics.columns if c != "model"]
            x = np.arange(len(metrics_to_plot))
            width = 0.8 / len(models)

            for i, (_, row) in enumerate(model_metrics.iterrows()):
                values = [row[m] for m in metrics_to_plot]
                ax2.bar(x + i * width, values, width, label=row["model"], color=colors[i])

            ax2.set_xticks(x + width * (len(models) - 1) / 2)
            ax2.set_xticklabels(metrics_to_plot)
            ax2.legend()
            ax2.set_title("Metrics Comparison")
        else:
            ax2.axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Model comparison plot saved to {save_path}")

    def plot_prediction_distribution(
        self,
        predictions: pd.DataFrame,
        date: str,
        pred_col: str = "prediction",
        title: str = None,
        save_name: str = "prediction_distribution.png"
    ):
        """
        绘制预测分布图

        Args:
            predictions: 预测DataFrame
            date: 日期
            pred_col: 预测列名
            title: 标题
            save_name: 保存文件名
        """
        day_pred = predictions[predictions["date"] == date]

        if len(day_pred) == 0:
            logger.warning(f"No predictions for date {date}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. 预测值分布
        ax1 = axes[0]
        ax1.hist(day_pred[pred_col], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax1.axvline(x=day_pred[pred_col].mean(), color="red", linestyle="--",
                    label=f"Mean: {day_pred[pred_col].mean():.4f}")
        ax1.set_title(f"Prediction Distribution ({date})")
        ax1.set_xlabel("Prediction Score")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # 2. Top/Bottom股票
        ax2 = axes[1]
        ax2.axis("off")

        top_10 = day_pred.nlargest(10, pred_col)
        bottom_10 = day_pred.nsmallest(10, pred_col)

        text = "Top 10 Stocks:\n"
        for _, row in top_10.iterrows():
            text += f"  {row['code']}: {row[pred_col]:.4f}\n"

        text += "\nBottom 10 Stocks:\n"
        for _, row in bottom_10.iterrows():
            text += f"  {row['code']}: {row[pred_col]:.4f}\n"

        ax2.text(0.1, 0.9, text, fontsize=10, family="monospace",
                 verticalalignment="top", transform=ax2.transAxes)

        if title is None:
            title = f"Prediction Analysis - {date}"
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = self.report_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Prediction distribution plot saved to {save_path}")

    def generate_full_report(
        self,
        evaluation_report: Dict[str, Any],
        backtest_results: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        model_comparison: Optional[pd.DataFrame] = None
    ):
        """
        生成完整报告

        Args:
            evaluation_report: 评估报告
            backtest_results: 回测结果
            feature_importance: 特征重要性
            model_comparison: 模型对比
        """
        logger.info("Generating full report...")

        # IC分析
        if "ic_daily" in evaluation_report:
            self.plot_ic_analysis(
                evaluation_report["ic_daily"],
                title="IC Analysis",
                save_name="1_ic_analysis.png"
            )

        if "rank_ic_daily" in evaluation_report:
            self.plot_ic_analysis(
                evaluation_report["rank_ic_daily"],
                title="Rank IC Analysis",
                save_name="2_rank_ic_analysis.png"
            )

        # 分组收益
        if "group_returns" in evaluation_report:
            self.plot_group_returns(
                evaluation_report["group_returns"],
                save_name="3_group_returns.png"
            )

        # 回测结果
        if backtest_results is not None:
            self.plot_backtest_results(
                backtest_results["portfolio_history"],
                backtest_results["metrics"],
                save_name="4_backtest_results.png"
            )

        # 特征重要性
        if feature_importance is not None:
            self.plot_feature_importance(
                feature_importance,
                save_name="5_feature_importance.png"
            )

        # 模型对比
        if model_comparison is not None:
            self.plot_model_comparison(
                model_comparison,
                save_name="6_model_comparison.png"
            )

        logger.info(f"Full report generated in {self.report_dir}")
