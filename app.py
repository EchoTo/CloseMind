#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CloseMind Web界面
基于Streamlit的本地可视化应用
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 页面配置
st.set_page_config(
    page_title="CloseMind - A股量化预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .buy-signal { color: #28a745; font-weight: bold; }
    .sell-signal { color: #dc3545; font-weight: bold; }
    .hold-signal { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """加载配置"""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=3600)
def load_data(_config):
    """加载数据"""
    from data.qlib_converter import SimpleDataLoader

    loader = SimpleDataLoader(_config)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    try:
        df = loader.load_data(start_date=start_date, end_date=end_date)
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None


@st.cache_data(ttl=3600)
def compute_features(_config, df):
    """计算特征"""
    from features import FeatureEngine

    feature_engine = FeatureEngine(_config)

    # 加载指数数据
    processed_dir = Path(_config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
    index_path = processed_dir / "index_processed.parquet"
    index_df = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)

    df_features = feature_engine.compute_all_features(df.copy(), index_df)
    return df_features


def get_feature_columns(df):
    """获取特征列"""
    exclude_cols = [
        "date", "code", "name", "industry",
        "open", "high", "low", "close", "volume", "amount",
        "turnover", "pct_change", "change", "amplitude",
        "year", "month", "day", "weekday", "quarter",
        "is_month_start", "is_month_end",
        "is_suspended", "is_limit_up", "is_limit_down", "is_abnormal",
        "return_1d", "return_5d", "return_1d_rank", "return_5d_rank",
        "label_binary_1d", "label_binary_5d"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.3]


def generate_predictions(config, df, feature_cols):
    """生成预测"""
    from models import EnsembleModel
    from strategy import SignalGenerator, PositionTracker

    # 加载模型
    model_dir = Path(config.get("paths", {}).get("model_dir", "./checkpoints"))
    ensemble_path = model_dir / "ensemble_ensemble_meta.pkl"

    if not ensemble_path.exists():
        return None, "模型未训练，请先运行训练脚本"

    ensemble = EnsembleModel(config)
    try:
        ensemble.load("ensemble")
    except Exception as e:
        return None, f"模型加载失败: {e}"

    # 获取最新数据
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()
    latest_df[feature_cols] = latest_df[feature_cols].fillna(0)

    # 预测
    predictions = ensemble.predict(latest_df, feature_cols)
    latest_df["prediction"] = predictions
    latest_df["pred_daily"] = predictions

    # 生成信号
    signal_generator = SignalGenerator(config)
    market_cols = ["date", "code", "is_limit_up", "is_limit_down", "is_suspended", "close"]
    market_data = latest_df[[c for c in market_cols if c in latest_df.columns]].copy()
    signals = signal_generator.generate_signals(latest_df, market_data)

    # 持仓跟踪
    tracker = PositionTracker(config)
    prices = market_data.set_index("code")["close"].to_dict()
    tracked = tracker.process_signals(signals, str(latest_date)[:10], prices)

    return tracked, None


def plot_stock_chart(df, code, days=60):
    """绘制个股K线图"""
    stock_data = df[df["code"] == code].tail(days).copy()

    if len(stock_data) == 0:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("价格走势", "成交量")
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=stock_data["date"],
            open=stock_data["open"],
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data["close"],
            name="K线"
        ),
        row=1, col=1
    )

    # 均线
    if "ma_5" in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data["date"], y=stock_data["ma_5"], name="MA5", line=dict(color="orange", width=1)),
            row=1, col=1
        )
    if "ma_20" in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data["date"], y=stock_data["ma_20"], name="MA20", line=dict(color="blue", width=1)),
            row=1, col=1
        )

    # 成交量
    colors = ["red" if c >= o else "green" for c, o in zip(stock_data["close"], stock_data["open"])]
    fig.add_trace(
        go.Bar(x=stock_data["date"], y=stock_data["volume"], name="成交量", marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    return fig


def plot_signal_distribution(signals):
    """绘制信号分布"""
    signal_counts = signals["signal"].value_counts()

    colors = {
        "strong_buy": "#28a745",
        "buy": "#5cb85c",
        "hold": "#ffc107",
        "sell": "#f0ad4e",
        "strong_sell": "#dc3545"
    }

    fig = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        color=signal_counts.index,
        color_discrete_map=colors,
        title="今日信号分布"
    )

    return fig


def plot_score_histogram(signals):
    """绘制分数分布"""
    fig = px.histogram(
        signals,
        x="combined_score",
        nbins=50,
        title="预测分数分布",
        labels={"combined_score": "预测分数", "count": "数量"}
    )
    fig.update_layout(template="plotly_white")
    return fig


def main():
    # 标题
    st.markdown('<p class="main-header">📈 CloseMind - A股量化预测系统</p>', unsafe_allow_html=True)

    # 加载配置
    config = load_config()

    # 侧边栏
    st.sidebar.title("⚙️ 控制面板")

    # 功能选择
    page = st.sidebar.selectbox(
        "选择功能",
        ["📊 今日预测", "📈 个股分析", "🔍 回测评估", "⚡ 系统状态"]
    )

    # 检查数据状态
    data_dir = Path(config.get("paths", {}).get("processed_data_dir", "./data_storage/processed"))
    data_exists = (data_dir / "stock_processed.parquet").exists()

    model_dir = Path(config.get("paths", {}).get("model_dir", "./checkpoints"))
    model_exists = (model_dir / "ensemble_ensemble_meta.pkl").exists()

    # ========== 今日预测 ==========
    if page == "📊 今日预测":
        st.header("📊 今日股票预测")

        if not data_exists:
            st.warning("⚠️ 数据未下载，请先运行: `python main.py download`")
            st.code("python main.py download --mode full", language="bash")
            return

        if not model_exists:
            st.warning("⚠️ 模型未训练，请先运行: `python main.py train`")
            st.code("python main.py train --evaluate", language="bash")
            return

        with st.spinner("正在加载数据和计算特征..."):
            df = load_data(config)
            if df is None:
                return

            df_features = compute_features(config, df)
            feature_cols = get_feature_columns(df_features)

        with st.spinner("正在生成预测..."):
            signals, error = generate_predictions(config, df_features, feature_cols)

        if error:
            st.error(error)
            return

        if signals is None or len(signals) == 0:
            st.warning("没有生成预测结果")
            return

        # 显示日期
        latest_date = df["date"].max()
        st.info(f"📅 预测日期: **{str(latest_date)[:10]}**")

        # 信号统计
        col1, col2, col3, col4, col5 = st.columns(5)
        signal_counts = signals["signal"].value_counts()

        with col1:
            st.metric("强烈买入", signal_counts.get("strong_buy", 0), delta="↑")
        with col2:
            st.metric("买入", signal_counts.get("buy", 0))
        with col3:
            st.metric("持有", signal_counts.get("hold", 0))
        with col4:
            st.metric("卖出", signal_counts.get("sell", 0))
        with col5:
            st.metric("强烈卖出", signal_counts.get("strong_sell", 0), delta="↓")

        # 图表
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_signal_distribution(signals), use_container_width=True)
        with col2:
            st.plotly_chart(plot_score_histogram(signals), use_container_width=True)

        # 买入建议表格
        st.subheader("🔥 买入建议 Top 30")
        buy_signals = signals[signals["signal"].isin(["strong_buy", "buy"])].copy()
        buy_signals = buy_signals.sort_values("combined_score", ascending=False).head(30)

        if len(buy_signals) > 0:
            display_cols = ["code", "signal", "combined_score", "confidence", "holding_days",
                           "current_gain", "expected_gain", "expected_holding_days"]
            display_cols = [c for c in display_cols if c in buy_signals.columns]

            # 格式化
            display_df = buy_signals[display_cols].copy()
            display_df["combined_score"] = display_df["combined_score"].apply(lambda x: f"{x:.4f}")
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

            if "current_gain" in display_df.columns:
                display_df["current_gain"] = display_df["current_gain"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "首次"
                )
            if "expected_gain" in display_df.columns:
                display_df["expected_gain"] = display_df["expected_gain"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )

            st.dataframe(display_df, use_container_width=True, height=400)

        # 卖出建议
        st.subheader("⚠️ 卖出建议 Top 20")
        sell_signals = signals[signals["signal"].isin(["sell", "strong_sell"])].copy()
        sell_signals = sell_signals.sort_values("combined_score", ascending=True).head(20)

        if len(sell_signals) > 0:
            display_cols = ["code", "signal", "combined_score", "last_buy_date", "holding_days", "current_gain"]
            display_cols = [c for c in display_cols if c in sell_signals.columns]

            display_df = sell_signals[display_cols].copy()
            if "current_gain" in display_df.columns:
                display_df["current_gain"] = display_df["current_gain"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )

            st.dataframe(display_df, use_container_width=True, height=300)

        # 下载按钮
        csv = signals.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载完整预测结果",
            data=csv,
            file_name=f"predictions_{latest_date}.csv",
            mime="text/csv"
        )

    # ========== 个股分析 ==========
    elif page == "📈 个股分析":
        st.header("📈 个股详细分析")

        if not data_exists:
            st.warning("⚠️ 数据未下载")
            return

        with st.spinner("加载数据中..."):
            df = load_data(config)
            if df is None:
                return
            df_features = compute_features(config, df)

        # 股票选择
        stock_codes = sorted(df["code"].unique())
        selected_code = st.sidebar.selectbox("选择股票代码", stock_codes)

        # 时间范围
        days = st.sidebar.slider("显示天数", 20, 120, 60)

        # 股票信息
        stock_data = df_features[df_features["code"] == selected_code]
        if len(stock_data) > 0:
            latest = stock_data.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最新价", f"{latest['close']:.2f}")
            with col2:
                change = latest.get("pct_change", 0)
                st.metric("涨跌幅", f"{change:.2f}%", delta=f"{change:.2f}%")
            with col3:
                st.metric("成交量", f"{latest['volume']/10000:.0f}万")
            with col4:
                if "turnover" in latest:
                    st.metric("换手率", f"{latest['turnover']:.2f}%")

            # K线图
            fig = plot_stock_chart(df_features, selected_code, days)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 技术指标
            st.subheader("📊 技术指标")
            col1, col2, col3 = st.columns(3)

            with col1:
                if "rsi_6" in latest:
                    st.metric("RSI(6)", f"{latest['rsi_6']:.1f}")
                if "rsi_12" in latest:
                    st.metric("RSI(12)", f"{latest['rsi_12']:.1f}")

            with col2:
                if "macd_dif" in latest:
                    st.metric("MACD DIF", f"{latest['macd_dif']:.3f}")
                if "macd_dea" in latest:
                    st.metric("MACD DEA", f"{latest['macd_dea']:.3f}")

            with col3:
                if "kdj_k" in latest:
                    st.metric("KDJ K", f"{latest['kdj_k']:.1f}")
                if "kdj_d" in latest:
                    st.metric("KDJ D", f"{latest['kdj_d']:.1f}")

    # ========== 回测评估 ==========
    elif page == "🔍 回测评估":
        st.header("🔍 模型回测评估")

        report_dir = Path(config.get("paths", {}).get("report_dir", "./reports"))

        # 查找报告图片
        report_images = list(report_dir.glob("*.png"))

        if report_images:
            st.info("📊 以下是最近的评估报告")

            for img_path in sorted(report_images)[:6]:
                st.image(str(img_path), caption=img_path.name)
        else:
            st.warning("暂无回测报告，请先运行训练和评估")
            st.code("python main.py train --evaluate", language="bash")

    # ========== 系统状态 ==========
    elif page == "⚡ 系统状态":
        st.header("⚡ 系统状态")

        # 数据状态
        st.subheader("📁 数据状态")
        col1, col2 = st.columns(2)

        with col1:
            if data_exists:
                st.success("✅ 股票数据已下载")
                data_path = data_dir / "stock_processed.parquet"
                if data_path.exists():
                    df = pd.read_parquet(data_path)
                    st.write(f"- 数据条数: {len(df):,}")
                    st.write(f"- 股票数量: {df['code'].nunique()}")
                    st.write(f"- 日期范围: {df['date'].min()} 至 {df['date'].max()}")
            else:
                st.error("❌ 股票数据未下载")

        with col2:
            if model_exists:
                st.success("✅ 模型已训练")
            else:
                st.error("❌ 模型未训练")

        # 快速命令
        st.subheader("🚀 快速命令")

        st.markdown("""
        ```bash
        # 1. 下载数据
        python main.py download --mode full

        # 2. 训练模型
        python main.py train --evaluate

        # 3. 生成预测
        python main.py predict --top-n 50

        # 4. 启动Web界面
        streamlit run app.py
        ```
        """)

        # 配置信息
        st.subheader("⚙️ 当前配置")
        st.json({
            "数据路径": config.get("paths", {}).get("data_dir"),
            "模型路径": config.get("paths", {}).get("model_dir"),
            "数据起始日期": config.get("data", {}).get("start_date"),
            "持仓数量": config.get("backtest", {}).get("portfolio", {}).get("top_n"),
        })


if __name__ == "__main__":
    main()
