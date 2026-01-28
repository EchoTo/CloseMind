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
def load_stock_names(_config):
    """加载股票名称映射表"""
    paths = _config.get("paths", {})
    raw_data_dir = Path(paths.get("raw_data_dir", "./data_storage/raw"))
    stock_list_path = raw_data_dir / "stock_list.csv"

    if stock_list_path.exists():
        df = pd.read_csv(stock_list_path, dtype={"code": str})
        return dict(zip(df["code"], df["name"]))
    return {}


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


def plot_stock_chart(df, code, days=60, predictions=None):
    """绘制个股K线图（含预测）"""
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

    # K线图（中国市场：红涨绿跌）
    fig.add_trace(
        go.Candlestick(
            x=stock_data["date"],
            open=stock_data["open"],
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data["close"],
            name="历史K线",
            increasing_line_color="#ef232a",  # 上涨-红色
            increasing_fillcolor="#ef232a",
            decreasing_line_color="#14b143",  # 下跌-绿色
            decreasing_fillcolor="#14b143"
        ),
        row=1, col=1
    )

    # 添加预测K线（如果有）
    if predictions is not None and len(predictions) > 0:
        fig.add_trace(
            go.Candlestick(
                x=predictions["date"],
                open=predictions["open"],
                high=predictions["high"],
                low=predictions["low"],
                close=predictions["close"],
                name="预测K线",
                increasing_line_color="#ff9999",  # 预测上涨-浅红色
                increasing_fillcolor="#ffcccc",
                decreasing_line_color="#99ff99",  # 预测下跌-浅绿色
                decreasing_fillcolor="#ccffcc",
                opacity=0.7
            ),
            row=1, col=1
        )

        # 添加预测区域标记线
        last_date = stock_data["date"].iloc[-1]
        last_date_str = pd.to_datetime(last_date).strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=last_date_str, x1=last_date_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
            row=1, col=1
        )
        # 添加标注
        fig.add_annotation(
            x=last_date_str,
            y=1.02,
            yref="paper",
            text="← 历史 | 预测 →",
            showarrow=False,
            font=dict(size=10, color="gray"),
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

    # 根据实际数据跳过非交易日，让K线连续显示
    trading_dates = pd.to_datetime(stock_data["date"]).sort_values()

    # 如果有预测数据，合并日期
    if predictions is not None and len(predictions) > 0:
        pred_dates = pd.to_datetime(predictions["date"])
        trading_dates = pd.concat([trading_dates, pred_dates]).sort_values()

    # 计算需要跳过的非交易日
    all_dates = pd.date_range(start=trading_dates.min(), end=trading_dates.max(), freq='D')
    non_trading_dates = all_dates.difference(trading_dates)

    if len(non_trading_dates) > 0:
        dt_breaks = [d.strftime("%Y-%m-%d") for d in non_trading_dates]
        fig.update_xaxes(
            rangebreaks=[dict(values=dt_breaks)]
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
        ["📊 今日预测", "🔮 全部预测结果", "📈 个股分析", "🔍 回测评估", "⚡ 系统状态"]
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

    # ========== 全部预测结果 ==========
    elif page == "🔮 全部预测结果":
        st.header("🔮 全部股票预测结果")

        if not data_exists:
            st.warning("⚠️ 数据未下载，请先运行: `python main.py download`")
            return

        if not model_exists:
            st.warning("⚠️ 模型未训练，请先运行: `python main.py train`")
            return

        with st.spinner("正在加载数据和生成预测..."):
            df = load_data(config)
            if df is None:
                return
            df_features = compute_features(config, df)
            feature_cols = get_feature_columns(df_features)
            signals, error = generate_predictions(config, df_features, feature_cols)

        if error:
            st.error(error)
            return

        if signals is None or len(signals) == 0:
            st.warning("没有生成预测结果")
            return

        # 加载股票名称
        stock_names = load_stock_names(config)
        signals["股票名称"] = signals["code"].map(lambda x: stock_names.get(x, "未知"))

        # 筛选控件
        st.sidebar.subheader("筛选条件")

        # 信号类型筛选
        signal_types = st.sidebar.multiselect(
            "信号类型",
            ["strong_buy", "buy", "hold", "sell", "strong_sell"],
            default=["strong_buy", "buy"]
        )

        # 分数范围
        score_range = st.sidebar.slider(
            "预测分数范围",
            0.0, 1.0, (0.5, 1.0)
        )

        # 排序方式
        sort_by = st.sidebar.selectbox(
            "排序方式",
            ["combined_score", "confidence", "code"],
            format_func=lambda x: {"combined_score": "预测分数", "confidence": "置信度", "code": "股票代码"}[x]
        )

        sort_ascending = st.sidebar.checkbox("升序排列", value=False)

        # 显示数量
        show_n = st.sidebar.slider("显示数量", 10, 500, 100)

        # 应用筛选
        filtered = signals.copy()
        if signal_types:
            filtered = filtered[filtered["signal"].isin(signal_types)]
        filtered = filtered[
            (filtered["combined_score"] >= score_range[0]) &
            (filtered["combined_score"] <= score_range[1])
        ]
        filtered = filtered.sort_values(sort_by, ascending=sort_ascending).head(show_n)

        # 显示统计
        st.info(f"📊 筛选后共 **{len(filtered)}** 只股票 (总计 {len(signals)} 只)")

        # 信号分布
        col1, col2, col3, col4, col5 = st.columns(5)
        signal_counts = filtered["signal"].value_counts()
        with col1:
            st.metric("强烈买入", signal_counts.get("strong_buy", 0))
        with col2:
            st.metric("买入", signal_counts.get("buy", 0))
        with col3:
            st.metric("持有", signal_counts.get("hold", 0))
        with col4:
            st.metric("卖出", signal_counts.get("sell", 0))
        with col5:
            st.metric("强烈卖出", signal_counts.get("strong_sell", 0))

        # 显示表格
        st.subheader("📋 预测结果列表")

        display_cols = ["code", "股票名称", "signal", "combined_score", "confidence"]
        display_cols = [c for c in display_cols if c in filtered.columns]

        display_df = filtered[display_cols].copy()
        display_df.columns = ["代码", "名称", "信号", "预测分数", "置信度"][:len(display_cols)]

        # 格式化
        if "预测分数" in display_df.columns:
            display_df["预测分数"] = display_df["预测分数"].apply(lambda x: f"{x:.4f}")
        if "置信度" in display_df.columns:
            display_df["置信度"] = display_df["置信度"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

        # 信号颜色映射
        def color_signal(val):
            colors = {
                "strong_buy": "background-color: #28a745; color: white",
                "buy": "background-color: #5cb85c; color: white",
                "hold": "background-color: #ffc107; color: black",
                "sell": "background-color: #f0ad4e; color: white",
                "strong_sell": "background-color: #dc3545; color: white"
            }
            return colors.get(val, "")

        st.dataframe(
            display_df.style.applymap(color_signal, subset=["信号"] if "信号" in display_df.columns else []),
            use_container_width=True,
            height=500
        )

        # 下载按钮
        csv = filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载筛选结果",
            data=csv,
            file_name=f"filtered_predictions.csv",
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

        # 股票选择（显示名称+代码）
        stock_names = load_stock_names(config)
        stock_codes = sorted(df["code"].unique())
        stock_options = [f"{stock_names.get(code, '未知')} ({code})" for code in stock_codes]
        code_map = {f"{stock_names.get(code, '未知')} ({code})": code for code in stock_codes}

        selected_option = st.sidebar.selectbox("选择股票", stock_options)
        selected_code = code_map.get(selected_option, stock_codes[0] if stock_codes else None)

        # 时间范围和预测设置
        days = st.sidebar.slider("显示天数", 20, 120, 60)
        show_prediction = st.sidebar.checkbox("显示价格预测", value=True)
        forecast_days = st.sidebar.slider("预测天数", 5, 20, 10) if show_prediction else 0

        # 股票信息
        stock_data = df_features[df_features["code"] == selected_code]
        if len(stock_data) > 0:
            latest = stock_data.iloc[-1]

            # 获取模型预测信号（如果有）
            stock_signal = None
            if model_exists:
                try:
                    feature_cols = get_feature_columns(df_features)
                    signals, _ = generate_predictions(config, df_features, feature_cols)
                    if signals is not None:
                        stock_signal = signals[signals["code"] == selected_code]
                        if len(stock_signal) > 0:
                            stock_signal = stock_signal.iloc[0]
                except:
                    pass

            # 显示基本信息
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

            # 显示模型预测（如果有）
            if stock_signal is not None:
                st.subheader("🤖 模型预测信号")
                col1, col2, col3, col4 = st.columns(4)

                signal_colors = {
                    "strong_buy": "🟢 强烈买入",
                    "buy": "🟢 买入",
                    "hold": "🟡 持有",
                    "sell": "🔴 卖出",
                    "strong_sell": "🔴 强烈卖出"
                }

                with col1:
                    signal_text = signal_colors.get(stock_signal["signal"], stock_signal["signal"])
                    st.metric("预测信号", signal_text)
                with col2:
                    st.metric("预测分数", f"{stock_signal['combined_score']:.4f}")
                with col3:
                    if "confidence" in stock_signal and pd.notna(stock_signal["confidence"]):
                        st.metric("置信度", f"{stock_signal['confidence']:.2%}")
                with col4:
                    if "expected_gain" in stock_signal and pd.notna(stock_signal["expected_gain"]):
                        st.metric("预期收益", f"{stock_signal['expected_gain']:.2%}")

            # 价格预测
            predictions = None
            if show_prediction and forecast_days > 0:
                with st.spinner("正在生成价格预测..."):
                    try:
                        from models.price_predictor import PricePredictor
                        predictor = PricePredictor(config)
                        predictions = predictor.predict_price(df_features, selected_code, forecast_days)

                        if predictions is not None:
                            # 显示预测摘要
                            summary = predictor.get_prediction_summary(df_features, selected_code, forecast_days)
                            if summary:
                                st.subheader(f"📈 未来 {forecast_days} 天价格预测")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("当前价格", f"{summary['current_price']:.2f}")
                                with col2:
                                    delta_color = "normal" if summary['predicted_return'] >= 0 else "inverse"
                                    st.metric(
                                        "预测价格",
                                        f"{summary['predicted_price']:.2f}",
                                        delta=f"{summary['predicted_return']:.2f}%"
                                    )
                                with col3:
                                    st.metric("预测下界 (10%)", f"{summary['lower_bound']:.2f}")
                                with col4:
                                    st.metric("预测上界 (90%)", f"{summary['upper_bound']:.2f}")
                    except Exception as e:
                        st.warning(f"价格预测失败: {e}")
                        predictions = None

            # K线图（含预测）
            fig = plot_stock_chart(df_features, selected_code, days, predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 预测K线数据表格
            if predictions is not None and len(predictions) > 0:
                st.subheader("📋 预测K线数据")
                pred_display = predictions[["date", "open", "high", "low", "close"]].copy()
                pred_display.columns = ["日期", "开盘价", "最高价", "最低价", "收盘价"]
                pred_display["日期"] = pred_display["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(pred_display, use_container_width=True)

            # 技术指标
            st.subheader("📊 技术指标")
            col1, col2, col3 = st.columns(3)

            with col1:
                if "rsi_6" in latest:
                    st.metric("RSI(6) 相对强弱指数", f"{latest['rsi_6']:.1f}")
                if "rsi_12" in latest:
                    st.metric("RSI(12) 相对强弱指数", f"{latest['rsi_12']:.1f}")

            with col2:
                if "macd_dif" in latest:
                    st.metric("MACD DIF 差离值", f"{latest['macd_dif']:.3f}")
                if "macd_dea" in latest:
                    st.metric("MACD DEA 信号线", f"{latest['macd_dea']:.3f}")

            with col3:
                if "kdj_k" in latest:
                    st.metric("KDJ K 随机指标", f"{latest['kdj_k']:.1f}")
                if "kdj_d" in latest:
                    st.metric("KDJ D 随机指标", f"{latest['kdj_d']:.1f}")

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
