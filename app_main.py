# app_main.py
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

# ======================== Import Module ========================
from stock_basic_data import (
    STOCK_TICKERS, STOCK_FULL_NAMES, TIME_PERIODS,
    get_stock_price_data, get_stock_fundamental_data,
    plot_ths_style_chart
)

# Reporting module (use mock if missing)
try:
    from data_collector import collect_social_data
    from report_core import generate_report_sections
    REPORT_AVAILABLE = True
except ImportError:
    def collect_social_data(ticker, count):
        return {"posts": [], "period_start": "", "period_end": "", "avg_sentiment": None, "fig": None}
    def generate_report_sections(*args, **kwargs):
        return "# Report Module Missing\nPlease add `data_collector.py` and `report_generator.py`"
    REPORT_AVAILABLE = False

# ======================== Page Configuration ========================
st.set_page_config(
    page_title="Nasdaq-100 Sentiment Intelligence",
    page_icon="Chart Increasing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("# DeepSent")  #Nasdaq-100 Social Sentiment Analysis Intelligence
st.markdown('#### *Nasdaq-100 Social Sentiment Analysis Intelligence*')
st.caption("Authors: Lechuan WANG, Minyoung WOO, Xuantao YUAN, Yijie WANG. (2025)")
st.markdown("---")

if 'selected_period' not in st.session_state:
    st.session_state.selected_period = "1 Year"

# ======================= Stock Selection ========================
selected_ticker = st.selectbox(
    "Search & Select Nasdaq-100 Stock",
    options=STOCK_TICKERS,
    index=STOCK_TICKERS.index("NVDA") if "NVDA" in STOCK_TICKERS else 0,
    format_func=lambda x: f"{x}",
    help="Type ticker (e.g. NVDA/AAPL/TSLA)",
    key="ticker_selectbox"
)

# ======================== Main Content Area: Ultimate Professional Layout ========================
st.subheader(f"{selected_ticker} — Core Fundamentals")

# 时间周期选择
col_period = st.columns([1, 6])[0]
with col_period:
    selected_period = st.selectbox(
        "Time Period",
        options=list(TIME_PERIODS.keys()),
        index=list(TIME_PERIODS.keys()).index(st.session_state.selected_period),
        key="period"
    )
    st.session_state.selected_period = selected_period

# Load price data + indicators
price_data = get_stock_price_data(selected_ticker, period=TIME_PERIODS[selected_period])
indicators = get_stock_fundamental_data(selected_ticker)

# ======================== Part Two: Candlestick Chart + Trading Volume ========================
if not price_data.empty:
    kline_fig, volume_fig = plot_ths_style_chart(selected_ticker, price_data, selected_period)
    st.plotly_chart(kline_fig, use_container_width=True, config={'displayModeBar': False})
    st.plotly_chart(volume_fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("No price data available")

# ======================== Part Three: All Core Indicators Arranged Horizontally (below the chart) ========================
st.markdown("#### Core Trading Indicators")

# 创建 5 列布局（可容纳所有指标）
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("5D Change (%)", indicators.get("5D Change (%)", "N/A"))
    st.metric("RSI (14)", indicators.get("RSI (14)", "N/A"))

with c2:
    st.metric("60D Change (%)", indicators.get("60D Change (%)", "N/A"))
    st.metric("ATR (14)", indicators.get("ATR (14)", "N/A"))

with c3:
    st.metric("Gross Margin (%)", indicators.get("Gross Margin (%)", "N/A"))
    st.metric("Forward P/E", indicators.get("Forward PE", "N/A"))

with c4:
    st.metric("ROE (%)", indicators.get("ROE (%)", "N/A"))
    st.metric("P/B Ratio", indicators.get("PB Ratio", "N/A"))

with c5:
    st.metric("P/S Ratio", indicators.get("PS Ratio", "N/A"))


# ======================== Mood Report Generation Area ========================
st.divider()
st.subheader("Sentiment Analysis Report Generation Agent")

st.markdown("##### Data Collection Settings")

col1, col2 = st.columns(2)

with col1:
    daily_limit = st.selectbox(
        "Max articles per day",
        options=[10, 20, 30, 50, 100],
        index=2,
        help="Recommended 30"
    )

with col2:
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=7))
    with col_end:
        end_date = st.date_input("End date", value=datetime.today())

# Prevent start date from being later than end date
if start_date > end_date:
    st.error("Start date cannot be later than end date")
    st.stop()

# Calculate the estimated number of entries and time
days = (end_date - start_date).days + 1
estimated_articles = min(daily_limit * days, 3000)
estimated_time = int(estimated_articles / 100 * 60)  

days = (end_date - start_date).days + 1
st.caption(f"Analysis period: **{days}** days · Totally **{estimated_articles}** articles")

if not REPORT_AVAILABLE:
    st.warning("Report module not found (data_collector.py / report_generator.py missing)")

generate_btn = st.button(
    f"Generate {selected_ticker} Sentiment Report",
    type="primary",
    use_container_width=True
)

# Cache key: Generate a unique key based on ticker + time period + daily limit
cache_key = f"report_{selected_ticker}_{start_date}_{end_date}_{daily_limit}"

if generate_btn or cache_key in st.session_state:
    if cache_key not in st.session_state:
        # First generation
        prog = st.progress(0)
        status = st.empty()

        status.text("Step 1: Collecting news from Alpha Vantage...")
        prog.progress(20)
        start_time = time.time()

        # Key point: Passing new parameters to data_collector
        result = collect_social_data(
            ticker=selected_ticker,
            daily_limit=daily_limit,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        status.text("Step 2: Generating institutional RAG report with GPT-4o...")
        prog.progress(60)

        report = generate_report_sections(
            ticker=selected_ticker,
            fundamentals=indicators,
            social_data=result["posts"],
            period=f"{result['period_start']} to {result['period_end']}",
            chart_path=result.get("trend_chart")
        )

        # Cache results
        st.session_state[cache_key] = {
            "report": report,
            "fig": result.get("fig"),
            "avg_sentiment": result.get("avg_sentiment"),
            "posts": result["posts"]
        }
        prog.progress(100)
        status.success(f"Report generated in {int(time.time()-start_time)}s")
        time.sleep(1.5)
        prog.empty()


    # Read cache and display
    cache = st.session_state[cache_key]

    st.markdown("---")

    # ==================== 1. First, fully render the Snapshot statistics table ====================
    report_lines = cache["report"].split('\n')
    snapshot_end_idx = next(
        (i for i, line in enumerate(report_lines)
         if line.startswith('## ') and i > 10),
        len(report_lines)
    )
    snapshot_part = '\n'.join(report_lines[:snapshot_end_idx])
    st.markdown(snapshot_part, unsafe_allow_html=True)

    # ==================== 2. Sentiment Trend Line Chart ====================
    if cache.get("fig"):
        st.markdown("#### Sentiment Trend Over Time")
        st.plotly_chart(
            cache["fig"],
            use_container_width=True,
            config={'displayModeBar': False},
            height=520
        )
    else:
        st.info("Sentiment trend chart not available")

    # ==================== 3. Added: Daily Sentiment Box Line Chart ===================
    try:
        from sentiment_boxplot import plot_daily_sentiment_boxplot

        if len(cache["posts"]) >= 10:  

            box_fig = plot_daily_sentiment_boxplot(
                posts=cache["posts"],
                ticker=selected_ticker,
                start_date=cache["posts"][-1]["date_str"] if cache["posts"] else None,
                end_date=cache["posts"][0]["date_str"] if cache["posts"] else None,
                min_articles_per_day=3
            )
            st.plotly_chart(box_fig, use_container_width=True, config={'displayModeBar': False})

    except ImportError:
        st.info("`sentiment_boxplot.py` not found → Daily boxplot disabled")
    except Exception as e:
        st.error(f"Boxplot rendering error: {e}")

    # ==================== 4. Remaining text of the rendering report ====================
    remaining_part = '\n'.join(report_lines[snapshot_end_idx:])
    st.markdown(remaining_part, unsafe_allow_html=True)

    # ==================== Download button, etc. ====================
    st.download_button(
        label="Download Full Report (Markdown)",
        data=cache["report"],
        file_name=f"{selected_ticker}_Sentiment_Report_{start_date}_to_{end_date}.md",
        mime="text/markdown",
        use_container_width=True
    )

    if cache.get("avg_sentiment") is not None:
        st.metric("Overall Sentiment Score", f"{cache['avg_sentiment']:+.4f}")

    if st.button("Clear Cache & Regenerate", type="secondary"):
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        st.rerun()

# ======================= Footer =========================
st.markdown("---")
st.caption("Powered by Alpha Vantage • Azure GPT-4o • yfinance • Streamlit | © 2025 DeepSent • Nasdaq-100 Sentiment Analysis Intelligence")
st.caption("© Team of Lechuan WANG, Minyoung WOO, Xuantao YUAN, Yijie WANG. (2025). All rights reserved")
