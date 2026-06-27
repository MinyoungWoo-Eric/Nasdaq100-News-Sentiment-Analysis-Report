# stock_basic_data.py
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import talib
import warnings
import requests
warnings.filterwarnings("ignore")
# ======================== Real-time Nasdaq-100 list (Ticker list is only captured once at startup) ========================
def _fetch_nasdaq100_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        df = tables[4]
        tickers = df["Ticker"].str.replace(".", "-").tolist()
        return sorted(tickers)
    except:
        return [
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","GOOG","TSLA","LLY",
            "JPM","UNH","XOM","V","MA","PG","JNJ","HD","COST","MRK","ABBV","CRM",
            "NFLX","BAC","AMD","CVX","KO","ADBE","PEP","TMO","LIN","WMT","ACN",
            "CSCO","MCD","ABT","TXN","QCOM","INTU","AMGN","VZ","PFE","IBM","CMCSA",
            "DIS","NOW","RTX","SPGI","UNP","ISRG","GE","CAT","BKNG","UBER","GS",
            "NEE","PM","MS","LOW","BLK","HON","SYK","ELV","TJX","VRTX","BSX","LRCX",
            "REGN","ETN","PLD","MDT","MU","PANW","ADP","KLAC","LMT","CB","ADI","DE",
            "MMC","ANET","SCHW","FI","BX","MDLZ","TMUS","AMT","SO","BMY","MO","GILD",
            "CL","ICE","CME","DUK","ZTS","SHW","TT","MCO","CVS","BN","EOG","ITW",
            "FCX","TGT","BDX","CSX","HCA","EMR","FDX","NOC"
        ]

NASDAQ100_TICKERS = _fetch_nasdaq100_tickers()

STOCK_TICKERS = NASDAQ100_TICKERS
STOCK_FULL_NAMES = {}  

def get_company_name(ticker: str) -> str:
    "Retrieve the company's full name in real time"
    if ticker in STOCK_FULL_NAMES:
        return STOCK_FULL_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName", ticker)
        STOCK_FULL_NAMES[ticker] = name
        return name
    except:
        return ticker

TIME_PERIODS = {
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y"
}

# ========================= indicators, chart functions, etc. ========================

def get_stock_price_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
    hist = hist.dropna()
    return hist

def get_stock_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    price_data = get_stock_price_data(ticker, period="1y")
    
    tech_indicators = {}
    if not price_data.empty and len(price_data) >= 60:
        tech_indicators["5D Change (%)"] = round(((price_data["Close"].iloc[-1] / price_data["Close"].iloc[-5]) - 1) * 100, 2)
        tech_indicators["60D Change (%)"] = round(((price_data["Close"].iloc[-1] / price_data["Close"].iloc[-60]) - 1) * 100, 2)
        
        close_prices = price_data["Close"].values
        rsi = talib.RSI(close_prices, timeperiod=14)
        tech_indicators["RSI (14)"] = round(rsi[-1], 2) if not np.isnan(rsi[-1]) else "N/A"
        
        high = price_data["High"].values
        low = price_data["Low"].values
        atr = talib.ATR(high, low, close_prices, timeperiod=14)
        tech_indicators["ATR (14)"] = round(atr[-1], 2) if not np.isnan(atr[-1]) else "N/A"
    
    # ===== Securely obtain info =====
    info = {}
    try:
        info = stock.info or {}  # If info is None, simply empty the dictionary.
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}  # Any anomalies will be covered by a zero-sum game.
    
    fund_indicators = {
        "Gross Margin (%)": round(info.get("grossMargins", 0) * 100, 2) if info.get("grossMargins") is not None else "N/A",
        "ROE (%)": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") is not None else "N/A",
        "Forward PE": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") is not None else "N/A",
        "PB Ratio": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") is not None else "N/A",
        "PS Ratio": round(info.get("priceToSalesTrailing12Months", 0), 2) if info.get("priceToSalesTrailing12Months") is not None else "N/A",
    }
    
    all_indicators = {**tech_indicators, **fund_indicators}
    
    # Securely obtain company name and sector
    all_indicators["Company Name"] = STOCK_FULL_NAMES.get(ticker, ticker)
    all_indicators["Sector"] = info.get("sector", "N/A") 
    
    return all_indicators

# ======================== 图表绘制函数（无修改，保留原有） ========================
def plot_ths_style_chart(ticker, df_price, period_label):
    "Candlestick chart + volume combination chart"
    if df_price.empty:
        return None, None
    
    # K线图
    fig_kline = go.Figure(data=[go.Candlestick(
        x=df_price["Date"],
        open=df_price["Open"],
        high=df_price["High"],
        low=df_price["Low"],
        close=df_price["Close"],
        name="Price",
        increasing_line_color="#008000",  #FF4500
        decreasing_line_color="#FF4500",  #008000
        showlegend=False
    )])
    
    # Moving Average
    df_price["MA5"] = df_price["Close"].rolling(window=5).mean()
    df_price["MA10"] = df_price["Close"].rolling(window=10).mean()
    df_price["MA20"] = df_price["Close"].rolling(window=20).mean()
    df_price["MA60"] = df_price["Close"].rolling(window=60).mean()
    
    fig_kline.add_trace(go.Scatter(
        x=df_price["Date"],
        y=df_price["MA5"],
        mode="lines",
        name="MA5",
        line=dict(color="#FFD700", width=1.2),
        showlegend=True
    ))
    fig_kline.add_trace(go.Scatter(
        x=df_price["Date"],
        y=df_price["MA10"],
        mode="lines",
        name="MA10",
        line=dict(color="#FFA500", width=1.2),
        showlegend=True
    ))
    fig_kline.add_trace(go.Scatter(
        x=df_price["Date"],
        y=df_price["MA20"],
        mode="lines",
        name="MA20",
        line=dict(color="#00BFFF", width=1.2),
        showlegend=True
    ))
    fig_kline.add_trace(go.Scatter(
        x=df_price["Date"],
        y=df_price["MA60"],
        mode="lines",
        name="MA60",
        line=dict(color="#9370DB", width=1.2),
        showlegend=True
    ))
    
    # Candlestick Chart Style
    fig_kline.update_layout(
        title=f"{ticker} Price Chart ({period_label})",
        title_font=dict(size=16, weight="bold", family="Arial"),
        xaxis_title="",
        yaxis_title="Price (USD)",
        template="plotly_white",
        width=1200,
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="right",
            x=1.0,
            font=dict(size=14),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    fig_kline.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor="#E6E6E6",
        tickfont=dict(size=10)
    )
    fig_kline.update_yaxes(
        showgrid=True,
        gridcolor="#E6E6E6",
        tickfont=dict(size=10)
    )
    
    # Trading Volume Chart
    fig_volume = go.Figure() #FF4500 #008000
    colors = ["#008000" if row["Close"] >= row["Open"] else "#FF4500" for _, row in df_price.iterrows()]
    fig_volume.add_trace(go.Bar(
        x=df_price["Date"],
        y=df_price["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.8,
        showlegend=False
    ))
    
    fig_volume.update_layout(
        title=f"{ticker} Volume Chart ({period_label})",
        title_font=dict(size=14, weight="bold", family="Arial"),
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white",
        width=1200,
        height=200,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=30, b=40),
        font=dict(size=10)
    )
    fig_volume.update_xaxes(
        showgrid=True,
        gridcolor="#E6E6E6",
        tickfont=dict(size=10)
    )
    fig_volume.update_yaxes(
        showgrid=True,
        gridcolor="#E6E6E6",
        tickfont=dict(size=10),
        tickformat=",.0s"
    )
    
    return fig_kline, fig_volume


def plot_sentiment_price_correlation(ticker, price_data, social_data, period_name):
    if len(social_data) < 10:
        return None

    df_posts = pd.DataFrame(social_data)
    df_posts["date"] = pd.to_datetime(df_posts["date_str"])
    
    # 1. Calculate the daily raw sentiment score mean (before smoothing)
    daily_raw_sent = df_posts.groupby("date")["sentiment"].mean().reset_index()
    daily_raw_sent.rename(columns={"sentiment": "raw_sentiment"}, inplace=True)
    
    # 2. Calculate the daily smoothed sentiment score mean (if the smooth_sentiment field exists).
    if "smooth_sentiment" in df_posts.columns:
        daily_smooth_sent = df_posts.groupby("date")["smooth_sentiment"].mean().reset_index()
        daily_smooth_sent.rename(columns={"smooth_sentiment": "smooth_sentiment"}, inplace=True)
        # Combine original + smoothed scores
        daily_sent = daily_raw_sent.merge(daily_smooth_sent, on="date", how="inner")
    else:
        # When there is no smoothed fraction, only the original fraction is retained.
        daily_sent = daily_raw_sent
        daily_sent["smooth_sentiment"] = daily_sent["raw_sentiment"]

    # Merging price and sentiment data
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    merged = price_data[["Date", "Close"]].merge(daily_sent, left_on="Date", right_on="date", how="left")
    merged = merged.dropna()  # Clearing Away Emotional Days

    if len(merged) < 10:
        return None

    fig = go.Figure()

    # 3. Draw the stock price line (main Y-axis)
    fig.add_trace(go.Scatter(
        x=merged["Date"],
        y=merged["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#1f77b4", width=3),
        yaxis="y"
    ))

    # 4. Plot the smoothed sentiment score (secondary Y-axis, solid circle, green).
    fig.add_trace(go.Scatter(
        x=merged["Date"],
        y=merged["smooth_sentiment"],
        mode="markers+lines", 
        name="Smoothed Sentiment",
        marker=dict(size=10, color="#00FF88", opacity=0.8),
        line=dict(color="#00FF88", width=1),
        yaxis="y2"
    ))

   # 5. Plot the original sentiment score (secondary Y-axis, hollow circle, orange, labeled next to the smoothed score).
    fig.add_trace(go.Scatter(
        x=merged["Date"],
        y=merged["raw_sentiment"],
        mode="markers",
        name="Raw Sentiment (Pre-smoothing)",
        marker=dict(
            size=8, 
            color="#FF8C00",  
            opacity=0.9,
            symbol="circle-open", 
            line=dict(color="#FF8C00", width=1)
        ),
        yaxis="y2",
        # Hover hints to supplement raw score values
        hovertemplate="Date: %{x}<br>Raw Sentiment: %{y:.4f}<extra></extra>"
    ))

    # 6. Layout Optimization
    fig.update_layout(
        title=f"{ticker} — Price vs Sentiment Correlation ({period_name})",
        xaxis_title="Date",
        yaxis=dict(
            title="Price (USD)",
            side="left",
            color="#1f77b4"
        ),
        yaxis2=dict(
            title="Sentiment Score",
            side="right",
            overlaying="y",
            range=[-1.1, 1.1],  # Fixed sentiment score range (-1 to 1)
            color="#00FF88"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
