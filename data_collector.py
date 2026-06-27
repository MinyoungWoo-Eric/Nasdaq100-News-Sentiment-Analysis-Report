# data_collector.py
import yfinance as yf
import requests
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

def get_alpha_vantage_key() -> str:
    return st.secrets["ALPHA_VANTAGE_API_KEY"]

def get_fundamental_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'Company Name': info.get('longName', 'N/A'),
        'Sector':       info.get('sector', 'N/A'),
        'Market Cap':   info.get('marketCap', 'N/A'),
        'PE Ratio':     info.get('trailingPE', 'N/A'),
        'EPS':          info.get('trailingEps', 'N/A'),
        'Revenue':      info.get('totalRevenue', 'N/A'),
    }

def smooth_curve(x, y, window_size=3):
    if len(x) <= window_size:
        return x, y, y, y
    y_series = pd.Series(y)
    y_mean = y_series.rolling(window=window_size, min_periods=1).mean().values
    y_min = y_series.rolling(window=window_size, min_periods=1).min().values
    y_max = y_series.rolling(window=window_size, min_periods=1).max().values
    return x, y_mean, y_min, y_max
    
# ======================== Core Functions: Supports Date Range + Daily Limit + Save URL ========================
def collect_social_data(
    ticker: str,
    daily_limit: int = 30,
    start_date: str = None,    # "2025-01-01"
    end_date: str = None       # "2025-12-08"
) -> dict:
    base_url = "https://www.alphavantage.co/query"
    api_key = get_alpha_vantage_key()
    all_posts = []
    seen_titles = set()
    daily_counter = defaultdict(int)

    # 日期处理（保持原逻辑）
    if end_date is None:
        end_dt = datetime.now(timezone.utc).date()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if start_date is None:
        start_dt = end_dt - timedelta(days=180)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Construct an 8-day interval
    intervals = []
    current = start_dt
    while current <= end_dt:
        interval_start = current
        interval_end = min(current + timedelta(days=7), end_dt)
        intervals.append((
            interval_start.strftime("%Y%m%dT0000"),
            interval_end.strftime("%Y%m%dT2359")
        ))
        current = interval_end + timedelta(days=1)

    print(f"为 {ticker} Capture sentiment data：{start_date or 'auto'} to {end_date or 'today'}，Maximum of {daily_limit} items per day")

    for i, (t_from, t_to) in enumerate(intervals): 

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "time_from": t_from,
            "time_to": t_to,
            "limit": 1000,
            "sort": "LATEST",
            "apikey": api_key
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json().get("feed", [])

            for item in data:
                title = item.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                full_text = (title + " " + item.get("summary", "")).strip()
                if len(full_text) < 30:
                    continue

                time_str = item.get("time_published", "")
                try:
                    pub_time = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                except:
                    continue

                date_key = pub_time.strftime("%Y-%m-%d")
                if daily_counter[date_key] >= daily_limit:
                    continue

                score = float(item.get("overall_sentiment_score", 0))

                # ========== Key Addition: Preserve Original News URL ==========
                link_url = item.get("url", "")  

                all_posts.append({
                    "post": full_text,
                    "sentiment": round(score, 4),
                    "label": item.get("overall_sentiment_label", "Neutral").upper(),
                    "source": "Alpha Vantage",
                    "time_published": pub_time,
                    "date_str": date_key,
                    "link": link_url  # ←←← Key field. This is what report_core reads.
                })
                daily_counter[date_key] += 1

            print(f"  {t_from[:8]} ~ {t_to[:8]} → 已收集 {len(all_posts)} 条")
            time.sleep(11)

        except Exception as e:
            print(f"  请求失败: {e}")
            time.sleep(15)

    all_posts.sort(key=lambda x: x["time_published"], reverse=True)

    # ======================== Generate a trend chart ========================
    img_path = None
    html_path = None
    fig = None
    overall_avg_sentiment = None

    if len(all_posts) >= 20:
        df = pd.DataFrame(all_posts)
        daily = df.groupby("date_str")["sentiment"].median().reset_index()
        daily = daily.sort_values("date_str")

        overall_avg_sentiment = daily["sentiment"].mean()

        x_smooth, y_mean, y_min, y_max = smooth_curve(daily["date_str"].tolist(), daily["sentiment"].tolist(), window_size=3)

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x_smooth, y=y_min, mode='lines',
                                 line=dict(color='#FFA07A', width=2, dash='dash'), name='Lower Boundary', opacity=0.7))
        fig.add_trace(go.Scatter(x=x_smooth, y=y_mean, mode='lines+markers',
                                 line=dict(color='#FF6B6B', width=3, shape='spline', smoothing=1.3), name='Rolling Score'))
        fig.add_trace(go.Scatter(x=x_smooth, y=y_max, mode='lines',
                                 line=dict(color='#DC143C', width=2, dash='dash'), name='Upper Boundary', opacity=0.7))
        fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["sentiment"], mode='markers',
                                 marker=dict(size=8, color='#FF8C00', symbol='circle-open', line=dict(color='#FF8C00', width=1.5)),
                                 name='Daily Score'))
        fig.add_trace(go.Bar(x=daily["date_str"], y=df.groupby("date_str").size(),
                             name='Article Count', yaxis='y2', opacity=0.25, marker_color='#4ECDC4'))
        fig.add_hline(y=overall_avg_sentiment, line_dash="dash", line_color="#2E8B57", line_width=2,
                      annotation_text=f"  Avg: {overall_avg_sentiment:.4f}", annotation_position="top right")

        fig.update_layout(
            title=f"{ticker} Social Sentiment Trend ({start_date or 'Auto'} ~ {end_date or 'Today'})",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis2=dict(title="Articles", overlaying="y", side="right", showgrid=False),
            template="plotly_white",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    return {
        "posts": all_posts,
        "total": len(all_posts),
        "period_start": start_dt.strftime("%Y-%m-%d"),
        "period_end": end_dt.strftime("%Y-%m-%d"),
        "trend_chart": img_path,
        "interactive_chart": html_path,
        "fig": fig,
        "avg_sentiment": round(overall_avg_sentiment, 4) if overall_avg_sentiment else None
    }

# ======================== Local Test ========================
if __name__ == "__main__":
    result = collect_social_data("NVDA", daily_limit=30, start_date="2025-06-01", end_date="2025-12-15")
    print(f"Successfully captured：{result['total']} comments，Average sentiment is {result['avg_sentiment']}")
    if result["fig"]:
        result["fig"].show()
