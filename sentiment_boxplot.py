# sentiment_boxplot.py
# 2025 Ultimate Edition
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from datetime import datetime

def plot_daily_sentiment_boxplot(
    posts: List[Dict[str, Any]],
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    min_articles_per_day: int = 3
) -> go.Figure:
    """
    绘制每日情绪得分的箱线图（Box Plot），展示分布、异常值、中位数、四分位等
    完美补充趋势线图，体现情绪波动强度与一致性（机构最看重的“分歧度”指标）
    """
    if not posts:
        raise ValueError("No posts data provided for boxplot")

    df = pd.DataFrame(posts)
    df['date'] = pd.to_datetime(df['date_str'])
    df = df.sort_values('date')

    # 按天分组，过滤文章数量太少的日子（避免误导）
    daily_groups = df.groupby('date_str')
    valid_days = []
    box_data = []

    for date_str, group in daily_groups:
        sentiments = group['sentiment'].tolist()
        if len(sentiments) < min_articles_per_day:
            continue  # 跳过样本太少的日期

        valid_days.append(date_str)
        box_data.append(sentiments)

    if len(box_data) == 0:
        # 如果没有足够数据，降级显示所有天（即使少于阈值）
        valid_days = df['date_str'].unique().tolist()
        box_data = [g['sentiment'].tolist() for _, g in daily_groups]

    # 美化日期显示：从 "2025-01-15" → "Jan 15"
    def format_date_label(date_str: str) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%b %d")  # Dec 06, Jan 01, Feb 14

    # 重新构建箱线图（关键改动在这里）
    fig = go.Figure()

    for date_str, scores in zip(valid_days, box_data):
        fig.add_trace(go.Box(
            y=scores,
            name=format_date_label(date_str),     
            boxpoints='outliers',
            jitter=0.4,
            pointpos=0,
            marker=dict(
                color='#FF6B6B',
                outliercolor='#DC143C',
                line=dict(color='#DC143C', width=1)
            ),
            line=dict(color='#FF6B6B'),
            fillcolor='rgba(255,107,107,0.15)',
            hovertemplate=(
                f"<b>{format_date_label(date_str)}</b><br>"
                "Posts: " + str(len(scores)) + "<br>"
                "Sentiment: %{y:.4f}<extra></extra>"
            )
        ))

    # 美化布局
    title = f"{ticker} Daily Sentiment Distribution"

    fig.update_layout(
        title=dict(text=title, x=0, xanchor='left'),
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white",   
        height=660,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=70, r=70, t=60, b=80),

        # 关键：这里是正确设置「Dec 06」格式的完整写法
        xaxis=dict(
            tickformat="%b %d",     # Dec 06, Jan 01, Feb 14...
            tickangle=0,            # 水平显示
            tickmode="auto",        # 自动决定显示多少个日期（不会挤）
            nticks=20,              # 最多显示 20 个刻度，超了自动抽稀（正确属性！）
        ),
    )


    return fig

# ======================== 测试代码 ========================
if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta
    
    # 模拟真实帖子数据（2024年12月到2025年1月）
    random.seed(42)
    
    start = datetime(2024, 12, 1)
    end = datetime(2025, 1, 15)
    current = start
    
    posts = []
    ticker = "NVDA"
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        # 模拟每天文章数量（3~35篇不等，周末少）
        if current.weekday() >= 5:  # 周六日
            n_posts = random.randint(0, 6)
        else:
            n_posts = random.randint(6, 35)
        
        # 生成当日情绪得分：整体偏正向，但有波动和极端观点
        base_sentiment = 0.3 + 0.15 * (current - start).days / 45  # 轻微上升趋势
        for _ in range(n_posts):
            # 80% 在均值附近，20% 是极端情绪
            if random.random() < 0.8:
                sentiment = random.gauss(base_sentiment, 0.18)  # 正常分布
            else:
                sentiment = random.choice([-0.8, -0.6, 0.7, 0.85])  # 极端观点
            
            sentiment = max(min(sentiment, 1.0), -1.0)  # 限制在 [-1, 1]
            
            posts.append({
                "date_str": date_str,
                "sentiment": round(sentiment, 4),
                "title": f"Sample post on {date_str}",
                "source": "X/Twitter"
            })
        
        current += timedelta(days=1)
    
    print(f"生成测试数据：{len(posts)} 条帖子，覆盖 {len(set(p['date_str'] for p in posts))} 天")
    print("最积极一天:", max(posts, key=lambda x: x['sentiment'])['sentiment'])
    print("最负面一天:", min(posts, key=lambda x: x['sentiment'])['sentiment'])
    
    # 绘制箱线图
    fig = plot_daily_sentiment_boxplot(
        posts=posts,
        ticker=ticker,
        min_articles_per_day=3
    )
    
    # 或者直接在 Jupyter/VSCode 中显示
    fig.show()
