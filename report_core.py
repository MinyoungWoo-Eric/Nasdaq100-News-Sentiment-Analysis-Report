# report_core.py
import os
import uuid
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import threading

from report_utils import (
    retry_on_azure_error,
    throttle,
    get_unique_chroma_dir,
    clean_chroma_temp_dirs,
    get_llm,
    get_embeddings,
    parse_timestamp_to_date,
    group_comments_by_date,
    detect_sentiment_anomalies
)

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==================== 核心 RAG 工具（强制带来源链接 + 防幻觉）===================
@retry_on_azure_error(max_retries=5, delay=3, backoff=1.5)
@throttle(seconds=3.0)
def get_rag_response_with_context(
    query: str, vector_db: Chroma, system_prompt: str,
    context_str: str = "", date_filter: str = None, top_k: int = 15
) -> str:

    # 加强 system prompt：强制要求引用来源
    enhanced_system = system_prompt + "\n\n" + """
CRITICAL CITATION RULE:
When referencing any specific comment, evidence, or event in your analysis,
you MUST include the corresponding '(Sentiment:[SCORE], Source: URL)' exactly as it appears below the comment.
Do not omit, paraphrase, or hide the Source line. This ensures full transparency and traceability.
Additionally, do not use "$" mark in your answer and quote to avoid Markdown problems. Answer in a structured bullet format.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("human", """
You are a senior sentiment analyst at a $25bn hedge fund.
Use only the provided comments below for analysis. 
You can use industry knowledge and investment theory to support your statement.
Do not introduce external news and evidence.
Answer in clear, professional English.

# Existing Context (reference if needed)
{context_str}

# Relevant Articles/Comments (top {top_k}) — Each has a Source URL at the end
{context}

# Task
{query}
""")
    ])

    chain = (
        {"context": lambda x: retrieve_relevant_comments(vector_db, x["query"], x.get("date_filter"), x["top_k"]),
         "context_str": RunnablePassthrough(), "query": RunnablePassthrough(),
         "top_k": RunnablePassthrough(), "date_filter": RunnablePassthrough()}
        | prompt | get_llm() | StrOutputParser()
    )

    try:
        return chain.invoke({
            "query": query,
            "context_str": context_str,
            "top_k": top_k,
            "date_filter": date_filter
        }).strip()
    except Exception as e:
        print(f"RAG failed: {e}")
        return "[Analysis unavailable]"


@retry_on_azure_error(max_retries=5, delay=3, backoff=1.5)
@throttle(seconds=2.0)
def build_vector_db(social_data: List[Dict], prefix: str = "vec") -> tuple[Chroma, str]:
    dir_path = get_unique_chroma_dir(prefix)
    docs = []
    for it in social_data:
        text = it.get("post") or it.get("full_text") or ""
        if len(text) < 10:
            continue
        date_str = it.get("date_str") or parse_timestamp_to_date(it.get("time_published"))
        link = it.get("link") or it.get("url") or ""  # 支持 link 或 url
        docs.append(Document(
            page_content=text,
            metadata={
                "sentiment_score": float(it.get("sentiment", 0)),
                "date_str": date_str,
                "doc_id": str(uuid.uuid4()),
                "link": link
            }
        ))
    if not docs:
        raise ValueError("No valid documents")
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=dir_path)
    return db, dir_path


def retrieve_relevant_comments(vector_db: Chroma, query: str, date_filter=None, top_k=15) -> str:
    retriever = vector_db.as_retriever(search_kwargs={
        "k": top_k,
        "filter": {"date_str": date_filter} if date_filter else None
    })
    docs = retriever.invoke(query)
    lines = []
    for d in docs:
        score = f"[{d.metadata['sentiment_score']:+.4f}]"
        content = d.page_content.strip()
        link = d.metadata.get("link", "").strip()

        lines.append(f"{score} {content}")
        if link:
            lines.append(f"Source: {link}")
        else:
            lines.append("Source: (no link available)")
        lines.append("")  # 空行分隔，提升可读性
    return "\n".join(lines).strip()


# ==================== 主报告生成器（最终专业版）===================
def generate_report_sections(
    ticker: str,
    fundamentals: dict,
    social_data: list,
    period: str = "Recent 30 days",
    chart_path: str = None,
    clean_temp_after: bool = True
) -> str:

    if not social_data:
        return f"# {ticker} — No Sentiment Data Available"

    # ============ 统计 ============
    sentiments = [it["sentiment"] for it in social_data]
    total = len(sentiments)
    avg_sent = sum(sentiments) / total
    strongly_pos = sum(1 for s in sentiments if s > 0.20)
    strongly_pos_ratio = strongly_pos / total * 100

    anomalies = detect_sentiment_anomalies(social_data, threshold=0.09)
    surge_cnt = sum(1 for a in anomalies if a["type"] == "surge")
    plunge_cnt = len(anomalies) - surge_cnt
    anomaly_dates = ", ".join(a["date"] for a in anomalies) if anomalies else "None"

    stats_table = f"""
## Social Sentiment Snapshot — {ticker}

| Metric                        | Value                                    | Notes                              |
|-------------------------------|------------------------------------------|------------------------------------|
| Data Period                   | **{period}**                             |                                    |
| Total Articles Analyzed       | **{total:,}**                            | Real-time news & commentary        |
| Overall Score                 | **{avg_sent:+.2f}**                      | Range: -0.35 to +0.35               |
| Strongly Positive (> +0.20)   | **{strongly_pos:,}** ({strongly_pos_ratio:.1f}%) | High conviction zone               |
| Total Anomaly Days            | **{len(anomalies)}**                     | Surge: {surge_cnt} │ Plunge: {plunge_cnt} |
| Key Anomaly Dates             | {anomaly_dates}                          | Major sentiment shifts             |
"""

    base_context = f"""
Ticker: {ticker} | Period: {period} | Articles: {total:,}
Avg sentiment: {avg_sent:+.4f} | Strong positive ratio: {strongly_pos_ratio:.1f}%
Anomalies: {len(anomalies)} (Surge: {surge_cnt}, Plunge: {plunge_cnt})
"""

    print(f"Building vector DB for {ticker}...")
    vector_db, chroma_dir = build_vector_db(social_data, prefix=ticker)

    # ============ 1. 异动分析 ============
    anomaly_section = ""
    if anomalies:
        anomaly_section = f"## 1. Sentiment Anomaly Drivers\n### Overview\n- Total: {len(anomalies)} (Surge: {surge_cnt} | Plunge: {plunge_cnt})\n- Dates: {anomaly_dates}\n\n### Root Cause Analysis\n"
        for a in anomalies:
            cause = get_rag_response_with_context(
                query=f"Analyze the root cause of the TICKER {ticker} sentiment {'surge' if a['type']=='surge' else 'plunge'} on {a['date']}. "
                      f"Identify key events and explain how specific article content triggered investor emotion. "
                      f"Combine market and industry context where relevant.",
                vector_db=vector_db,
                system_prompt="You are a senior social sentiment analyst at a tier-1 global hedge fund.",
                context_str=base_context,
                date_filter=a["date"],
                top_k=30
            )
            label = "Surge" if a["type"] == "surge" else "Plunge"
            anomaly_section += f"#### {a['date']} — {label} ({a['sent_change']:+.4f})\n{cause}\n\n"
    else:
        anomaly_section = "## 1. Sentiment Anomaly Drivers\nNo significant anomalies detected in the period.\n"

    # ============ Appendix: 每日事件 ============
    all_dates = sorted({
        it.get("date_str") or parse_timestamp_to_date(it.get("time_published"))
        for it in social_data if it.get("date_str") != "unknown_date"
    })

    daily_events = []
    for date_str in all_dates:
        summary = get_rag_response_with_context(
            query=f"Summarize the 2–3 most trade-relevant discussion topics about TICKER {ticker} on {date_str}.",
            vector_db=vector_db,
            system_prompt="You are a senior social sentiment analyst at a tier-1 global hedge fund.",
            context_str=base_context,
            date_filter=date_str,
            top_k=20
        )
        daily_events.append(f"### {date_str}\n{summary}\n")
    
    appendix_daily = "\n".join(daily_events) if daily_events else "No notable daily concentration."

    # ============ 2. 多空论战 ============
    bull_bear = get_rag_response_with_context(
        query=f"Identify and refine the top 3 bullish and top 3 bearish arguments for TICKER {ticker} most relevant to near-term price action. "
              f"Then clearly state which narrative currently dominates for TICKER {ticker}.",
        vector_db=vector_db,
        system_prompt="You are a senior social sentiment analyst at a tier-1 global hedge fund.",
        context_str=f"{base_context}\n## Anomalies\n{anomaly_section}\n## Daily Events\n{appendix_daily}",
        top_k=40
    )

    # ============ 3. 短期价格推演 ============
    price_outlook = get_rag_response_with_context(
        query=f"Based on anomaly patterns, bull/bear balance, and recent topics of TICKER {ticker} (BUT DO NOT repeat),"
              f"Provide the following outputs:"
              f"• 1-week TICKER {ticker} stock price investment suggestions: clearly state the position (Buy/Hold/Sell) and a brief rationale\n"
              f"• Confidence level: select one option from [High / Medium / Low] and explain the reason\n"
              f"• Primary risk factor (e.g., short squeeze, exhaustion, mean reversion, catalyst fade) and explain the reason\n"
              f"Note that you can make summary and don't need to quote resources for this part. ",
        vector_db=vector_db,
        system_prompt="You are a senior social sentiment analyst at a tier-1 global hedge fund.",
        context_str=f"{base_context}\n## Bull vs Bear\n{bull_bear}\n## Anomalies\n{anomaly_section}",
        top_k=30
    )

    # ============ 最终报告 ============
    report = f"""# {ticker} · Social Sentiment Analysis Report
**Generated:** {datetime.now().strftime('%B %d, %Y · %H:%M %Z')}  
**Period:** {period} · **Articles:** {total:,}  
**Engine:** GPT-4o + Multi-staged RAG Agent

{stats_table}

{anomaly_section}

## 2. Bull vs Bear Narrative Dominance
{bull_bear}

## 3. Short-Term Price Implication
{price_outlook}

## Appendix: Daily Event Timeline
{appendix_daily}

---
Generated by Institutional-Grade Multi-staged Sentiment Analysis Agent | Sources fully traceable 
"""

    # ============ 清理 ============
    def cleanup():
        time.sleep(10)
        try:
            shutil.rmtree(chroma_dir, ignore_errors=True)
        except:
            pass
        if clean_temp_after:
            try:
                clean_chroma_temp_dirs(keep_latest=3)
            except:
                pass
    threading.Thread(target=cleanup, daemon=True).start()

    print(f"Final report generated with traceable source links: {ticker}")
    return report.strip() + "\n"
