# report_utils.py
import os
import uuid
import shutil 
import time
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from collections import defaultdict
from functools import wraps
import pandas as pd
import numpy as np

# ============ Fix Azure OpenAI proxy bug ============
import warnings
warnings.filterwarnings("ignore")

import openai
from openai._base_client import SyncHttpxClientWrapper, AsyncHttpxClientWrapper
from openai import InternalServerError, APIError, RateLimitError

class FixedSyncClient(SyncHttpxClientWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

class FixedAsyncClient(AsyncHttpxClientWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

openai._base_client.SyncHttpxClientWrapper  = FixedSyncClient
openai._base_client.AsyncHttpxClientWrapper = FixedAsyncClient

# ============ 核心优化：重试+节流装饰器 ============
def retry_on_azure_error(max_retries: int = 5, delay: float = 3.0, backoff: float = 1.5):
    """
    装饰器：Azure OpenAI调用失败时自动重试
    :param max_retries: 最大重试次数
    :param delay: 初始延迟（秒）
    :param backoff: 延迟倍数（每次重试延迟*backoff）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (InternalServerError, APIError, RateLimitError) as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ Max retries ({max_retries}) reached for {func.__name__}: {e}")
                        raise
                    print(f"⚠️ Azure API error (retry {retries}/{max_retries}): {e}. Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    print(f"❌ Non-retryable error in {func.__name__}: {e}")
                    raise
            return None
        return wrapper
    return decorator

def throttle(seconds: float = 1.0):
    """
    装饰器：限制函数调用频率（每次调用间隔至少seconds秒）
    :param seconds: 最小调用间隔（秒）
    """
    def decorator(func):
        last_called = 0.0
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_called
            elapsed = time.time() - last_called
            if elapsed < seconds:
                sleep_time = seconds - elapsed
                print(f"⏳ Throttling {func.__name__}, sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            last_called = time.time()
            return result
        return wrapper
    return decorator

# ============ 核心优化：统一Chroma临时目录管理 ============
CHROMA_ROOT_DIR = "./chroma_temp_root"
os.makedirs(CHROMA_ROOT_DIR, exist_ok=True)

import tempfile

def get_unique_chroma_dir(prefix: str = "vector_db") -> str:
    """
    在系统临时目录中创建唯一的Chroma目录
    Streamlit Cloud 文件系统临时且只读，重启后自动清理，无需手动管理
    """
    # 使用系统临时目录，每次运行独立，避免冲突
    temp_root = tempfile.mkdtemp(prefix="chroma_")
    unique_id = str(uuid.uuid4())[:8]
    unique_dir = os.path.join(temp_root, f"{prefix}_{unique_id}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def clean_chroma_temp_dirs(keep_latest: int = 0):
    """
    清理Chroma临时目录（可选保留最新的N个）
    :param keep_latest: 保留最新的目录数量，0表示全部清理
    """
    if not os.path.exists(CHROMA_ROOT_DIR):
        return
    
    dirs = []
    for item in os.listdir(CHROMA_ROOT_DIR):
        item_path = os.path.join(CHROMA_ROOT_DIR, item)
        if os.path.isdir(item_path):
            create_time = os.path.getctime(item_path)
            dirs.append((item_path, create_time))
    
    dirs.sort(key=lambda x: x[1], reverse=True)
    to_delete = dirs[keep_latest:] if keep_latest > 0 else dirs
    
    for dir_path, _ in to_delete:
        try:
            shutil.rmtree(dir_path)
            print(f"🧹 Cleaned Chroma temp dir: {dir_path}")
        except Exception as e:
            print(f"❌ Failed to clean {dir_path}: {e}")

# ============ 初始化LLM和嵌入模型（懒加载） ============
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ============ Azure OpenAI Config（通过 Secrets 安全读取） ============
def get_azure_config():
    """从 st.secrets 安全获取 Azure 配置"""
    try:
        return {
            "api_key": st.secrets["AZURE_OPENAI_API_KEY"],
            "endpoint": st.secrets["AZURE_OPENAI_ENDPOINT"],
            "api_version": st.secrets.get("OPENAI_API_VERSION", "2023-05-15"),
            "chat_deployment": st.secrets.get("CHAT_DEPLOYMENT", "gpt-4o"),
            "embedding_deployment": st.secrets.get("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        }
    except Exception as e:
        st.error("❌ 未检测到 Azure OpenAI 配置！请在 Streamlit Secrets 中设置相关密钥。")
        raise e

# ============ 初始化LLM和嵌入模型（懒加载 + Secrets） ============
_llm: Optional[AzureChatOpenAI] = None
_embeddings: Optional[AzureOpenAIEmbeddings] = None

def get_llm():
    global _llm
    if _llm is None:
        config = get_azure_config()
        _llm = AzureChatOpenAI(
            azure_endpoint=config["endpoint"],
            azure_deployment=config["chat_deployment"],
            api_version=config["api_version"],
            api_key=config["api_key"],
            temperature=0.7,
            max_tokens=4000,
            timeout=180,
            max_retries=3,
        )
    return _llm

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        config = get_azure_config()
        _embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config["endpoint"],
            azure_deployment=config["embedding_deployment"],
            api_version=config["api_version"],
            api_key=config["api_key"],
            request_timeout=60,
            max_retries=3,
        )
    return _embeddings

# ============ 日期处理+情感异动检测工具函数 ============
def parse_timestamp_to_date(timestamp_input) -> str:
    """兼容datetime对象/字符串的时间解析"""
    if not timestamp_input:
        return "unknown_date"
    
    # 如果是datetime对象，直接格式化
    if isinstance(timestamp_input, datetime):
        return timestamp_input.strftime("%Y-%m-%d")
    
    # 如果是字符串，尝试解析
    try:
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%dT%H%M%S"]:
            return datetime.strptime(str(timestamp_input)[:10], fmt).strftime("%Y-%m-%d")
    except:
        return "unknown_date"

def group_comments_by_date(social_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按日期分组社交评论数据（适配data_collector的输出）"""
    date_groups = defaultdict(list)
    for item in social_data:
        # 优先读取date_str（data_collector已生成），无则解析time_published
        date_str = item.get("date_str") or parse_timestamp_to_date(item.get("time_published"))
        if date_str != "unknown_date":
            date_groups[date_str].append(item)
    sorted_dates = sorted(date_groups.keys())
    return {date: date_groups[date] for date in sorted_dates}

def detect_sentiment_anomalies(social_data: List[Dict[str, Any]], threshold: float = 0.2) -> List[Dict[str, Any]]:
    """检测情感分数异动（修复KeyError，适配data_collector输出）"""
    # 转换为DataFrame并确保关键列存在
    df = pd.DataFrame(social_data)
    
    # 核心修复：补充缺失的列，适配data_collector的键名
    if "time_published" in df.columns:
        df["timestamp"] = df["time_published"]  # 兼容旧逻辑，映射为timestamp
    elif "date_str" in df.columns:
        df["timestamp"] = df["date_str"]
    else:
        return []  # 无时间数据，直接返回空
    
    # 确保sentiment列存在
    if "sentiment" not in df.columns:
        return []
    
    # 解析日期
    df["date_str"] = df["timestamp"].apply(parse_timestamp_to_date)
    df = df[df["date_str"] != "unknown_date"]
    
    # 按日期聚合
    daily_sent = df.groupby("date_str")["sentiment"].agg(["median", "count"]).reset_index()
    daily_sent.columns = ["date", "avg_sentiment", "comment_count"]
    daily_sent = daily_sent.sort_values("date")
    
    # 过滤评论数过少的日期
    daily_sent = daily_sent[daily_sent["comment_count"] >= 5]
    if len(daily_sent) < 2:
        return []
    
    # 计算每日情感变化
    daily_sent["sent_change"] = daily_sent["avg_sentiment"].diff()
    daily_sent["abs_change"] = daily_sent["sent_change"].abs()
    
    # 识别异动
    anomalies = daily_sent[daily_sent["abs_change"] >= threshold].to_dict("records")
    
    # 标记异动类型
    for anomaly in anomalies:
        if anomaly["sent_change"] > 0:
            anomaly["type"] = "surge"
            anomaly["type_cn"] = "情感暴涨"
        else:
            anomaly["type"] = "plunge"
            anomaly["type_cn"] = "情感暴跌"
    
    return anomalies
