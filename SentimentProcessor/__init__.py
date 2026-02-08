# =====================================================
# SentimentProcessor
# 情感分析数据预处理模块
# =====================================================
#
# 功能:
# - 文本清洗 (URL、邮箱、@提及、表情符号等)
# - 中文分词 (jieba)
# - 关键词提取 (TF-IDF / TextRank)
# - 网络用语规范化
# - 繁体转简体
#
# 使用方法:
#   from SentimentProcessor import ContentProcessor, CommentProcessor
#
#   # 处理内容
#   processor = ContentProcessor()
#   result = processor.run()
#
#   # 处理评论
#   processor = CommentProcessor()
#   result = processor.run()
#
# CLI使用:
#   python -m SentimentProcessor stats       # 查看统计信息
#   python -m SentimentProcessor content     # 处理内容
#   python -m SentimentProcessor comment     # 处理评论
#   python -m SentimentProcessor all         # 处理所有
#
# =====================================================

from .config import Settings, get_settings
from .processor import (
    TextCleaner,
    Segmenter,
    KeywordExtractor,
    ContentProcessor,
    CommentProcessor,
)
from .utils import StopwordsManager, SlangNormalizer

__version__ = "1.0.0"

__all__ = [
    # 配置
    "Settings",
    "get_settings",
    # 处理器
    "TextCleaner",
    "Segmenter",
    "KeywordExtractor",
    "ContentProcessor",
    "CommentProcessor",
    # 工具
    "StopwordsManager",
    "SlangNormalizer",
]
