# =====================================================
# SentimentProcessor - Database Module
# 数据库模块
# =====================================================

from .connection import get_connection, execute_query, execute_many
from .repository import (
    UnifiedContentRepo,
    UnifiedCommentRepo,
    ProcessedContentRepo,
    ProcessedCommentRepo,
)

__all__ = [
    "get_connection",
    "execute_query",
    "execute_many",
    "UnifiedContentRepo",
    "UnifiedCommentRepo",
    "ProcessedContentRepo",
    "ProcessedCommentRepo",
]
