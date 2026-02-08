# =====================================================
# Hot News Module - Database Package
# =====================================================

from .connection import DatabaseConnection, get_db, get_async_db_connection
from .repositories.hot_news_repo import HotNewsRepository
from .repositories.analysis_repo import AnalysisRepository
from .repositories.keyword_repo import KeywordRepository
from .repositories.crawl_log_repo import CrawlLogRepository
from .repositories.task_log_repo import TaskLogRepository

__all__ = [
    "DatabaseConnection",
    "get_db",
    "get_async_db_connection",
    "HotNewsRepository",
    "AnalysisRepository",
    "KeywordRepository",
    "CrawlLogRepository",
    "TaskLogRepository",
]
