# =====================================================
# Hot News Module - Repositories Package
# =====================================================

from .hot_news_repo import HotNewsRepository
from .analysis_repo import AnalysisRepository
from .keyword_repo import KeywordRepository
from .crawl_log_repo import CrawlLogRepository
from .task_log_repo import TaskLogRepository

__all__ = [
    "HotNewsRepository",
    "AnalysisRepository",
    "KeywordRepository",
    "CrawlLogRepository",
    "TaskLogRepository",
]
