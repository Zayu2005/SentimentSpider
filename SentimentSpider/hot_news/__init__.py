# =====================================================
# Hot News Module - Main Package
# =====================================================

from .config import settings, get_settings
from .fetcher import HotNewsFactory, OrzAiFetcher, OrzAiClient
from .analyzer import LLMClientFactory, DomainChecker, KeywordExtractor
from .logger import logger, setup_logger, info, debug, warning, error
from .models.entities import (
    HotNewsItem,
    DomainInfo,
    DomainMatchResult,
    KeywordResult,
    LLMResponse,
    TaskResult,
    CrawlTask,
)
from .database import (
    get_db,
    HotNewsRepository,
    AnalysisRepository,
    KeywordRepository,
    CrawlLogRepository,
    TaskLogRepository,
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    # Logger
    "logger",
    "setup_logger",
    "info",
    "debug",
    "warning",
    "error",
    # Fetcher
    "HotNewsFactory",
    "OrzAiFetcher",
    "OrzAiClient",
    # Analyzer
    "LLMClientFactory",
    "DomainChecker",
    "KeywordExtractor",
    # Models
    "HotNewsItem",
    "DomainInfo",
    "DomainMatchResult",
    "KeywordResult",
    "LLMResponse",
    "TaskResult",
    "CrawlTask",
    # Database
    "get_db",
    "HotNewsRepository",
    "AnalysisRepository",
    "KeywordRepository",
    "CrawlLogRepository",
    "TaskLogRepository",
]
