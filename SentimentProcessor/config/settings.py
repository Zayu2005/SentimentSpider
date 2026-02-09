# =====================================================
# SentimentProcessor - Settings
# 配置加载器
# =====================================================

import os
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel
from dotenv import load_dotenv


# 加载 .env 文件 (优先从项目根目录加载)
def _find_env_file() -> Path:
    """查找 .env 文件，优先级: 根目录 > MediaCrawler"""
    root_dir = Path(__file__).parent.parent.parent
    # 优先使用根目录的 .env
    root_env = root_dir / ".env"
    if root_env.exists():
        return root_env
    # 兼容旧路径
    legacy_env = root_dir / "SentimentSpider" / "MediaCrawler" / ".env"
    if legacy_env.exists():
        return legacy_env
    return root_env  # 返回根目录路径，让 load_dotenv 自动查找

env_file = _find_env_file()
load_dotenv(str(env_file))


class DBConfig(BaseModel):
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "sentiment"

    @property
    def connection_url(self) -> str:
        """获取数据库连接URL"""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset=utf8mb4"

    @property
    def connection_kwargs(self) -> dict:
        """获取pymysql连接参数"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": "utf8mb4",
        }


class ProcessorConfig(BaseModel):
    """预处理器配置"""
    # 批处理大小
    batch_size: int = 100

    # 分词配置
    use_paddle: bool = False  # 是否使用paddle模式(更准确但更慢)
    cut_all: bool = False  # 是否全模式分词

    # 停用词配置
    use_stopwords: bool = True
    custom_stopwords: List[str] = []

    # 清洗配置
    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True  # @用户
    remove_hashtags: bool = False  # #话题# (保留话题内容)
    remove_emojis: bool = True
    remove_html: bool = True
    normalize_whitespace: bool = True
    to_simplified: bool = True  # 繁体转简体

    # 网络用语规范化
    normalize_slang: bool = True


class Settings:
    """全局配置"""

    _instance: Optional["Settings"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """初始化配置"""
        self.db = DBConfig(
            host=os.getenv("MYSQL_DB_HOST", "localhost"),
            port=int(os.getenv("MYSQL_DB_PORT", 3306)),
            user=os.getenv("MYSQL_DB_USER", "root"),
            password=os.getenv("MYSQL_DB_PWD", ""),
            database=os.getenv("MYSQL_DB_NAME", "sentiment"),
        )
        self.processor = ProcessorConfig()

    def reload(self):
        """重新加载配置"""
        self._init_config()


def get_settings() -> Settings:
    """获取配置实例"""
    return Settings()
