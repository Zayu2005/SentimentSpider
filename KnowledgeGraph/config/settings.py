# -*- coding: utf-8 -*-
"""
知识图谱配置管理

使用 Pydantic 进行配置验证和管理
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from functools import lru_cache

# 加载 .env 文件 (优先从项目根目录加载)
try:
    from dotenv import load_dotenv

    def _find_env_file() -> Path:
        """查找 .env 文件，优先级: 根目录 > MediaCrawler"""
        root_dir = Path(__file__).parent.parent.parent
        root_env = root_dir / ".env"
        if root_env.exists():
            return root_env
        legacy_env = root_dir / "SentimentSpider" / "MediaCrawler" / ".env"
        if legacy_env.exists():
            return legacy_env
        return root_env

    env_file = _find_env_file()
    load_dotenv(str(env_file))
except ImportError:
    pass


class Neo4jConfig(BaseModel):
    """Neo4j 图数据库配置"""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j连接URI")
    user: str = Field(default="neo4j", description="Neo4j用户名")
    password: str = Field(default="", description="Neo4j密码")
    database: str = Field(default="neo4j", description="Neo4j数据库名")

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """从环境变量加载配置"""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


class DeepSeekConfig(BaseModel):
    """DeepSeek API 配置"""
    model_config = {"protected_namespaces": ()}

    api_key: str = Field(default="", description="DeepSeek API Key")
    api_base: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 基础URL"
    )
    model_name: str = Field(default="deepseek-chat", description="模型名称")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="生成温度"
    )
    max_tokens: int = Field(default=4096, ge=1, description="最大生成token数")
    timeout: float = Field(default=60.0, description="API超时时间(秒)")

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )


class KGDatabaseConfig(BaseModel):
    """MySQL 数据库配置"""
    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=3306, description="数据库端口")
    user: str = Field(default="root", description="数据库用户名")
    password: str = Field(default="", description="数据库密码")
    database: str = Field(default="sentiment", description="数据库名称")
    charset: str = Field(default="utf8mb4", description="字符集")

    @classmethod
    def from_env(cls) -> "KGDatabaseConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("MYSQL_DB_HOST", "localhost"),
            port=int(os.getenv("MYSQL_DB_PORT", "3306")),
            user=os.getenv("MYSQL_DB_USER", "root"),
            password=os.getenv("MYSQL_DB_PWD", ""),
            database=os.getenv("MYSQL_DB_NAME", "sentiment"),
        )


class ExtractionConfig(BaseModel):
    """抽取配置"""
    batch_size: int = Field(
        default=5, ge=1, description="每批发送API的内容条数"
    )
    max_content_length: int = Field(
        default=500, description="截取的最大内容长度"
    )
    rate_limit_delay: float = Field(
        default=1.0, ge=0.0, description="API调用间隔(秒)"
    )
    max_retries: int = Field(
        default=3, ge=1, description="API失败重试次数"
    )


class KGSettings(BaseModel):
    """KnowledgeGraph 全局设置"""
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    database: KGDatabaseConfig = Field(default_factory=KGDatabaseConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    base_dir: str = Field(default="", description="项目根目录")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.base_dir:
            self.base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

    @classmethod
    def from_env(cls) -> "KGSettings":
        """从环境变量加载设置"""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            deepseek=DeepSeekConfig.from_env(),
            database=KGDatabaseConfig.from_env(),
            extraction=ExtractionConfig(),
        )


@lru_cache()
def get_kg_settings() -> KGSettings:
    """获取全局设置 (单例模式)"""
    return KGSettings.from_env()
