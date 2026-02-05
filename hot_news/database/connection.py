# =====================================================
# Hot News Module - Database Connection
# =====================================================

import pymysql
from typing import Optional
from contextlib import contextmanager, asynccontextmanager
from ..config.settings import get_settings


class DatabaseConnection:
    """数据库连接管理"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.settings = get_settings()

    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = pymysql.connect(**self.settings._db_config.connection_kwargs)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """获取游标"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()


def get_db() -> DatabaseConnection:
    """获取数据库连接实例"""
    return DatabaseConnection()


@asynccontextmanager
async def get_async_db_connection():
    """
    获取异步数据库连接

    使用 aiomysql 实现异步数据库操作
    """
    import aiomysql

    settings = get_settings()
    db_config = settings._db_config

    conn = await aiomysql.connect(
        host=db_config.host,
        port=db_config.port,
        user=db_config.user,
        password=db_config.password,
        db=db_config.database,
        charset="utf8mb4",
        autocommit=False,
    )
    try:
        yield conn
    finally:
        conn.close()
