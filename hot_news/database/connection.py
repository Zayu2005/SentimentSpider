# =====================================================
# Hot News Module - Database Connection
# =====================================================

import pymysql
from typing import Optional
from contextlib import contextmanager
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
