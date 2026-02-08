# =====================================================
# Hot News Module - Crawl Log Repository
# =====================================================

from typing import List, Optional
from datetime import datetime
from ..connection import get_db


class CrawlLogRepository:
    """爬取记录数据访问层"""

    def __init__(self):
        self.db = get_db()

    def create_log(
        self, keyword_id: int, platform_code: str, status: str = "pending"
    ) -> int:
        """创建爬取记录"""
        sql = """
            INSERT INTO keyword_crawl_log 
            (keyword_id, platform_code, status, started_at)
            VALUES (%s, %s, %s, NOW())
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (keyword_id, platform_code, status))
            cursor.execute("SELECT LAST_INSERT_ID()")
            return cursor.fetchone()[0]

    def update_log(
        self,
        log_id: int,
        status: str,
        notes_count: int = 0,
        comments_count: int = 0,
        error_message: str = None,
    ) -> bool:
        """更新爬取记录"""
        sql = """
            UPDATE keyword_crawl_log 
            SET status = %s, notes_count = %s, comments_count = %s, 
                error_message = %s, completed_at = NOW()
            WHERE id = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql, (status, notes_count, comments_count, error_message, log_id)
            )
            return cursor.rowcount > 0

    def get_by_keyword(self, keyword_id: int) -> List[dict]:
        """获取关键词的爬取记录"""
        sql = "SELECT * FROM keyword_crawl_log WHERE keyword_id = %s ORDER BY created_at DESC"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (keyword_id,))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_recent(self, limit: int = 50) -> List[dict]:
        """获取最近的爬取记录"""
        sql = "SELECT * FROM keyword_crawl_log ORDER BY created_at DESC LIMIT %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_pending(self, limit: int = 100) -> List[dict]:
        """获取待处理的记录"""
        sql = "SELECT * FROM keyword_crawl_log WHERE status = 'pending' ORDER BY created_at ASC LIMIT %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]
