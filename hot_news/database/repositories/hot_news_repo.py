# =====================================================
# Hot News Module - Hot News Repository
# =====================================================

from typing import List, Optional
from datetime import datetime
from ..connection import get_db
from hot_news.models.entities import HotNewsItem


class HotNewsRepository:
    """热点新闻数据访问层"""

    def __init__(self):
        self.db = get_db()

    def save(self, news: HotNewsItem) -> bool:
        """保存热点新闻"""
        sql = """
            INSERT INTO hot_news (news_id, platform_code, title, url, score, description, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                title=VALUES(title),
                url=VALUES(url),
                score=VALUES(score),
                description=VALUES(description)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql,
                (
                    news.news_id,
                    news.platform_code,
                    news.title,
                    news.url,
                    news.score,
                    news.description,
                    news.created_at,
                ),
            )
            return True

    def bulk_save(self, news_list: List[HotNewsItem]) -> int:
        """批量保存热点新闻"""
        if not news_list:
            return 0

        sql = """
            INSERT INTO hot_news (news_id, platform_code, title, url, score, description, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                title=VALUES(title),
                url=VALUES(url),
                score=VALUES(score),
                description=VALUES(description)
        """
        data = [
            (
                n.news_id,
                n.platform_code,
                n.title,
                n.url,
                n.score,
                n.description,
                n.created_at,
            )
            for n in news_list
        ]

        with self.db.get_cursor() as cursor:
            cursor.executemany(sql, data)
            return len(news_list)

    def get_by_id(self, news_id: str) -> Optional[dict]:
        """根据ID获取热点"""
        sql = "SELECT * FROM hot_news WHERE news_id = %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (news_id,))
            row = cursor.fetchone()
            return dict(zip([c[0] for c in cursor.description], row)) if row else None

    def get_recent(self, limit: int = 100, platform: str = None) -> List[dict]:
        """获取最近的热点"""
        sql = "SELECT * FROM hot_news"
        if platform:
            sql += " WHERE platform_code = %s"
        sql += " ORDER BY created_at DESC LIMIT %s"

        with self.db.get_cursor(commit=False) as cursor:
            if platform:
                cursor.execute(sql, (platform, limit))
            else:
                cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_by_platforms(self, platforms: List[str], limit: int = 100) -> List[dict]:
        """根据平台列表获取热点"""
        if not platforms:
            return self.get_recent(limit)

        placeholders = ",".join(["%s"] * len(platforms))
        sql = f"""
            SELECT * FROM hot_news 
            WHERE platform_code IN ({placeholders})
            ORDER BY created_at DESC LIMIT %s
        """

        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (*platforms, limit))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_unanalyzed_news(
        self, domain_id: int = None, limit: int = 100
    ) -> List[dict]:
        """获取未分析的热点"""
        if domain_id is None:
            sql = """
                SELECT * FROM hot_news 
                WHERE news_id NOT IN (SELECT DISTINCT news_id FROM hot_news_analysis)
                ORDER BY created_at DESC LIMIT %s
            """
            with self.db.get_cursor(commit=False) as cursor:
                cursor.execute(sql, (limit,))
                rows = cursor.fetchall()
                return [
                    dict(zip([c[0] for c in cursor.description], row)) for row in rows
                ]
        else:
            sql = """
                SELECT h.* FROM hot_news h
                LEFT JOIN hot_news_analysis a ON h.news_id = a.news_id AND a.domain_id = %s
                WHERE a.id IS NULL
                ORDER BY h.created_at DESC LIMIT %s
            """
            with self.db.get_cursor(commit=False) as cursor:
                cursor.execute(sql, (domain_id, limit))
                rows = cursor.fetchall()
                return [
                    dict(zip([c[0] for c in cursor.description], row)) for row in rows
                ]

    def count(self, platform: str = None) -> int:
        """统计数量"""
        sql = "SELECT COUNT(*) FROM hot_news"
        if platform:
            sql += " WHERE platform_code = %s"

        with self.db.get_cursor(commit=False) as cursor:
            if platform:
                cursor.execute(sql, (platform,))
            else:
                cursor.execute(sql)
            return cursor.fetchone()[0]

    def delete_old(self, days: int = 7) -> int:
        """删除旧数据"""
        sql = "DELETE FROM hot_news WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)"
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (days,))
            return cursor.rowcount
