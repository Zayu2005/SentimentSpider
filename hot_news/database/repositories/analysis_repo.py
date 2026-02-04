# =====================================================
# Hot News Module - Analysis Repository
# =====================================================

from typing import List, Optional
from datetime import datetime
from ..connection import get_db


class AnalysisRepository:
    """分析结果数据访问层"""

    def __init__(self):
        self.db = get_db()

    def save_analysis(
        self,
        news_id: str,
        domain_id: int,
        is_match: bool,
        llm_provider: str,
        analysis_content: str,
        confidence: float,
    ) -> bool:
        """保存分析结果"""
        sql = """
            INSERT INTO hot_news_analysis 
            (news_id, domain_id, is_match, llm_provider, analysis_content, confidence, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE 
                is_match=VALUES(is_match),
                llm_provider=VALUES(llm_provider),
                analysis_content=VALUES(analysis_content),
                confidence=VALUES(confidence)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql,
                (
                    news_id,
                    domain_id,
                    is_match,
                    llm_provider,
                    analysis_content,
                    confidence,
                ),
            )
            return True

    def get_analysis(self, news_id: str, domain_id: int) -> Optional[dict]:
        """获取分析结果"""
        sql = "SELECT * FROM hot_news_analysis WHERE news_id = %s AND domain_id = %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (news_id, domain_id))
            row = cursor.fetchone()
            return dict(zip([c[0] for c in cursor.description], row)) if row else None

    def get_matched_news(self, domain_id: int, limit: int = 100) -> List[dict]:
        """获取匹配的热点"""
        sql = """
            SELECT h.*, a.confidence, a.analysis_content
            FROM hot_news h
            JOIN hot_news_analysis a ON h.news_id = a.news_id
            WHERE a.domain_id = %s AND a.is_match = 1
            ORDER BY a.confidence DESC, h.created_at DESC
            LIMIT %s
        """
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (domain_id, limit))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_matched_news_by_keywords(
        self, keyword_ids: List[int], limit: int = 100
    ) -> List[dict]:
        """根据关键词获取匹配的热点"""
        if not keyword_ids:
            return []

        placeholders = ",".join(["%s"] * len(keyword_ids))
        sql = f"""
            SELECT DISTINCT h.*
            FROM hot_news h
            JOIN extracted_keywords k ON k.source_news_id = h.news_id
            WHERE k.id IN ({placeholders})
            ORDER BY h.created_at DESC
            LIMIT %s
        """
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (*keyword_ids, limit))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]
