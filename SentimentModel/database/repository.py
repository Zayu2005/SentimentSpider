# -*- coding: utf-8 -*-
"""
情感分析数据仓库

负责读取待分析数据和更新分析结果
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .connection import get_connection, execute_query, execute_many, execute_update


class SentimentContentRepo:
    """内容情感分析仓库"""

    @staticmethod
    def get_unanalyzed(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取未进行情感分析的内容

        从 processed_content 表读取已清洗但未分析情感的内容

        Args:
            limit: 获取数量
            offset: 偏移量

        Returns:
            未分析的内容列表，包含 unified_id, platform, content_cleaned, title_cleaned
        """
        sql = """
            SELECT
                pc.unified_id,
                pc.platform,
                pc.content_cleaned,
                pc.title_cleaned
            FROM processed_content pc
            JOIN unified_content uc ON pc.unified_id = uc.id
            WHERE uc.sentiment IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
            ORDER BY pc.id
            LIMIT %s OFFSET %s
        """
        return execute_query(sql, (limit, offset))

    @staticmethod
    def update_sentiment(
        unified_id: int,
        sentiment: str,
        sentiment_score: float,
        emotion_tags: Optional[List[str]] = None
    ) -> int:
        """
        更新内容的情感分析结果

        Args:
            unified_id: 统一内容ID
            sentiment: 情感标签 (positive/negative/neutral)
            sentiment_score: 情感分数 (-1.0 ~ 1.0)
            emotion_tags: 情绪标签列表 (可选)

        Returns:
            影响的行数
        """
        sql = """
            UPDATE unified_content
            SET
                sentiment = %s,
                sentiment_score = %s,
                emotion_tags = %s,
                sentiment_analyzed_at = NOW()
            WHERE id = %s
        """
        emotion_tags_json = json.dumps(emotion_tags, ensure_ascii=False) if emotion_tags else None
        return execute_update(sql, (sentiment, sentiment_score, emotion_tags_json, unified_id))

    @staticmethod
    def batch_update_sentiment(results: List[Dict[str, Any]]) -> int:
        """
        批量更新情感分析结果

        Args:
            results: 结果列表，每个元素包含:
                - unified_id: 统一内容ID
                - sentiment: 情感标签
                - sentiment_score: 情感分数
                - emotion_tags: 情绪标签 (可选)

        Returns:
            影响的行数
        """
        if not results:
            return 0

        sql = """
            UPDATE unified_content
            SET
                sentiment = %s,
                sentiment_score = %s,
                emotion_tags = %s,
                sentiment_analyzed_at = NOW()
            WHERE id = %s
        """
        params_list = [
            (
                r["sentiment"],
                r["sentiment_score"],
                json.dumps(r.get("emotion_tags"), ensure_ascii=False) if r.get("emotion_tags") else None,
                r["unified_id"]
            )
            for r in results
        ]

        # 使用单个连接批量更新
        count = 0
        with get_connection() as conn:
            with conn.cursor() as cursor:
                for params in params_list:
                    cursor.execute(sql, params)
                    count += cursor.rowcount
        return count

    @staticmethod
    def count_unanalyzed() -> int:
        """统计未分析的内容数量"""
        sql = """
            SELECT COUNT(*) as cnt
            FROM processed_content pc
            JOIN unified_content uc ON pc.unified_id = uc.id
            WHERE uc.sentiment IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
        """
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_analyzed() -> int:
        """统计已分析的内容数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_content WHERE sentiment IS NOT NULL"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总内容数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_content"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def get_sentiment_stats() -> Dict[str, Any]:
        """
        获取情感分布统计

        Returns:
            按平台和情感分类的统计信息
        """
        sql = """
            SELECT
                platform,
                sentiment,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score
            FROM unified_content
            WHERE sentiment IS NOT NULL
            GROUP BY platform, sentiment
            ORDER BY platform, sentiment
        """
        return execute_query(sql)


class SentimentCommentRepo:
    """评论情感分析仓库"""

    @staticmethod
    def get_unanalyzed(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取未进行情感分析的评论

        Args:
            limit: 获取数量
            offset: 偏移量

        Returns:
            未分析的评论列表
        """
        sql = """
            SELECT
                pc.unified_id,
                pc.platform,
                pc.content_cleaned
            FROM processed_comment pc
            JOIN unified_comment uc ON pc.unified_id = uc.id
            WHERE uc.sentiment IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
            ORDER BY pc.id
            LIMIT %s OFFSET %s
        """
        return execute_query(sql, (limit, offset))

    @staticmethod
    def update_sentiment(
        unified_id: int,
        sentiment: str,
        sentiment_score: float,
        emotion_tags: Optional[List[str]] = None
    ) -> int:
        """
        更新评论的情感分析结果

        Args:
            unified_id: 统一评论ID
            sentiment: 情感标签
            sentiment_score: 情感分数
            emotion_tags: 情绪标签列表

        Returns:
            影响的行数
        """
        sql = """
            UPDATE unified_comment
            SET
                sentiment = %s,
                sentiment_score = %s,
                emotion_tags = %s,
                sentiment_analyzed_at = NOW()
            WHERE id = %s
        """
        emotion_tags_json = json.dumps(emotion_tags, ensure_ascii=False) if emotion_tags else None
        return execute_update(sql, (sentiment, sentiment_score, emotion_tags_json, unified_id))

    @staticmethod
    def batch_update_sentiment(results: List[Dict[str, Any]]) -> int:
        """
        批量更新评论情感分析结果

        Args:
            results: 结果列表

        Returns:
            影响的行数
        """
        if not results:
            return 0

        sql = """
            UPDATE unified_comment
            SET
                sentiment = %s,
                sentiment_score = %s,
                emotion_tags = %s,
                sentiment_analyzed_at = NOW()
            WHERE id = %s
        """
        count = 0
        with get_connection() as conn:
            with conn.cursor() as cursor:
                for r in results:
                    emotion_tags_json = json.dumps(r.get("emotion_tags"), ensure_ascii=False) if r.get("emotion_tags") else None
                    cursor.execute(sql, (
                        r["sentiment"],
                        r["sentiment_score"],
                        emotion_tags_json,
                        r["unified_id"]
                    ))
                    count += cursor.rowcount
        return count

    @staticmethod
    def count_unanalyzed() -> int:
        """统计未分析的评论数量"""
        sql = """
            SELECT COUNT(*) as cnt
            FROM processed_comment pc
            JOIN unified_comment uc ON pc.unified_id = uc.id
            WHERE uc.sentiment IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
        """
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_analyzed() -> int:
        """统计已分析的评论数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_comment WHERE sentiment IS NOT NULL"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总评论数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_comment"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def get_sentiment_stats() -> Dict[str, Any]:
        """获取评论情感分布统计"""
        sql = """
            SELECT
                platform,
                sentiment,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score
            FROM unified_comment
            WHERE sentiment IS NOT NULL
            GROUP BY platform, sentiment
            ORDER BY platform, sentiment
        """
        return execute_query(sql)
