# =====================================================
# SentimentProcessor - Repository
# 数据仓库 - 各表的CRUD操作
# =====================================================

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .connection import get_connection, execute_query, execute_many


class UnifiedContentRepo:
    """统一内容表仓库"""

    @staticmethod
    def get_unprocessed(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取未处理的内容

        Args:
            limit: 获取数量
            offset: 偏移量

        Returns:
            未处理的内容列表
        """
        sql = """
            SELECT uc.id, uc.platform, uc.content_id, uc.title, uc.content
            FROM unified_content uc
            LEFT JOIN processed_content pc ON uc.id = pc.unified_id
            WHERE pc.id IS NULL
            ORDER BY uc.id
            LIMIT %s OFFSET %s
        """
        return execute_query(sql, (limit, offset))

    @staticmethod
    def get_by_id(unified_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取内容"""
        sql = "SELECT * FROM unified_content WHERE id = %s"
        return execute_query(sql, (unified_id,), fetch_one=True)

    @staticmethod
    def count_unprocessed() -> int:
        """统计未处理的内容数量"""
        sql = """
            SELECT COUNT(*) as cnt
            FROM unified_content uc
            LEFT JOIN processed_content pc ON uc.id = pc.unified_id
            WHERE pc.id IS NULL
        """
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总内容数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_content"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0


class UnifiedCommentRepo:
    """统一评论表仓库"""

    @staticmethod
    def get_unprocessed(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取未处理的评论"""
        sql = """
            SELECT uc.id, uc.platform, uc.comment_id, uc.content_id, uc.content
            FROM unified_comment uc
            LEFT JOIN processed_comment pc ON uc.id = pc.unified_id
            WHERE pc.id IS NULL
            ORDER BY uc.id
            LIMIT %s OFFSET %s
        """
        return execute_query(sql, (limit, offset))

    @staticmethod
    def get_by_id(unified_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取评论"""
        sql = "SELECT * FROM unified_comment WHERE id = %s"
        return execute_query(sql, (unified_id,), fetch_one=True)

    @staticmethod
    def count_unprocessed() -> int:
        """统计未处理的评论数量"""
        sql = """
            SELECT COUNT(*) as cnt
            FROM unified_comment uc
            LEFT JOIN processed_comment pc ON uc.id = pc.unified_id
            WHERE pc.id IS NULL
        """
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总评论数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_comment"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0


class ProcessedContentRepo:
    """预处理内容表仓库"""

    @staticmethod
    def insert(data: Dict[str, Any]) -> int:
        """
        插入预处理结果

        Args:
            data: 预处理数据字典

        Returns:
            插入的记录ID
        """
        sql = """
            INSERT INTO processed_content (
                unified_id, platform, content_id,
                original_title, original_content,
                title_cleaned, content_cleaned,
                title_segmented, content_segmented, keywords,
                char_count, word_count,
                process_status, preprocessed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                title_cleaned = VALUES(title_cleaned),
                content_cleaned = VALUES(content_cleaned),
                title_segmented = VALUES(title_segmented),
                content_segmented = VALUES(content_segmented),
                keywords = VALUES(keywords),
                char_count = VALUES(char_count),
                word_count = VALUES(word_count),
                process_status = VALUES(process_status),
                preprocessed_at = VALUES(preprocessed_at)
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    data["unified_id"],
                    data["platform"],
                    data["content_id"],
                    data.get("original_title", ""),
                    data.get("original_content", ""),
                    data.get("title_cleaned", ""),
                    data.get("content_cleaned", ""),
                    json.dumps(data.get("title_segmented", []), ensure_ascii=False),
                    json.dumps(data.get("content_segmented", []), ensure_ascii=False),
                    json.dumps(data.get("keywords", []), ensure_ascii=False),
                    data.get("char_count", 0),
                    data.get("word_count", 0),
                    data.get("process_status", "completed"),
                    data.get("preprocessed_at", datetime.now()),
                ))
                return cursor.lastrowid

    @staticmethod
    def batch_insert(data_list: List[Dict[str, Any]]) -> int:
        """批量插入预处理结果"""
        if not data_list:
            return 0

        sql = """
            INSERT INTO processed_content (
                unified_id, platform, content_id,
                original_title, original_content,
                title_cleaned, content_cleaned,
                title_segmented, content_segmented, keywords,
                char_count, word_count,
                process_status, preprocessed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                title_cleaned = VALUES(title_cleaned),
                content_cleaned = VALUES(content_cleaned),
                title_segmented = VALUES(title_segmented),
                content_segmented = VALUES(content_segmented),
                keywords = VALUES(keywords),
                char_count = VALUES(char_count),
                word_count = VALUES(word_count),
                process_status = VALUES(process_status),
                preprocessed_at = VALUES(preprocessed_at)
        """
        params_list = [
            (
                d["unified_id"],
                d["platform"],
                d["content_id"],
                d.get("original_title", ""),
                d.get("original_content", ""),
                d.get("title_cleaned", ""),
                d.get("content_cleaned", ""),
                json.dumps(d.get("title_segmented", []), ensure_ascii=False),
                json.dumps(d.get("content_segmented", []), ensure_ascii=False),
                json.dumps(d.get("keywords", []), ensure_ascii=False),
                d.get("char_count", 0),
                d.get("word_count", 0),
                d.get("process_status", "completed"),
                d.get("preprocessed_at", datetime.now()),
            )
            for d in data_list
        ]
        return execute_many(sql, params_list)

    @staticmethod
    def count_by_status(status: str = "completed") -> int:
        """按状态统计数量"""
        sql = "SELECT COUNT(*) as cnt FROM processed_content WHERE process_status = %s"
        result = execute_query(sql, (status,), fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总数量"""
        sql = "SELECT COUNT(*) as cnt FROM processed_content"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0


class ProcessedCommentRepo:
    """预处理评论表仓库"""

    @staticmethod
    def insert(data: Dict[str, Any]) -> int:
        """插入预处理结果"""
        sql = """
            INSERT INTO processed_comment (
                unified_id, platform, comment_id, content_id,
                original_content, content_cleaned,
                content_segmented, keywords,
                char_count, word_count,
                process_status, preprocessed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                content_cleaned = VALUES(content_cleaned),
                content_segmented = VALUES(content_segmented),
                keywords = VALUES(keywords),
                char_count = VALUES(char_count),
                word_count = VALUES(word_count),
                process_status = VALUES(process_status),
                preprocessed_at = VALUES(preprocessed_at)
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    data["unified_id"],
                    data["platform"],
                    data["comment_id"],
                    data["content_id"],
                    data.get("original_content", ""),
                    data.get("content_cleaned", ""),
                    json.dumps(data.get("content_segmented", []), ensure_ascii=False),
                    json.dumps(data.get("keywords", []), ensure_ascii=False),
                    data.get("char_count", 0),
                    data.get("word_count", 0),
                    data.get("process_status", "completed"),
                    data.get("preprocessed_at", datetime.now()),
                ))
                return cursor.lastrowid

    @staticmethod
    def batch_insert(data_list: List[Dict[str, Any]]) -> int:
        """批量插入预处理结果"""
        if not data_list:
            return 0

        sql = """
            INSERT INTO processed_comment (
                unified_id, platform, comment_id, content_id,
                original_content, content_cleaned,
                content_segmented, keywords,
                char_count, word_count,
                process_status, preprocessed_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                content_cleaned = VALUES(content_cleaned),
                content_segmented = VALUES(content_segmented),
                keywords = VALUES(keywords),
                char_count = VALUES(char_count),
                word_count = VALUES(word_count),
                process_status = VALUES(process_status),
                preprocessed_at = VALUES(preprocessed_at)
        """
        params_list = [
            (
                d["unified_id"],
                d["platform"],
                d["comment_id"],
                d["content_id"],
                d.get("original_content", ""),
                d.get("content_cleaned", ""),
                json.dumps(d.get("content_segmented", []), ensure_ascii=False),
                json.dumps(d.get("keywords", []), ensure_ascii=False),
                d.get("char_count", 0),
                d.get("word_count", 0),
                d.get("process_status", "completed"),
                d.get("preprocessed_at", datetime.now()),
            )
            for d in data_list
        ]
        return execute_many(sql, params_list)

    @staticmethod
    def count_by_status(status: str = "completed") -> int:
        """按状态统计数量"""
        sql = "SELECT COUNT(*) as cnt FROM processed_comment WHERE process_status = %s"
        result = execute_query(sql, (status,), fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_all() -> int:
        """统计总数量"""
        sql = "SELECT COUNT(*) as cnt FROM processed_comment"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0
