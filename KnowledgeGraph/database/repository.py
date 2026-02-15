# -*- coding: utf-8 -*-
"""
知识图谱数据仓库

KGExtractionRepo  - 操作 kg_extraction 表 (抽取结果暂存)
KGBuildLogRepo    - 操作 kg_build_log 表  (构建日志)
KGContentRepo     - 读取 unified_content + processed_content
"""

import json
from typing import List, Dict, Any, Optional

from .mysql_connection import (
    execute_query,
    execute_update,
    execute_insert,
)


class KGExtractionRepo:
    """抽取结果仓库"""

    @staticmethod
    def upsert(
        topic_id: int,
        unified_id: int,
        entities: List[Dict],
        relations: List[Dict],
        model_name: str = "deepseek-chat",
        token_usage: Optional[int] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> int:
        """插入或更新抽取结果 (UPSERT on topic_id+unified_id)"""
        sql = """
            INSERT INTO kg_extraction
                (topic_id, unified_id, entities, relations,
                 model_name, token_usage, extraction_status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                entities = VALUES(entities),
                relations = VALUES(relations),
                model_name = VALUES(model_name),
                token_usage = VALUES(token_usage),
                extraction_status = VALUES(extraction_status),
                error_message = VALUES(error_message),
                created_at = CURRENT_TIMESTAMP
        """
        entities_json = json.dumps(entities, ensure_ascii=False)
        relations_json = json.dumps(relations, ensure_ascii=False)
        return execute_insert(sql, (
            topic_id, unified_id, entities_json, relations_json,
            model_name, token_usage, status, error_message,
        ))

    @staticmethod
    def get_by_topic(topic_id: int, status: str = "completed") -> List[Dict[str, Any]]:
        """获取话题下所有已完成的抽取结果"""
        sql = """
            SELECT id, topic_id, unified_id, entities, relations,
                   model_name, token_usage, extraction_status, created_at
            FROM kg_extraction
            WHERE topic_id = %s AND extraction_status = %s
            ORDER BY unified_id
        """
        return execute_query(sql, (topic_id, status))

    @staticmethod
    def count_by_topic(topic_id: int) -> Dict[str, int]:
        """统计话题的抽取情况"""
        sql = """
            SELECT extraction_status, COUNT(*) as cnt
            FROM kg_extraction
            WHERE topic_id = %s
            GROUP BY extraction_status
        """
        rows = execute_query(sql, (topic_id,))
        return {row["extraction_status"]: row["cnt"] for row in rows} if rows else {}

    @staticmethod
    def get_extracted_unified_ids(topic_id: int) -> set:
        """获取话题下已抽取的 unified_id 集合 (用于跳过已抽取内容)"""
        sql = """
            SELECT unified_id FROM kg_extraction
            WHERE topic_id = %s AND extraction_status IN ('completed', 'empty')
        """
        rows = execute_query(sql, (topic_id,))
        return {row["unified_id"] for row in rows} if rows else set()

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """全局抽取统计"""
        sql = """
            SELECT
                COUNT(DISTINCT topic_id) as topic_count,
                COUNT(*) as total_extractions,
                SUM(CASE WHEN extraction_status='completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN extraction_status='failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN extraction_status='empty' THEN 1 ELSE 0 END) as empty_results,
                SUM(token_usage) as total_tokens
            FROM kg_extraction
        """
        return execute_query(sql, fetch_one=True)


class KGBuildLogRepo:
    """构建日志仓库"""

    @staticmethod
    def insert(
        topic_id: int,
        entity_count: int,
        relation_count: int,
        deduplicated_entities: int,
        build_status: str = "success",
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> int:
        """插入构建日志"""
        sql = """
            INSERT INTO kg_build_log
                (topic_id, entity_count, relation_count, deduplicated_entities,
                 build_status, error_message, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        return execute_insert(sql, (
            topic_id, entity_count, relation_count, deduplicated_entities,
            build_status, error_message, duration_seconds,
        ))

    @staticmethod
    def get_latest_by_topic(topic_id: int) -> Optional[Dict[str, Any]]:
        """获取话题最近的构建日志"""
        sql = """
            SELECT * FROM kg_build_log
            WHERE topic_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        return execute_query(sql, (topic_id,), fetch_one=True)

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """全局构建统计"""
        sql = """
            SELECT
                COUNT(DISTINCT topic_id) as topics_built,
                SUM(entity_count) as total_entities,
                SUM(relation_count) as total_relations,
                SUM(CASE WHEN build_status='success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN build_status='failed' THEN 1 ELSE 0 END) as failed_count
            FROM kg_build_log
        """
        return execute_query(sql, fetch_one=True)


class KGContentRepo:
    """读取话题内容 (用于抽取的数据源)"""

    @staticmethod
    def get_topic_content(topic_id: int, limit: int = 200) -> List[Dict[str, Any]]:
        """获取话题下的内容 (JOIN processed_content)"""
        sql = """
            SELECT
                uc.id AS unified_id,
                uc.title,
                uc.platform,
                pc.content_cleaned,
                pc.title_cleaned,
                pc.keywords,
                uc.sentiment,
                uc.sentiment_score
            FROM unified_content uc
            JOIN processed_content pc ON pc.unified_id = uc.id
            WHERE uc.topic_id = %s
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
            ORDER BY uc.topic_similarity DESC
            LIMIT %s
        """
        return execute_query(sql, (topic_id, limit))

    @staticmethod
    def get_topic_info(topic_id: int) -> Optional[Dict[str, Any]]:
        """获取话题基本信息"""
        sql = """
            SELECT id, event_name, event_description, keywords,
                   status, content_count
            FROM topic_event
            WHERE id = %s
        """
        return execute_query(sql, (topic_id,), fetch_one=True)
