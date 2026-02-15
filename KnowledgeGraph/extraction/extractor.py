# -*- coding: utf-8 -*-
"""
实体关系抽取器

使用 OneKE (WWW 2025) 三 Agent 流水线进行实体关系抽取
Schema Agent → Extraction Agent → Reflection Agent
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..config import get_kg_settings
from ..database import KGExtractionRepo, KGContentRepo
from ..utils import get_logger
from .schema import KG_ENTITY_TYPES, KG_RELATION_TYPES

logger = get_logger("KnowledgeGraph.extractor")

# OneKE src 路径
_ONEKE_SRC = str(Path(__file__).parent.parent.parent / "OneKE" / "src")


@dataclass
class ExtractionResult:
    """抽取结果统计"""
    total_content: int = 0
    extracted: int = 0
    empty: int = 0
    failed: int = 0
    skipped: int = 0
    total_entities: int = 0
    total_relations: int = 0


class EntityRelationExtractor:
    """基于 OneKE 的实体关系抽取器"""

    def __init__(self, mode: str = "quick"):
        """
        Args:
            mode: OneKE 抽取模式
                  "quick"    - 快速模式 (Schema + Extraction)
                  "standard" - 标准模式 (Schema + Extraction + Reflection)
        """
        settings = get_kg_settings()
        self.api_key = settings.deepseek.api_key
        self.api_base = settings.deepseek.api_base
        self.model_name = settings.deepseek.model_name
        self.max_content_length = settings.extraction.max_content_length
        self.rate_limit_delay = settings.extraction.rate_limit_delay
        self.mode = mode

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置，请在 .env 中设置")

        # 初始化 OneKE Pipeline
        self.pipeline = self._init_pipeline()

        # 构建 constraint (实体类型 + 关系类型)
        self.entity_types = list(KG_ENTITY_TYPES.keys())
        self.relation_types = list(KG_RELATION_TYPES.keys())
        self.constraint = [self.entity_types, self.relation_types]

    def _init_pipeline(self):
        """初始化 OneKE Pipeline"""
        if _ONEKE_SRC not in sys.path:
            sys.path.insert(0, _ONEKE_SRC)

        from models import DeepSeek
        from pipeline import Pipeline

        model = DeepSeek(
            model_name_or_path=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base,
        )
        pipeline = Pipeline(model)
        logger.info(
            f"OneKE Pipeline 初始化完成 (模型: {self.model_name}, "
            f"模式: {self.mode})"
        )
        return pipeline

    def _parse_triple_result(self, result: Any) -> Dict[str, List]:
        """
        解析 OneKE Triple 任务的返回结果

        OneKE 返回格式:
        {"triple_list": [
            {"head": "小米", "head_type": "品牌",
             "relation": "生产", "relation_type": "生产",
             "tail": "SU7", "tail_type": "产品"}
        ]}
        或直接返回列表

        转换为我们的格式:
        {
            "entities": [{"name": ..., "type": ..., "properties": {}}],
            "relations": [{"head": ..., "tail": ..., "relation": ..., "confidence": ...}]
        }
        """
        triples = []
        if isinstance(result, dict):
            triples = result.get("triple_list", [])
        elif isinstance(result, list):
            triples = result
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    triples = parsed.get("triple_list", [])
                elif isinstance(parsed, list):
                    triples = parsed
            except json.JSONDecodeError:
                logger.warning(f"无法解析 OneKE 结果: {result[:200]}")
                return {"entities": [], "relations": []}

        valid_entity_types = set(KG_ENTITY_TYPES.keys())
        valid_relation_types = set(KG_RELATION_TYPES.keys())

        # 收集实体和关系
        entity_map = {}  # (name, type) -> entity dict
        relations = []

        for triple in triples:
            if not isinstance(triple, dict):
                continue

            head = triple.get("head", "").strip()
            tail = triple.get("tail", "").strip()
            head_type = triple.get("head_type", "").strip()
            tail_type = triple.get("tail_type", "").strip()
            relation = triple.get("relation", "").strip()

            if not head or not tail or not relation:
                continue

            # 添加实体 (即使类型不在预定义列表中也保留)
            if head and head_type:
                key = (head, head_type)
                if key not in entity_map:
                    entity_map[key] = {
                        "name": head,
                        "type": head_type,
                        "properties": {},
                    }

            if tail and tail_type:
                key = (tail, tail_type)
                if key not in entity_map:
                    entity_map[key] = {
                        "name": tail,
                        "type": tail_type,
                        "properties": {},
                    }

            # 添加关系
            relations.append({
                "head": head,
                "tail": tail,
                "relation": relation,
                "confidence": 0.9,
            })

        entities = list(entity_map.values())
        return {"entities": entities, "relations": relations}

    def extract_single(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """抽取单条内容的实体和关系"""
        title = content.get("title_cleaned") or content.get("title") or ""
        body = content.get("content_cleaned") or ""
        platform = content.get("platform", "未知")
        sentiment = content.get("sentiment") or "未知"

        text = (
            f"[{platform}] [{sentiment}] {title}\n"
            f"{body[:self.max_content_length]}"
        )

        result, trajectory, schema, _ = self.pipeline.get_extract_result(
            task="Triple",
            text=text,
            constraint=self.constraint,
            mode=self.mode,
            show_trajectory=False,
        )

        return self._parse_triple_result(result)

    def extract_for_topic(
        self,
        topic_id: int,
        limit: int = 200,
        dry_run: bool = False,
    ) -> ExtractionResult:
        """
        为话题下的所有内容执行实体关系抽取

        Args:
            topic_id: 话题ID
            limit: 最大处理内容数
            dry_run: 试运行

        Returns:
            ExtractionResult 统计
        """
        stats = ExtractionResult()

        # 获取话题内容
        contents = KGContentRepo.get_topic_content(topic_id, limit=limit)
        if not contents:
            logger.warning(f"话题 {topic_id} 没有可用内容")
            return stats

        stats.total_content = len(contents)

        # 获取已抽取的 unified_id (跳过)
        already_extracted = KGExtractionRepo.get_extracted_unified_ids(topic_id)

        for i, content in enumerate(contents, 1):
            uid = content["unified_id"]

            if uid in already_extracted:
                stats.skipped += 1
                continue

            title = (content.get("title_cleaned")
                     or content.get("title") or "")
            logger.info(
                f"[{i}/{stats.total_content}] 抽取内容 {uid}: "
                f"{title[:30]}..."
            )

            try:
                result = self.extract_single(content)
                entities = result.get("entities", [])
                relations = result.get("relations", [])

                if not entities:
                    status = "empty"
                    stats.empty += 1
                    logger.info(f"  -> 空结果 (无实体)")
                else:
                    status = "completed"
                    stats.extracted += 1
                    stats.total_entities += len(entities)
                    stats.total_relations += len(relations)
                    logger.info(
                        f"  -> {len(entities)} 实体, "
                        f"{len(relations)} 关系"
                    )

                if not dry_run:
                    KGExtractionRepo.upsert(
                        topic_id=topic_id,
                        unified_id=uid,
                        entities=entities,
                        relations=relations,
                        model_name=f"OneKE/{self.model_name}",
                        status=status,
                    )

                # 速率限制
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"内容 {uid} 抽取失败: {e}")
                stats.failed += 1

                if not dry_run:
                    KGExtractionRepo.upsert(
                        topic_id=topic_id,
                        unified_id=uid,
                        entities=[],
                        relations=[],
                        model_name=f"OneKE/{self.model_name}",
                        status="failed",
                        error_message=str(e)[:500],
                    )

        logger.info(
            f"话题 {topic_id} 抽取完成: "
            f"总计 {stats.total_content}, 成功 {stats.extracted}, "
            f"空 {stats.empty}, 失败 {stats.failed}, "
            f"跳过 {stats.skipped}, "
            f"实体 {stats.total_entities}, 关系 {stats.total_relations}"
        )
        return stats
