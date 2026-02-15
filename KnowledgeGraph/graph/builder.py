# -*- coding: utf-8 -*-
"""
Neo4j 知识图谱构建器

从 MySQL kg_extraction 表读取抽取结果，构建 Neo4j 图
"""

import json
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

from ..config import get_kg_settings
from ..database import KGExtractionRepo, KGBuildLogRepo, KGContentRepo
from ..database.neo4j_connection import neo4j_session
from ..utils import get_logger

logger = get_logger("KnowledgeGraph.builder")


@dataclass
class BuildResult:
    """构建结果统计"""
    raw_entities: int = 0
    raw_relations: int = 0
    deduplicated_entities: int = 0
    merged_relations: int = 0
    neo4j_nodes_created: int = 0
    neo4j_rels_created: int = 0


class GraphBuilder:
    """Neo4j 知识图谱构建器"""

    def __init__(self):
        self.settings = get_kg_settings()

    def _aggregate_extractions(
        self, topic_id: int
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """
        聚合话题下所有抽取结果，去重合并实体

        Returns:
            (entity_map, relation_list)
            entity_map: {(name_lower, type): {name, type, properties, mention_count}}
            relation_list: [{head, tail, relation, confidence, source_count}]
        """
        extractions = KGExtractionRepo.get_by_topic(topic_id)
        entity_map = {}
        relation_agg = defaultdict(
            lambda: {"confidence_sum": 0.0, "count": 0}
        )

        for ext in extractions:
            entities_raw = ext.get("entities")
            relations_raw = ext.get("relations")

            if isinstance(entities_raw, str):
                entities_raw = json.loads(entities_raw)
            if isinstance(relations_raw, str):
                relations_raw = json.loads(relations_raw)

            # 聚合实体
            for e in (entities_raw or []):
                key = (e["name"].strip().lower(), e["type"])
                if key not in entity_map:
                    entity_map[key] = {
                        "name": e["name"].strip(),
                        "type": e["type"],
                        "properties": e.get("properties", {}),
                        "mention_count": 1,
                    }
                else:
                    entity_map[key]["mention_count"] += 1
                    props = e.get("properties", {})
                    if props:
                        entity_map[key]["properties"].update(props)

            # 聚合关系
            for r in (relations_raw or []):
                head_key = r["head"].strip().lower()
                tail_key = r["tail"].strip().lower()
                rel_key = (head_key, tail_key, r["relation"])
                relation_agg[rel_key]["confidence_sum"] += r.get(
                    "confidence", 0.8
                )
                relation_agg[rel_key]["count"] += 1
                if "head_name" not in relation_agg[rel_key]:
                    relation_agg[rel_key]["head_name"] = r["head"].strip()
                    relation_agg[rel_key]["tail_name"] = r["tail"].strip()
                    relation_agg[rel_key]["relation"] = r["relation"]

        # 组装关系列表
        relations = []
        for key, agg in relation_agg.items():
            relations.append({
                "head": agg["head_name"],
                "tail": agg["tail_name"],
                "relation": agg["relation"],
                "confidence": round(
                    agg["confidence_sum"] / agg["count"], 3
                ),
                "source_count": agg["count"],
            })

        return entity_map, relations

    def _write_to_neo4j(
        self,
        topic_id: int,
        topic_info: Dict[str, Any],
        entity_map: Dict,
        relations: List[Dict],
    ) -> BuildResult:
        """将聚合后的实体和关系写入 Neo4j"""
        result = BuildResult()
        result.raw_entities = sum(
            e["mention_count"] for e in entity_map.values()
        )
        result.raw_relations = sum(r["source_count"] for r in relations)
        result.deduplicated_entities = len(entity_map)
        result.merged_relations = len(relations)

        with neo4j_session() as session:
            # 1. 创建或更新 TopicEvent 节点
            topic_name = topic_info.get("event_name", f"话题-{topic_id}")
            topic_desc = topic_info.get("event_description", "")
            session.run(
                """
                MERGE (t:TopicEvent {topic_id: $topic_id})
                SET t.name = $name,
                    t.description = $description,
                    t.updated_at = datetime()
                """,
                topic_id=topic_id,
                name=topic_name,
                description=topic_desc,
            )

            # 2. 创建实体节点并关联到话题
            for key, entity in entity_map.items():
                session.run(
                    """
                    MERGE (e:Entity {name: $name, entity_type: $type})
                    SET e.mention_count = COALESCE(e.mention_count, 0)
                                          + $mentions,
                        e.properties = $properties
                    WITH e
                    MATCH (t:TopicEvent {topic_id: $topic_id})
                    MERGE (e)-[:BELONGS_TO_TOPIC]->(t)
                    """,
                    name=entity["name"],
                    type=entity["type"],
                    mentions=entity["mention_count"],
                    properties=json.dumps(
                        entity["properties"], ensure_ascii=False
                    ),
                    topic_id=topic_id,
                )
                result.neo4j_nodes_created += 1

            # 3. 创建关系
            for rel in relations:
                session.run(
                    """
                    MATCH (h:Entity {name: $head})
                    MATCH (t:Entity {name: $tail})
                    MERGE (h)-[r:RELATES_TO {
                        relation_type: $rel_type,
                        topic_id: $topic_id
                    }]->(t)
                    SET r.confidence = $confidence,
                        r.source_count = $source_count
                    """,
                    head=rel["head"],
                    tail=rel["tail"],
                    rel_type=rel["relation"],
                    confidence=rel["confidence"],
                    source_count=rel["source_count"],
                    topic_id=topic_id,
                )
                result.neo4j_rels_created += 1

        return result

    def build_for_topic(
        self,
        topic_id: int,
        clear_existing: bool = False,
    ) -> BuildResult:
        """
        为话题构建 Neo4j 知识图谱

        Args:
            topic_id: 话题ID
            clear_existing: 是否清除该话题已有的图数据

        Returns:
            BuildResult 统计
        """
        start_time = time.time()

        # 获取话题信息
        topic_info = KGContentRepo.get_topic_info(topic_id)
        if not topic_info:
            raise ValueError(f"话题 {topic_id} 不存在")

        logger.info(
            f"开始构建话题 {topic_id} 的知识图谱: "
            f"{topic_info.get('event_name', '?')}"
        )

        # 清除已有数据
        if clear_existing:
            with neo4j_session() as session:
                session.run(
                    """
                    MATCH (e:Entity)-[:BELONGS_TO_TOPIC]->(t:TopicEvent {topic_id: $tid})
                    DETACH DELETE e
                    """,
                    tid=topic_id,
                )
                session.run(
                    "MATCH (t:TopicEvent {topic_id: $tid}) DETACH DELETE t",
                    tid=topic_id,
                )
            logger.info(f"已清除话题 {topic_id} 的已有图数据")

        # 聚合抽取结果
        entity_map, relations = self._aggregate_extractions(topic_id)

        if not entity_map:
            logger.warning(f"话题 {topic_id} 没有可用的抽取结果")
            return BuildResult()

        logger.info(
            f"聚合结果: {len(entity_map)} 个去重实体, "
            f"{len(relations)} 个去重关系"
        )

        # 写入 Neo4j
        result = self._write_to_neo4j(
            topic_id, topic_info, entity_map, relations
        )

        duration = time.time() - start_time

        # 记录构建日志
        KGBuildLogRepo.insert(
            topic_id=topic_id,
            entity_count=result.neo4j_nodes_created,
            relation_count=result.neo4j_rels_created,
            deduplicated_entities=result.deduplicated_entities,
            build_status="success",
            duration_seconds=round(duration, 2),
        )

        logger.info(
            f"话题 {topic_id} 知识图谱构建完成: "
            f"节点 {result.neo4j_nodes_created}, "
            f"关系 {result.neo4j_rels_created}, "
            f"耗时 {duration:.1f}s"
        )

        return result

    def query_topic_graph(self, topic_id: int) -> Dict[str, Any]:
        """查询话题图谱信息"""
        with neo4j_session() as session:
            # 节点统计
            node_result = session.run(
                """
                MATCH (e:Entity)-[:BELONGS_TO_TOPIC]->
                      (t:TopicEvent {topic_id: $tid})
                RETURN e.entity_type AS type, COUNT(*) AS cnt,
                       COLLECT(e.name)[..5] AS examples
                ORDER BY cnt DESC
                """,
                tid=topic_id,
            )
            node_stats = [dict(r) for r in node_result]

            # 关系统计
            rel_result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE r.topic_id = $tid
                RETURN r.relation_type AS type, COUNT(*) AS cnt
                ORDER BY cnt DESC
                """,
                tid=topic_id,
            )
            rel_stats = [dict(r) for r in rel_result]

            # 高提及实体 top 10
            top_entities_result = session.run(
                """
                MATCH (e:Entity)-[:BELONGS_TO_TOPIC]->
                      (t:TopicEvent {topic_id: $tid})
                RETURN e.name AS name, e.entity_type AS type,
                       e.mention_count AS mentions
                ORDER BY e.mention_count DESC
                LIMIT 10
                """,
                tid=topic_id,
            )
            top_entities = [dict(r) for r in top_entities_result]

            # 关键关系 top 10
            top_rels_result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE r.topic_id = $tid
                RETURN h.name AS head, r.relation_type AS relation,
                       t.name AS tail, r.confidence AS confidence,
                       r.source_count AS sources
                ORDER BY r.source_count DESC, r.confidence DESC
                LIMIT 10
                """,
                tid=topic_id,
            )
            top_relations = [dict(r) for r in top_rels_result]

        return {
            "node_stats": node_stats,
            "rel_stats": rel_stats,
            "top_entities": top_entities,
            "top_relations": top_relations,
        }
