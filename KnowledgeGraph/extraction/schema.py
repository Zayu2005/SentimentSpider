# -*- coding: utf-8 -*-
"""
知识图谱 Schema 定义

定义中文社交媒体舆情分析的实体类型和关系类型
基于 OneKE 的 Schema Agent 理念，预定义以降低 API 调用成本
"""

# 实体类型定义 (中文 key，DeepSeek 直接输出中文)
KG_ENTITY_TYPES = {
    "人物": {
        "description": "具体的自然人，包括公众人物、企业家、网络红人等",
        "properties": ["角色", "职位", "所属机构"],
    },
    "组织机构": {
        "description": "企业、政府机构、NGO、媒体机构等",
        "properties": ["类型", "行业", "所在地"],
    },
    "品牌": {
        "description": "产品品牌、服务品牌",
        "properties": ["行业", "母公司"],
    },
    "产品": {
        "description": "具体产品或服务（如小米SU7、iPhone 16）",
        "properties": ["类别", "品牌", "版本"],
    },
    "地点": {
        "description": "地理位置、城市、国家、具体地点",
        "properties": ["级别", "国家"],
    },
    "事件": {
        "description": "具体发生的事件（如起火事故、发布会）",
        "properties": ["日期", "严重程度", "类别"],
    },
    "平台": {
        "description": "社交媒体平台（如微博、小红书、抖音）",
        "properties": ["类型"],
    },
}

# 关系类型定义 (中文 key)
KG_RELATION_TYPES = {
    "涉及": {
        "description": "人物或组织参与了某个事件",
        "head_types": ["人物", "组织机构", "品牌"],
        "tail_types": ["事件"],
    },
    "生产": {
        "description": "产品由品牌或组织生产/发布",
        "head_types": ["产品"],
        "tail_types": ["品牌", "组织机构"],
    },
    "位于": {
        "description": "事件、组织或人物关联的地理位置",
        "head_types": ["事件", "组织机构", "人物"],
        "tail_types": ["地点"],
    },
    "共同提及": {
        "description": "两个实体在同一上下文中被共同提及",
        "head_types": ["人物", "组织机构", "品牌", "产品"],
        "tail_types": ["人物", "组织机构", "品牌", "产品"],
    },
    "导致": {
        "description": "一个事件导致了另一个事件",
        "head_types": ["事件"],
        "tail_types": ["事件"],
    },
    "回应": {
        "description": "人物或组织对某事件做出回应",
        "head_types": ["人物", "组织机构", "品牌"],
        "tail_types": ["事件"],
    },
    "属于": {
        "description": "品牌属于组织，人物隶属于组织",
        "head_types": ["品牌", "人物"],
        "tail_types": ["组织机构"],
    },
    "竞争": {
        "description": "品牌或产品之间的竞争关系",
        "head_types": ["品牌", "产品"],
        "tail_types": ["品牌", "产品"],
    },
}

# 组合为 DeepSeek prompt 可用的 schema 描述
EXTRACTION_SCHEMA = {
    "entity_types": {
        k: v["description"] for k, v in KG_ENTITY_TYPES.items()
    },
    "relation_types": {
        k: v["description"] for k, v in KG_RELATION_TYPES.items()
    },
}
