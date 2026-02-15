-- ============================================================
-- 005_knowledge_graph.sql
-- 知识图谱实体关系抽取结果存储
-- ============================================================

-- ==================== 1. 知识图谱抽取结果表 ====================
-- 存储 DeepSeek API 抽取的实体和关系（MySQL 侧暂存）

CREATE TABLE IF NOT EXISTS kg_extraction (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '抽取记录自增主键',
    topic_id BIGINT NOT NULL COMMENT '关联话题事件ID (topic_event.id)',
    unified_id BIGINT NOT NULL COMMENT '关联内容ID (unified_content.id)',

    -- ==================== 抽取结果 ====================
    entities JSON NOT NULL COMMENT '实体列表, 如[{"name":"小米","type":"Brand","properties":{}}]',
    relations JSON NOT NULL COMMENT '关系列表, 如[{"head":"小米","tail":"SU7","relation":"produced_by","properties":{}}]',

    -- ==================== 元数据 ====================
    model_name VARCHAR(100) DEFAULT 'deepseek-chat' COMMENT '抽取使用的模型',
    token_usage INT COMMENT 'API调用消耗的token数',
    extraction_status ENUM('pending', 'completed', 'failed', 'empty')
        DEFAULT 'completed' COMMENT '抽取状态',
    error_message TEXT COMMENT '失败时的错误信息',

    -- ==================== 系统时间 ====================
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '抽取时间',

    -- ==================== 索引 ====================
    UNIQUE KEY uk_topic_unified (topic_id, unified_id) COMMENT '同一话题同一内容仅保留一条抽取结果',
    INDEX idx_topic_id (topic_id) COMMENT '按话题查询所有抽取结果',
    INDEX idx_status (extraction_status) COMMENT '按状态筛选'

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '知识图谱抽取结果表 - DeepSeek API 实体关系抽取暂存';


-- ==================== 2. 知识图谱构建日志表 ====================

CREATE TABLE IF NOT EXISTS kg_build_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '日志自增主键',
    topic_id BIGINT NOT NULL COMMENT '关联话题事件ID',
    entity_count INT DEFAULT 0 COMMENT '写入Neo4j的实体数',
    relation_count INT DEFAULT 0 COMMENT '写入Neo4j的关系数',
    deduplicated_entities INT DEFAULT 0 COMMENT '去重后的实体数',
    build_status ENUM('success', 'failed', 'partial') DEFAULT 'success' COMMENT '构建状态',
    error_message TEXT COMMENT '失败时的错误信息',
    duration_seconds DECIMAL(10,2) COMMENT '构建耗时(秒)',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '构建时间',

    INDEX idx_topic_id (topic_id),
    INDEX idx_created_at (created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '知识图谱构建日志表 - 记录每次Neo4j图构建的统计信息';
