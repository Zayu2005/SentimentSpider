-- =====================================================
-- 统一内容表 & 统一评论表 DDL
-- =====================================================

-- 统一内容表
CREATE TABLE IF NOT EXISTS unified_content (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
    platform VARCHAR(20) NOT NULL COMMENT '平台代码: xhs/dy/wb/bili/ks/tieba/zhihu',
    content_id VARCHAR(64) NOT NULL COMMENT '原平台内容ID',
    content_type VARCHAR(20) DEFAULT 'note' COMMENT '内容类型: note/video/image',

    -- ==================== 作者信息 ====================
    user_id VARCHAR(64) COMMENT '作者用户ID',
    nickname VARCHAR(255) COMMENT '作者昵称',
    avatar TEXT COMMENT '作者头像URL',
    ip_location VARCHAR(100) COMMENT 'IP属地',
    gender VARCHAR(10) COMMENT '性别',

    -- ==================== 内容信息 ====================
    title TEXT COMMENT '标题',
    content TEXT COMMENT '正文内容',
    content_url VARCHAR(500) COMMENT '原文链接',

    -- ==================== 媒体信息 ====================
    media_type VARCHAR(20) COMMENT '媒体类型: video/image/text',
    cover_url TEXT COMMENT '封面图URL',
    video_url TEXT COMMENT '视频播放URL',
    video_download_url TEXT COMMENT '视频下载URL',
    image_list TEXT COMMENT '图片列表(JSON数组)',
    music_url TEXT COMMENT '音乐URL(抖音)',
    tag_list TEXT COMMENT '标签列表(JSON数组)',

    -- ==================== 互动数据 ====================
    liked_count INT DEFAULT 0 COMMENT '点赞数',
    comment_count INT DEFAULT 0 COMMENT '评论数',
    share_count INT DEFAULT 0 COMMENT '分享/转发数',
    collect_count INT DEFAULT 0 COMMENT '收藏数',
    view_count INT DEFAULT 0 COMMENT '播放/浏览数',
    coin_count INT DEFAULT 0 COMMENT '投币数(B站)',
    danmaku_count INT DEFAULT 0 COMMENT '弹幕数(B站)',

    -- ==================== 来源追踪 ====================
    source_keyword VARCHAR(200) COMMENT '搜索来源关键词',
    keyword_id INT COMMENT '关联extracted_keywords表ID',

    -- ==================== 情感分析结果 ====================
    sentiment ENUM('positive', 'negative', 'neutral') COMMENT '情感倾向',
    sentiment_score DECIMAL(5,4) COMMENT '情感得分(-1到1)',
    emotion_tags VARCHAR(200) COMMENT '情绪标签(愤怒/喜悦等)',
    sentiment_analyzed_at DATETIME COMMENT '情感分析时间',

    -- ==================== 时间信息 ====================
    original_created_at DATETIME COMMENT '内容原始发布时间',
    add_ts BIGINT COMMENT '原始入库时间戳(毫秒)',
    synced_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '同步到本表时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',

    -- ==================== 索引 ====================
    UNIQUE KEY uk_platform_content (platform, content_id),
    INDEX idx_source_keyword (source_keyword),
    INDEX idx_keyword_id (keyword_id),
    INDEX idx_sentiment (sentiment),
    INDEX idx_media_type (media_type),
    INDEX idx_synced_at (synced_at),
    INDEX idx_original_created (original_created_at),
    INDEX idx_platform_time (platform, original_created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '统一内容表 - 汇聚各平台内容数据';


-- 统一评论表
CREATE TABLE IF NOT EXISTS unified_comment (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '自增主键',
    platform VARCHAR(20) NOT NULL COMMENT '平台代码: xhs/dy/wb/bili/ks/tieba/zhihu',
    comment_id VARCHAR(64) NOT NULL COMMENT '评论唯一ID',
    content_id VARCHAR(64) NOT NULL COMMENT '所属内容ID',
    parent_comment_id VARCHAR(64) COMMENT '父评论ID(回复时)',

    -- ==================== 评论者信息 ====================
    user_id VARCHAR(64) COMMENT '评论者用户ID',
    nickname VARCHAR(255) COMMENT '评论者昵称',
    avatar TEXT COMMENT '评论者头像URL',
    ip_location VARCHAR(100) COMMENT 'IP属地',
    gender VARCHAR(10) COMMENT '性别',

    -- ==================== 评论内容 ====================
    content TEXT COMMENT '评论文本内容',
    pictures TEXT COMMENT '评论图片(JSON数组)',

    -- ==================== 互动数据 ====================
    liked_count INT DEFAULT 0 COMMENT '评论点赞数',
    reply_count INT DEFAULT 0 COMMENT '回复数/子评论数',

    -- ==================== 情感分析结果 ====================
    sentiment ENUM('positive', 'negative', 'neutral') COMMENT '情感倾向',
    sentiment_score DECIMAL(5,4) COMMENT '情感得分(-1到1)',
    emotion_tags VARCHAR(200) COMMENT '情绪标签',
    sentiment_analyzed_at DATETIME COMMENT '情感分析时间',

    -- ==================== 时间信息 ====================
    original_created_at DATETIME COMMENT '评论原始发布时间',
    add_ts BIGINT COMMENT '原始入库时间戳(毫秒)',
    synced_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '同步到本表时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',

    -- ==================== 索引 ====================
    UNIQUE KEY uk_platform_comment (platform, comment_id),
    INDEX idx_content (platform, content_id),
    INDEX idx_parent (parent_comment_id),
    INDEX idx_sentiment (sentiment),
    INDEX idx_synced_at (synced_at),
    INDEX idx_original_created (original_created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '统一评论表 - 汇聚各平台评论数据';
