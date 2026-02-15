-- ============================================================
-- 004_wordcloud.sql
-- 话题词云数据字段
-- ============================================================

ALTER TABLE topic_event
    ADD COLUMN wordcloud_data JSON COMMENT '词云数据(聚合TF-IDF关键词), 如[{"word":"售后","weight":15.8}]'
        AFTER keywords;
