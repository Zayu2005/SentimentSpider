# =====================================================
# SentimentProcessor - Keyword Extractor
# 关键词提取器
# =====================================================

from typing import List, Tuple

import jieba.analyse


class KeywordExtractor:
    """关键词提取器"""

    def __init__(self, topK: int = 10, with_weight: bool = False):
        """
        初始化关键词提取器

        Args:
            topK: 提取关键词数量
            with_weight: 是否返回权重
        """
        self.topK = topK
        self.with_weight = with_weight

    def extract_tfidf(
        self,
        text: str,
        topK: int = None,
        with_weight: bool = None,
        allow_pos: Tuple[str, ...] = ()
    ) -> List[str] | List[Tuple[str, float]]:
        """
        使用TF-IDF提取关键词

        Args:
            text: 输入文本
            topK: 提取数量（None使用默认值）
            with_weight: 是否返回权重
            allow_pos: 允许的词性（空元组表示不限制）

        Returns:
            关键词列表或(关键词, 权重)列表
        """
        if not text:
            return []

        topK = topK or self.topK
        with_weight = with_weight if with_weight is not None else self.with_weight

        if allow_pos:
            return jieba.analyse.extract_tags(
                text,
                topK=topK,
                withWeight=with_weight,
                allowPOS=allow_pos
            )
        return jieba.analyse.extract_tags(
            text,
            topK=topK,
            withWeight=with_weight
        )

    def extract_textrank(
        self,
        text: str,
        topK: int = None,
        with_weight: bool = None,
        allow_pos: Tuple[str, ...] = ('ns', 'n', 'vn', 'v')
    ) -> List[str] | List[Tuple[str, float]]:
        """
        使用TextRank提取关键词

        Args:
            text: 输入文本
            topK: 提取数量（None使用默认值）
            with_weight: 是否返回权重
            allow_pos: 允许的词性

        Returns:
            关键词列表或(关键词, 权重)列表
        """
        if not text:
            return []

        topK = topK or self.topK
        with_weight = with_weight if with_weight is not None else self.with_weight

        return jieba.analyse.textrank(
            text,
            topK=topK,
            withWeight=with_weight,
            allowPOS=allow_pos
        )

    def extract(
        self,
        text: str,
        method: str = "tfidf",
        topK: int = None,
        with_weight: bool = None
    ) -> List[str] | List[Tuple[str, float]]:
        """
        提取关键词

        Args:
            text: 输入文本
            method: 提取方法 ("tfidf" 或 "textrank")
            topK: 提取数量
            with_weight: 是否返回权重

        Returns:
            关键词列表
        """
        if method == "textrank":
            return self.extract_textrank(text, topK, with_weight)
        return self.extract_tfidf(text, topK, with_weight)

    def set_idf_path(self, idf_path: str):
        """设置IDF语料库路径"""
        jieba.analyse.set_idf_path(idf_path)

    def set_stop_words(self, stop_words_path: str):
        """设置停用词路径"""
        jieba.analyse.set_stop_words(stop_words_path)
