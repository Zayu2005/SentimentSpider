# =====================================================
# SentimentProcessor - Segmenter
# 中文分词器
# =====================================================

from typing import List, Optional

import jieba
import jieba.analyse

from ..config import get_settings
from ..utils import StopwordsManager


class Segmenter:
    """中文分词器"""

    def __init__(self):
        """初始化分词器"""
        self.settings = get_settings().processor
        self._stopwords_manager: Optional[StopwordsManager] = None

        # 初始化停用词管理器
        if self.settings.use_stopwords:
            self._stopwords_manager = StopwordsManager(
                custom_stopwords=self.settings.custom_stopwords
            )

        # 初始化jieba
        self._init_jieba()

    def _init_jieba(self):
        """初始化jieba分词器"""
        # 预加载词典
        jieba.initialize()

        # 如果使用paddle模式
        if self.settings.use_paddle:
            try:
                jieba.enable_paddle()
            except Exception:
                # paddle不可用时使用默认模式
                pass

    def segment(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if not text:
            return []

        # 执行分词
        if self.settings.cut_all:
            words = list(jieba.cut(text, cut_all=True))
        else:
            words = list(jieba.cut(text))

        # 过滤空白词
        words = [w.strip() for w in words if w.strip()]

        # 过滤停用词
        if self._stopwords_manager and self.settings.use_stopwords:
            words = self._stopwords_manager.filter_stopwords(words)

        return words

    def segment_search(self, text: str) -> List[str]:
        """
        搜索引擎模式分词（更细粒度）

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if not text:
            return []

        words = list(jieba.cut_for_search(text))
        words = [w.strip() for w in words if w.strip()]

        if self._stopwords_manager and self.settings.use_stopwords:
            words = self._stopwords_manager.filter_stopwords(words)

        return words

    def add_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None):
        """添加自定义词语"""
        jieba.add_word(word, freq=freq, tag=tag)

    def del_word(self, word: str):
        """删除词语"""
        jieba.del_word(word)

    def load_userdict(self, file_path: str):
        """加载用户词典"""
        jieba.load_userdict(file_path)

    @property
    def stopwords_manager(self) -> Optional[StopwordsManager]:
        """获取停用词管理器"""
        return self._stopwords_manager
