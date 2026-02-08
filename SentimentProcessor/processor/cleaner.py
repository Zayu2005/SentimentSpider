# =====================================================
# SentimentProcessor - Text Cleaner
# 文本清洗器
# =====================================================

import re
from typing import Optional

from opencc import OpenCC

from ..config import get_settings
from ..utils import SlangNormalizer


class TextCleaner:
    """文本清洗器"""

    # 正则表达式模式
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'www\.[^\s<>"{}|\\^`\[\]]+'
    )
    EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    MENTION_PATTERN = re.compile(r'@[\w\u4e00-\u9fff]+')
    HASHTAG_PATTERN = re.compile(r'#([^#\s]+)#?')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    # Emoji 正则表达式 - 只匹配真正的 emoji，避免匹配中文
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "\U00002600-\U000026FF"  # misc symbols (不包含中文范围)
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # playing cards
        "]+",
        flags=re.UNICODE
    )
    # 小红书等平台的表情格式: [笑哭R] [赞R] [哭惹R] 等
    PLATFORM_EMOJI_PATTERN = re.compile(r'\[[^\]]{1,10}R\]')
    # 小红书话题标签后缀: [话题]
    TOPIC_SUFFIX_PATTERN = re.compile(r'\[话题\]')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\u4e00-\u9fff\s]')

    def __init__(self):
        """初始化文本清洗器"""
        self.settings = get_settings().processor
        self._opencc: Optional[OpenCC] = None
        self._slang_normalizer: Optional[SlangNormalizer] = None

        # 延迟加载 OpenCC
        if self.settings.to_simplified:
            self._opencc = OpenCC('t2s')

        # 延迟加载网络用语规范化器
        if self.settings.normalize_slang:
            self._slang_normalizer = SlangNormalizer()

    def clean(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        result = text

        # 移除HTML标签
        if self.settings.remove_html:
            result = self.remove_html(result)

        # 移除URL
        if self.settings.remove_urls:
            result = self.remove_urls(result)

        # 移除邮箱
        if self.settings.remove_emails:
            result = self.remove_emails(result)

        # 移除@提及
        if self.settings.remove_mentions:
            result = self.remove_mentions(result)

        # 处理话题标签
        if self.settings.remove_hashtags:
            result = self.remove_hashtags(result)
        else:
            result = self.extract_hashtag_content(result)

        # 移除表情符号
        if self.settings.remove_emojis:
            result = self.remove_emojis(result)

        # 繁体转简体
        if self.settings.to_simplified and self._opencc:
            result = self._opencc.convert(result)

        # 网络用语规范化
        if self.settings.normalize_slang and self._slang_normalizer:
            result = self._slang_normalizer.normalize_all(result)

        # 规范化空白字符
        if self.settings.normalize_whitespace:
            result = self.normalize_whitespace(result)

        return result.strip()

    def remove_urls(self, text: str) -> str:
        """移除URL"""
        return self.URL_PATTERN.sub('', text)

    def remove_emails(self, text: str) -> str:
        """移除邮箱"""
        return self.EMAIL_PATTERN.sub('', text)

    def remove_mentions(self, text: str) -> str:
        """移除@提及"""
        return self.MENTION_PATTERN.sub('', text)

    def remove_hashtags(self, text: str) -> str:
        """移除话题标签"""
        return self.HASHTAG_PATTERN.sub('', text)

    def extract_hashtag_content(self, text: str) -> str:
        """提取话题标签内容（保留话题文字，移除#号和[话题]后缀）"""
        result = self.HASHTAG_PATTERN.sub(r'\1', text)
        result = self.TOPIC_SUFFIX_PATTERN.sub('', result)
        return result

    def remove_html(self, text: str) -> str:
        """移除HTML标签"""
        return self.HTML_PATTERN.sub('', text)

    def remove_emojis(self, text: str) -> str:
        """移除表情符号"""
        result = self.EMOJI_PATTERN.sub('', text)
        # 移除平台特定表情格式 [笑哭R] 等
        result = self.PLATFORM_EMOJI_PATTERN.sub('', result)
        return result

    def normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        return self.WHITESPACE_PATTERN.sub(' ', text)

    def remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        return self.PUNCTUATION_PATTERN.sub('', text)

    def clean_for_segmentation(self, text: str) -> str:
        """为分词准备的清洗（保留更多信息）"""
        if not text:
            return ""

        result = text

        # 基础清洗
        result = self.remove_html(result)
        result = self.remove_urls(result)

        # 繁体转简体
        if self.settings.to_simplified and self._opencc:
            result = self._opencc.convert(result)

        # 规范化空白
        result = self.normalize_whitespace(result)

        return result.strip()
