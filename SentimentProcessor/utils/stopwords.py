# =====================================================
# SentimentProcessor - Stopwords Manager
# 停用词管理
# =====================================================

from pathlib import Path
from typing import Set, List, Optional


class StopwordsManager:
    """停用词管理器"""

    # 默认停用词（常用中文停用词）
    DEFAULT_STOPWORDS = {
        # 标点符号
        "，", "。", "！", "？", "、", "；", "：", """, """, "'", "'",
        "（", "）", "【", "】", "《", "》", "…", "—", "～", "·",
        ",", ".", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]",
        "<", ">", "-", "_", "/", "\\", "|", "@", "#", "$", "%", "^",
        "&", "*", "+", "=", "`", "~",
        # 常用虚词
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
        "她", "它", "们", "这个", "那个", "什么", "怎么", "为什么",
        "哪", "哪里", "哪个", "谁", "多少", "几", "怎样", "如何",
        "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "噢", "哈", "呵",
        "嘿", "哎", "唉", "哇", "咦", "喂", "嗨", "哼", "嘻", "呃",
        "而", "但", "但是", "然而", "可是", "虽然", "虽", "因为",
        "所以", "因此", "于是", "如果", "假如", "要是", "只要",
        "除非", "不管", "无论", "既然", "即使", "尽管", "并且",
        "而且", "或者", "还是", "不但", "不仅", "只是", "就是",
        "这样", "那样", "这么", "那么", "怎么样", "如此", "比较",
        "更", "最", "太", "非常", "特别", "十分", "相当", "极",
        "挺", "蛮", "颇", "稍", "略", "还", "又", "再", "已经",
        "曾经", "正在", "将要", "刚刚", "刚", "马上", "立刻",
        "从", "向", "往", "朝", "对", "给", "把", "被", "让", "叫",
        "使", "令", "为", "为了", "关于", "对于", "至于", "按照",
        "根据", "通过", "经过", "随着", "由于", "除了", "除",
        "可以", "能", "能够", "会", "应该", "必须", "需要", "可能",
        "或许", "大概", "也许", "一定", "肯定", "当然", "确实",
        "其实", "实际上", "事实上", "本来", "原来", "反正",
        "只", "仅", "仅仅", "光", "单", "才", "就", "便", "竟",
        "竟然", "居然", "果然", "终于", "总算", "到底", "究竟",
        "几乎", "差不多", "大约", "左右", "上下", "前后",
        "之", "之前", "之后", "之间", "之中", "以", "以前",
        "以后", "以来", "以内", "以外", "以上", "以下",
        "及", "及其", "以及", "并", "且", "与", "同",
        # 网络用语停用词
        "哈哈", "哈哈哈", "呵呵", "嘿嘿", "嘻嘻", "233", "666",
        "啦", "咯", "喽", "嘛", "么", "哒", "呐", "捏", "撒",
    }

    def __init__(self, custom_stopwords: Optional[List[str]] = None):
        """
        初始化停用词管理器

        Args:
            custom_stopwords: 自定义停用词列表
        """
        self._stopwords: Set[str] = self.DEFAULT_STOPWORDS.copy()
        if custom_stopwords:
            self._stopwords.update(custom_stopwords)

    def add(self, word: str) -> None:
        """添加停用词"""
        self._stopwords.add(word)

    def add_many(self, words: List[str]) -> None:
        """批量添加停用词"""
        self._stopwords.update(words)

    def remove(self, word: str) -> None:
        """移除停用词"""
        self._stopwords.discard(word)

    def is_stopword(self, word: str) -> bool:
        """判断是否为停用词"""
        return word in self._stopwords

    def filter_stopwords(self, words: List[str]) -> List[str]:
        """过滤停用词"""
        return [w for w in words if w not in self._stopwords and w.strip()]

    def load_from_file(self, file_path: str) -> None:
        """从文件加载停用词（每行一个词）"""
        path = Path(file_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self._stopwords.add(word)

    @property
    def stopwords(self) -> Set[str]:
        """获取所有停用词"""
        return self._stopwords.copy()

    def __len__(self) -> int:
        return len(self._stopwords)

    def __contains__(self, word: str) -> bool:
        return word in self._stopwords
