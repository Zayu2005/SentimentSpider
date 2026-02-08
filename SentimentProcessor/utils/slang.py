# =====================================================
# SentimentProcessor - Slang Normalizer
# 网络用语规范化
# =====================================================

from typing import Dict, Optional
import re


class SlangNormalizer:
    """网络用语规范化器"""

    # 常见网络用语映射表
    DEFAULT_SLANG_MAP: Dict[str, str] = {
        # 情绪表达
        "yyds": "永远的神",
        "YYDS": "永远的神",
        "绝绝子": "太绝了",
        "无语子": "无语",
        "笑死": "很好笑",
        "笑死了": "很好笑",
        "笑哭": "很好笑",
        "裂开": "崩溃",
        "破防": "情绪崩溃",
        "破大防": "情绪崩溃",
        "爆哭": "大哭",
        "暴风哭泣": "大哭",
        "DNA动了": "被触动",
        "上头": "沉迷",
        "下头": "失望",
        "栓Q": "谢谢",
        "蚌埠住了": "绷不住了",
        "泰裤辣": "太酷了",
        "特种兵式": "高效快速",

        # 肯定/赞美
        "牛": "厉害",
        "牛逼": "厉害",
        "nb": "厉害",
        "NB": "厉害",
        "牛批": "厉害",
        "6": "厉害",
        "66": "厉害",
        "666": "厉害",
        "6666": "厉害",
        "tql": "太强了",
        "TQL": "太强了",
        "太强了": "很强",
        "绝了": "太好了",
        "芜湖": "好棒",
        "awsl": "可爱死了",
        "AWSL": "可爱死了",
        "太可了": "太可爱了",
        "真香": "真好",
        "神仙": "很好",
        "神": "很好",
        "顶": "支持",
        "顶顶": "支持",
        "火钳刘明": "火前留名",
        "蹲": "等待",

        # 否定/批评
        "吐了": "讨厌",
        "yue了": "想吐",
        "无语": "无话可说",
        "离谱": "不可思议",
        "离大谱": "非常离谱",
        "服了": "无奈",
        "醉了": "无语",
        "佛了": "无语",
        "怎么说呢": "不好评价",
        "就这": "很差",
        "就这？": "很差",
        "就这?": "很差",
        "辣眼睛": "很难看",
        "辣鸡": "垃圾",
        "渣渣": "垃圾",
        "拉胯": "很差",
        "拉跨": "很差",
        "摆烂": "放弃",
        "摆了": "放弃",
        "老六": "阴险的人",
        "挂": "作弊",
        "脱粉": "不再喜欢",
        "避雷": "不推荐",

        # 语气词
        "hhhh": "哈哈哈哈",
        "hhh": "哈哈哈",
        "hh": "哈哈",
        "233": "哈哈哈",
        "2333": "哈哈哈",
        "23333": "哈哈哈",
        "xswl": "笑死我了",
        "XSWL": "笑死我了",
        "emmm": "嗯",
        "emm": "嗯",
        "em": "嗯",
        "额": "嗯",
        "昂": "嗯",
        "咋": "怎么",
        "咋了": "怎么了",
        "咋整": "怎么办",
        "咋滴": "怎么",
        "啥": "什么",
        "啥玩意": "什么东西",
        "搞毛": "干什么",
        "咋肥四": "怎么回事",

        # 互联网缩写
        "xdm": "兄弟们",
        "XDM": "兄弟们",
        "jms": "姐妹们",
        "JMS": "姐妹们",
        "gg": "哥哥",
        "GG": "哥哥",
        "jj": "姐姐",
        "JJ": "姐姐",
        "mm": "妹妹",
        "MM": "妹妹",
        "dd": "弟弟",
        "DD": "弟弟",
        "gf": "女朋友",
        "GF": "女朋友",
        "bf": "男朋友",
        "BF": "男朋友",
        "py": "朋友",
        "PY": "朋友",
        "zqsg": "真情实感",
        "ZQSG": "真情实感",
        "dbq": "对不起",
        "DBQ": "对不起",
        "nsdd": "你说得对",
        "NSDD": "你说得对",
        "xjj": "小姐姐",
        "XJJ": "小姐姐",
        "xgg": "小哥哥",
        "XGG": "小哥哥",
        "bdjw": "不懂就问",
        "BDJW": "不懂就问",
        "yygq": "阴阳怪气",
        "YYGQ": "阴阳怪气",
        "ssfd": "瑟瑟发抖",
        "SSFD": "瑟瑟发抖",
        "cpdd": "处对象",
        "CPDD": "处对象",

        # 饭圈用语
        "安利": "推荐",
        "种草": "推荐",
        "拔草": "买了",
        "入坑": "开始喜欢",
        "脱坑": "不再喜欢",
        "pick": "选择支持",
        "打call": "支持",
        "应援": "支持",
        "爬墙": "转移喜好",
        "糊了": "不火了",
        "糊": "不火",
        "出圈": "变得流行",
        "塌房": "偶像出问题",
        "锤": "揭露黑料",
        "洗白": "辩解",
        "控评": "控制评论",
    }

    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        """
        初始化网络用语规范化器

        Args:
            custom_map: 自定义映射表
        """
        self._slang_map = self.DEFAULT_SLANG_MAP.copy()
        if custom_map:
            self._slang_map.update(custom_map)
        # 按长度降序排列，确保长词优先匹配
        self._sorted_keys = sorted(self._slang_map.keys(), key=len, reverse=True)

    def add(self, slang: str, normalized: str) -> None:
        """添加映射"""
        self._slang_map[slang] = normalized
        self._sorted_keys = sorted(self._slang_map.keys(), key=len, reverse=True)

    def normalize(self, text: str) -> str:
        """规范化文本中的网络用语"""
        result = text
        for slang in self._sorted_keys:
            if slang in result:
                result = result.replace(slang, self._slang_map[slang])
        return result

    def normalize_repeated_chars(self, text: str, max_repeat: int = 3) -> str:
        """
        规范化重复字符

        例如: "好好好好好" -> "好好好"
        """
        # 匹配连续重复的字符
        pattern = r'(.)\1{' + str(max_repeat) + r',}'
        return re.sub(pattern, r'\1' * max_repeat, text)

    def normalize_all(self, text: str, max_repeat: int = 3) -> str:
        """应用所有规范化"""
        text = self.normalize(text)
        text = self.normalize_repeated_chars(text, max_repeat)
        return text

    @property
    def slang_map(self) -> Dict[str, str]:
        """获取映射表"""
        return self._slang_map.copy()
