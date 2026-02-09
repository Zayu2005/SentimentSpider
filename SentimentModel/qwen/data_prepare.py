# -*- coding: utf-8 -*-
"""
数据准备模块

将微博情感数据集转换为 Qwen2.5 指令微调格式
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# 情绪标签定义 (参考 GoEmotions 但针对中文场景优化)
EMOTION_TAGS = [
    "喜悦", "兴奋", "满足", "感激", "爱",          # 正面情绪
    "愤怒", "厌恶", "悲伤", "恐惧", "失望",        # 负面情绪
    "惊讶", "困惑", "好奇", "期待", "焦虑",        # 复杂情绪
    "平静", "无聊", "冷漠",                        # 中性情绪
]

# 情绪到情感的映射
EMOTION_TO_SENTIMENT = {
    "喜悦": "positive", "兴奋": "positive", "满足": "positive",
    "感激": "positive", "爱": "positive",
    "愤怒": "negative", "厌恶": "negative", "悲伤": "negative",
    "恐惧": "negative", "失望": "negative",
    "惊讶": "neutral", "困惑": "neutral", "好奇": "neutral",
    "期待": "positive", "焦虑": "negative",
    "平静": "neutral", "无聊": "neutral", "冷漠": "neutral",
}

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的情感分析助手。请分析用户提供的文本，输出以下信息：
1. sentiment: 情感倾向 (positive/negative/neutral)
2. sentiment_score: 情感分数 (-1.0 到 1.0，负面为负，正面为正)
3. emotion_tags: 情绪标签列表

请以 JSON 格式输出结果。"""


def create_instruction_prompt(text: str) -> str:
    """创建指令提示词"""
    return f"请分析以下文本的情感：\n\n{text}"


def create_response(
    sentiment: str,
    sentiment_score: float,
    emotion_tags: List[str]
) -> str:
    """创建响应内容"""
    result = {
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 2),
        "emotion_tags": emotion_tags
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def infer_emotion_tags(text: str, sentiment: str) -> List[str]:
    """
    根据文本内容推断情绪标签

    这是一个简化的启发式方法，实际应用中可以使用更复杂的规则或预标注数据
    """
    text_lower = text.lower()
    tags = []

    # 正面情绪关键词
    positive_keywords = {
        "喜悦": ["开心", "高兴", "快乐", "太好了", "哈哈", "嘻嘻", "棒", "好", "赞"],
        "兴奋": ["激动", "兴奋", "太棒了", "太厉害", "牛", "nb", "awesome"],
        "满足": ["满足", "满意", "舒服", "享受", "幸福"],
        "感激": ["感谢", "谢谢", "多亏", "感恩", "感激"],
        "爱": ["爱", "喜欢", "爱你", "心", "❤", "💕", "😍"],
    }

    # 负面情绪关键词
    negative_keywords = {
        "愤怒": ["气死", "生气", "愤怒", "怒", "火大", "可恶", "操", "艹"],
        "厌恶": ["恶心", "讨厌", "烦", "垃圾", "恶", "呸"],
        "悲伤": ["难过", "伤心", "哭", "悲", "痛", "😢", "😭"],
        "恐惧": ["害怕", "恐惧", "吓", "怕", "担心"],
        "失望": ["失望", "遗憾", "可惜", "唉", "差", "烂"],
    }

    # 中性/复杂情绪关键词
    neutral_keywords = {
        "惊讶": ["惊", "震惊", "没想到", "居然", "竟然", "wow"],
        "困惑": ["困惑", "迷惑", "不懂", "什么意思", "为什么"],
        "好奇": ["好奇", "想知道", "是什么", "怎么"],
        "期待": ["期待", "希望", "盼", "等"],
        "焦虑": ["焦虑", "着急", "急", "慌"],
    }

    all_keywords = {**positive_keywords, **negative_keywords, **neutral_keywords}

    for emotion, keywords in all_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                if emotion not in tags:
                    tags.append(emotion)
                break

    # 如果没有匹配到具体情绪，根据情感倾向给默认标签
    if not tags:
        if sentiment == "positive":
            tags = ["喜悦"]
        elif sentiment == "negative":
            tags = ["失望"]
        else:
            tags = ["平静"]

    return tags[:3]  # 最多返回3个标签


def convert_weibo_label(label: int) -> Tuple[str, float]:
    """
    转换微博数据集标签

    Args:
        label: 原始标签 (0=负面, 1=正面)

    Returns:
        (sentiment, sentiment_score) 元组
    """
    if label == 0:
        return "negative", random.uniform(-0.9, -0.5)
    else:
        return "positive", random.uniform(0.5, 0.9)


def prepare_sentiment_data(
    input_file: Optional[str] = None,
    output_dir: str = "data/qwen_finetune",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Dict[str, str]:
    """
    准备微调数据集

    Args:
        input_file: 输入 CSV 文件路径 (微博数据集)
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        max_samples: 最大样本数 (用于测试)
        seed: 随机种子

    Returns:
        包含 train/val/test 文件路径的字典
    """
    import pandas as pd
    from datasets import load_dataset

    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if input_file:
        df = pd.read_csv(input_file)
        texts = df["review"].tolist()
        labels = df["label"].tolist()
    else:
        # 使用 HuggingFace 数据集
        logger.info("从 HuggingFace 加载 weibo_senti_100k 数据集...")
        dataset = load_dataset("IDEA-CCNL/weibo_senti_100k", split="train")
        texts = dataset["review"]
        labels = dataset["label"]

    if max_samples:
        indices = random.sample(range(len(texts)), min(max_samples, len(texts)))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    logger.info(f"共 {len(texts)} 条数据")

    # 转换为指令格式
    data = []
    for text, label in zip(texts, labels):
        sentiment, score = convert_weibo_label(label)
        emotion_tags = infer_emotion_tags(text, sentiment)

        instruction = create_instruction_prompt(text)
        response = create_response(sentiment, score, emotion_tags)

        data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "system": SYSTEM_PROMPT,
            # 保留原始信息用于验证
            "_text": text,
            "_label": label,
        })

    # 打乱数据
    random.shuffle(data)

    # 划分数据集
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    logger.info(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 保存数据
    paths = {}
    for split, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        file_path = output_path / f"{split}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        paths[split] = str(file_path)
        logger.info(f"保存 {split} 到 {file_path}")

    return paths


def load_finetune_data(file_path: str) -> List[Dict]:
    """加载微调数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试数据准备
    paths = prepare_sentiment_data(
        output_dir="data/qwen_finetune",
        max_samples=1000  # 测试用
    )
    print(f"数据已准备: {paths}")
