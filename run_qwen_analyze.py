# -*- coding: utf-8 -*-
"""
使用 Qwen2.5 分析数据库中的数据

用法:
    python run_qwen_analyze.py                    # 分析全部数据
    python run_qwen_analyze.py --type content     # 只分析内容
    python run_qwen_analyze.py --type comment     # 只分析评论
    python run_qwen_analyze.py --batch-size 50    # 设置批次大小
    python run_qwen_analyze.py --dry-run          # 试运行 (不保存)
"""

import os
import re
import json
import argparse
import logging
from typing import List, Dict, Any

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的情感分析助手。请分析用户提供的文本，输出以下信息：
1. sentiment: 情感倾向 (positive/negative/neutral)
2. sentiment_score: 情感分数 (-1.0 到 1.0，负面为负，正面为正)
3. emotion_tags: 情绪标签列表 (可选: 喜悦、兴奋、满足、感激、爱、愤怒、厌恶、悲伤、恐惧、失望、惊讶、困惑、好奇、期待、焦虑、平静、无聊、冷漠)

请只输出 JSON 格式结果，不要有其他内容。"""

# 有效情绪标签
VALID_EMOTIONS = {
    "喜悦", "兴奋", "满足", "感激", "爱",
    "愤怒", "厌恶", "悲伤", "恐惧", "失望",
    "惊讶", "困惑", "好奇", "期待", "焦虑",
    "平静", "无聊", "冷漠"
}


class QwenAnalyzer:
    """Qwen2.5 情感分析器"""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return

        logger.info(f"正在加载模型: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("模型加载完成")

    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析模型响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                # 规范化 sentiment
                sentiment = result.get("sentiment", "neutral")
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"

                # 规范化 score
                score = result.get("sentiment_score", 0.0)
                if not isinstance(score, (int, float)):
                    score = 0.0
                score = max(-1.0, min(1.0, float(score)))

                # 规范化 emotion_tags
                emotion_tags = result.get("emotion_tags", [])
                if isinstance(emotion_tags, str):
                    emotion_tags = [emotion_tags]
                elif not isinstance(emotion_tags, list):
                    emotion_tags = []
                # 过滤无效标签
                emotion_tags = [t for t in emotion_tags if t in VALID_EMOTIONS]
                if not emotion_tags:
                    emotion_tags = ["平静"]

                return {
                    "sentiment": sentiment,
                    "sentiment_score": score,
                    "emotion_tags": emotion_tags
                }
            except json.JSONDecodeError:
                pass

        # 解析失败
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "emotion_tags": ["平静"]
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """预测单条文本"""
        self.load_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下文本的情感：\n\n{text}"}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return self.parse_response(response)

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """批量预测"""
        self.load_model()

        results = []
        iterator = tqdm(texts, desc="分析中") if show_progress else texts

        for text in iterator:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"预测失败: {e}")
                results.append({
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "emotion_tags": ["平静"]
                })

        return results


def analyze_content(analyzer: QwenAnalyzer, batch_size: int = 50, dry_run: bool = False) -> Dict[str, int]:
    """分析内容数据"""
    from SentimentModel.database import SentimentContentRepo

    stats = {"total": 0, "success": 0, "error": 0}

    # 统计待分析数量
    unanalyzed_count = SentimentContentRepo.count_unanalyzed()
    logger.info(f"待分析内容: {unanalyzed_count} 条")

    if unanalyzed_count == 0:
        logger.info("没有待分析的内容")
        return stats

    while True:
        # 获取一批数据
        data = SentimentContentRepo.get_unanalyzed(limit=batch_size)

        if not data:
            break

        stats["total"] += len(data)

        # 提取文本 (优先使用 content_cleaned，没有则用 title_cleaned)
        texts = []
        for item in data:
            text = item.get("content_cleaned") or item.get("title_cleaned") or ""
            texts.append(text[:500])  # 限制长度

        # 预测
        try:
            predictions = analyzer.predict_batch(texts, show_progress=True)

            # 组装结果
            results = []
            for item, pred in zip(data, predictions):
                results.append({
                    "unified_id": item["unified_id"],
                    "sentiment": pred["sentiment"],
                    "sentiment_score": pred["sentiment_score"],
                    "emotion_tags": pred["emotion_tags"]
                })

            if not dry_run:
                # 更新数据库
                updated = SentimentContentRepo.batch_update_sentiment(results)
                stats["success"] += updated
                logger.info(f"已更新 {updated} 条内容")
            else:
                stats["success"] += len(results)
                logger.info(f"[试运行] 分析了 {len(results)} 条内容")

                # 打印部分结果
                for i, (item, pred) in enumerate(zip(data[:3], predictions[:3])):
                    text = texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i]
                    print(f"  {text}")
                    print(f"    -> {pred['sentiment']} ({pred['sentiment_score']:.2f}) {pred['emotion_tags']}")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            stats["error"] += len(data)

    return stats


def analyze_comment(analyzer: QwenAnalyzer, batch_size: int = 50, dry_run: bool = False) -> Dict[str, int]:
    """分析评论数据"""
    from SentimentModel.database import SentimentCommentRepo

    stats = {"total": 0, "success": 0, "error": 0}

    # 统计待分析数量
    unanalyzed_count = SentimentCommentRepo.count_unanalyzed()
    logger.info(f"待分析评论: {unanalyzed_count} 条")

    if unanalyzed_count == 0:
        logger.info("没有待分析的评论")
        return stats

    while True:
        data = SentimentCommentRepo.get_unanalyzed(limit=batch_size)

        if not data:
            break

        stats["total"] += len(data)

        texts = [item.get("content_cleaned", "")[:500] for item in data]

        try:
            predictions = analyzer.predict_batch(texts, show_progress=True)

            results = []
            for item, pred in zip(data, predictions):
                results.append({
                    "unified_id": item["unified_id"],
                    "sentiment": pred["sentiment"],
                    "sentiment_score": pred["sentiment_score"],
                    "emotion_tags": pred["emotion_tags"]
                })

            if not dry_run:
                updated = SentimentCommentRepo.batch_update_sentiment(results)
                stats["success"] += updated
                logger.info(f"已更新 {updated} 条评论")
            else:
                stats["success"] += len(results)
                logger.info(f"[试运行] 分析了 {len(results)} 条评论")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            stats["error"] += len(data)

    return stats


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen2.5 分析数据库数据")
    parser.add_argument("--type", choices=["all", "content", "comment"], default="all", help="分析类型")
    parser.add_argument("--batch-size", type=int, default=50, help="批次大小")
    parser.add_argument("--dry-run", action="store_true", help="试运行 (不保存到数据库)")
    parser.add_argument("--model", default=MODEL_NAME, help="模型名称")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5 数据库情感分析")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"类型: {args.type}")
    print(f"批次: {args.batch_size}")
    print(f"试运行: {args.dry_run}")
    print("=" * 60)

    # 创建分析器
    analyzer = QwenAnalyzer(model_name=args.model)

    total_stats = {"total": 0, "success": 0, "error": 0}

    # 分析内容
    if args.type in ["all", "content"]:
        print("\n📝 分析内容...")
        stats = analyze_content(analyzer, args.batch_size, args.dry_run)
        print(f"内容分析完成: 总数 {stats['total']}, 成功 {stats['success']}, 失败 {stats['error']}")
        for k in total_stats:
            total_stats[k] += stats[k]

    # 分析评论
    if args.type in ["all", "comment"]:
        print("\n💬 分析评论...")
        stats = analyze_comment(analyzer, args.batch_size, args.dry_run)
        print(f"评论分析完成: 总数 {stats['total']}, 成功 {stats['success']}, 失败 {stats['error']}")
        for k in total_stats:
            total_stats[k] += stats[k]

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"总计: {total_stats['total']} 条")
    print(f"成功: {total_stats['success']} 条")
    print(f"失败: {total_stats['error']} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
