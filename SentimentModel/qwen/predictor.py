# -*- coding: utf-8 -*-
"""
Qwen2.5 情感预测器

用于对文本进行情感预测，输出 sentiment、sentiment_score、emotion_tags
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .data_prepare import SYSTEM_PROMPT, EMOTION_TAGS

logger = logging.getLogger(__name__)


@dataclass
class QwenSentimentResult:
    """Qwen 情感分析结果"""
    sentiment: str              # positive/negative/neutral
    sentiment_score: float      # -1.0 ~ 1.0
    emotion_tags: List[str]     # 情绪标签列表
    text: Optional[str] = None  # 原始文本
    raw_response: Optional[str] = None  # 模型原始响应

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "emotion_tags": self.emotion_tags,
            "text": self.text,
        }


class QwenSentimentPredictor:
    """
    Qwen2.5 情感预测器

    支持加载微调后的 LoRA 模型或合并后的完整模型
    """

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        is_merged: bool = False,
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        """
        初始化预测器

        Args:
            model_path: 模型路径 (LoRA 适配器或合并后的模型)
            base_model_name: 基础模型名称 (仅 LoRA 模式需要)
            is_merged: 是否是合并后的模型
            device: 计算设备
            load_in_4bit: 是否使用 4bit 量化加载
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.is_merged = is_merged
        self.device = device

        logger.info(f"加载模型: {model_path}")

        # 量化配置
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        if is_merged:
            # 加载合并后的完整模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        else:
            # 加载基础模型 + LoRA 适配器
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("模型加载完成")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析模型响应

        Args:
            response: 模型生成的文本

        Returns:
            解析后的字典
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                # 验证和规范化字段
                sentiment = result.get("sentiment", "neutral")
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"

                score = result.get("sentiment_score", 0.0)
                if not isinstance(score, (int, float)):
                    score = 0.0
                score = max(-1.0, min(1.0, float(score)))

                emotion_tags = result.get("emotion_tags", [])
                if not isinstance(emotion_tags, list):
                    emotion_tags = []
                # 过滤无效标签
                emotion_tags = [t for t in emotion_tags if t in EMOTION_TAGS]

                return {
                    "sentiment": sentiment,
                    "sentiment_score": score,
                    "emotion_tags": emotion_tags or ["平静"]
                }
            except json.JSONDecodeError:
                pass

        # 解析失败，使用默认值
        logger.warning(f"无法解析响应: {response[:100]}...")
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "emotion_tags": ["平静"]
        }

    def predict(self, text: str) -> QwenSentimentResult:
        """
        预测单条文本

        Args:
            text: 输入文本

        Returns:
            QwenSentimentResult 实例
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        show_progress: bool = False
    ) -> List[QwenSentimentResult]:
        """
        批量预测

        Args:
            texts: 文本列表
            max_new_tokens: 最大生成 token 数
            temperature: 生成温度
            show_progress: 是否显示进度

        Returns:
            QwenSentimentResult 列表
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="预测中")
        else:
            iterator = texts

        for text in iterator:
            # 构建消息
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析以下文本的情感：\n\n{text}"}
            ]

            # 应用聊天模板
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 分词
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # 解析响应
            parsed = self._parse_response(response)

            results.append(QwenSentimentResult(
                sentiment=parsed["sentiment"],
                sentiment_score=parsed["sentiment_score"],
                emotion_tags=parsed["emotion_tags"],
                text=text,
                raw_response=response
            ))

        return results

    def predict_for_db(
        self,
        data: List[Dict[str, Any]],
        text_field: str = "content",
        id_field: str = "unified_id",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对数据库数据进行预测

        Args:
            data: 数据库查询结果列表
            text_field: 文本字段名
            id_field: ID 字段名
            show_progress: 是否显示进度

        Returns:
            预测结果列表
        """
        texts = [item[text_field] for item in data]
        ids = [item[id_field] for item in data]

        results = self.predict_batch(texts, show_progress=show_progress)

        return [
            {
                "unified_id": uid,
                "sentiment": r.sentiment,
                "sentiment_score": r.sentiment_score,
                "emotion_tags": r.emotion_tags,
            }
            for uid, r in zip(ids, results)
        ]


def analyze_with_qwen(
    model_path: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    is_merged: bool = False,
    batch_size: int = 50,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    使用 Qwen 模型分析数据库中的内容

    Args:
        model_path: 模型路径
        base_model_name: 基础模型名称
        is_merged: 是否是合并后的模型
        batch_size: 批次大小
        dry_run: 是否试运行

    Returns:
        处理统计信息
    """
    from ..database import SentimentContentRepo

    predictor = QwenSentimentPredictor(
        model_path=model_path,
        base_model_name=base_model_name,
        is_merged=is_merged
    )

    stats = {"total": 0, "success": 0, "error": 0}

    while True:
        data = SentimentContentRepo.get_unanalyzed(limit=batch_size)

        if not data:
            break

        stats["total"] += len(data)

        try:
            results = predictor.predict_for_db(data)

            if not dry_run:
                # 更新数据库 (需要扩展 repository 支持 emotion_tags)
                for r in results:
                    SentimentContentRepo.update_sentiment_with_emotion(
                        unified_id=r["unified_id"],
                        sentiment=r["sentiment"],
                        sentiment_score=r["sentiment_score"],
                        emotion_tags=r["emotion_tags"]
                    )
                stats["success"] += len(results)
            else:
                stats["success"] += len(results)
                logger.info(f"[试运行] 分析了 {len(results)} 条内容")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            stats["error"] += len(data)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试预测
    predictor = QwenSentimentPredictor(
        model_path="models/qwen_sentiment/final_model",
        is_merged=True
    )

    test_texts = [
        "这个产品真的太好用了！",
        "非常失望，完全是垃圾",
        "还行吧，没什么特别的",
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n文本: {text}")
        print(f"情感: {result.sentiment}")
        print(f"分数: {result.sentiment_score:.2f}")
        print(f"情绪: {result.emotion_tags}")
