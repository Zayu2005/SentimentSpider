# -*- coding: utf-8 -*-
"""
情感预测器

用于对文本进行情感预测
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from ..models import BertSentimentClassifier
from ..data import SentimentDataset, create_db_dataloader
from ..config import get_settings, ModelConfig, InferenceConfig, LABEL_NAMES


logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """情感分析结果"""
    label: str  # positive/negative/neutral
    label_id: int  # 0/1/2
    score: float  # -1.0 ~ 1.0
    probs: Dict[str, float]  # 各类别概率
    text: Optional[str] = None  # 原始文本

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "label": self.label,
            "label_id": self.label_id,
            "score": self.score,
            "probs": self.probs,
            "text": self.text
        }


class SentimentPredictor:
    """
    情感预测器

    支持单条和批量预测
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[BertSentimentClassifier] = None,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None
    ):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
            model: 模型实例 (如果提供则忽略 model_path)
            model_config: 模型配置
            inference_config: 推理配置
        """
        settings = get_settings()
        self.model_config = model_config or settings.model
        self.inference_config = inference_config or settings.inference

        # 设置设备
        self.device = self._get_device()
        logger.info(f"预测器使用设备: {self.device}")

        # 加载模型
        if model is not None:
            self.model = model
            self.model.to(self.device)
            model_name = model.model_name
        else:
            model_path = model_path or self.inference_config.model_path
            self.model = BertSentimentClassifier.load(model_path, self.device)
            model_name = self.model.model_name
            logger.info(f"从 {model_path} 加载模型")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 标签映射
        self.id2label = self.model_config.id2label
        self.label2id = self.model_config.label2id

    def _get_device(self) -> torch.device:
        """获取计算设备"""
        device_str = self.inference_config.device

        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_str)

    def predict(self, text: str) -> SentimentResult:
        """
        预测单条文本

        Args:
            text: 输入文本

        Returns:
            SentimentResult 实例
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[SentimentResult]:
        """
        批量预测

        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条

        Returns:
            SentimentResult 列表
        """
        batch_size = batch_size or self.inference_config.batch_size
        max_length = self.model_config.max_length

        # 创建数据集
        dataset = SentimentDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # 预测
        self.model.eval()
        all_results = []

        iterator = tqdm(dataloader, desc="Predicting") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                predictions, probs, scores = self.model.predict(input_ids, attention_mask)

                # 转换为结果
                for i in range(len(predictions)):
                    label_id = predictions[i].item()
                    label = self.id2label[label_id]
                    score = scores[i].item()
                    prob_dict = {
                        LABEL_NAMES[j]: probs[i][j].item()
                        for j in range(len(LABEL_NAMES))
                    }

                    all_results.append(SentimentResult(
                        label=label,
                        label_id=label_id,
                        score=score,
                        probs=prob_dict,
                        text=None  # 不保存原文以节省内存
                    ))

        return all_results

    def predict_from_db(
        self,
        data: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对数据库数据进行预测

        Args:
            data: 数据库查询结果列表
            batch_size: 批次大小
            show_progress: 是否显示进度条

        Returns:
            预测结果列表，每个元素包含:
                - unified_id
                - sentiment
                - sentiment_score
                - probs
        """
        batch_size = batch_size or self.inference_config.batch_size
        max_length = self.model_config.max_length

        # 创建数据加载器
        dataloader = create_db_dataloader(
            data=data,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_length=max_length
        )

        # 预测
        self.model.eval()
        all_results = []

        iterator = tqdm(dataloader, desc="分析中") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                unified_ids = batch["unified_ids"]

                predictions, probs, scores = self.model.predict(input_ids, attention_mask)

                # 转换为结果
                for i in range(len(predictions)):
                    label_id = predictions[i].item()
                    label = self.id2label[label_id]
                    score = scores[i].item()

                    all_results.append({
                        "unified_id": unified_ids[i],
                        "sentiment": label,
                        "sentiment_score": score,
                        "label_id": label_id,
                        "probs": {
                            LABEL_NAMES[j]: probs[i][j].item()
                            for j in range(len(LABEL_NAMES))
                        }
                    })

        return all_results


def analyze_content(
    predictor: SentimentPredictor,
    batch_size: int = 100,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    分析数据库中的内容

    Args:
        predictor: 预测器实例
        batch_size: 每批处理数量
        dry_run: 是否试运行 (不保存结果)

    Returns:
        处理统计信息
    """
    from ..database import SentimentContentRepo

    stats = {"total": 0, "success": 0, "error": 0}

    while True:
        # 获取未分析的内容
        data = SentimentContentRepo.get_unanalyzed(limit=batch_size)

        if not data:
            break

        stats["total"] += len(data)

        # 预测
        try:
            results = predictor.predict_from_db(data)

            if not dry_run:
                # 更新数据库
                updated = SentimentContentRepo.batch_update_sentiment(results)
                stats["success"] += updated
            else:
                stats["success"] += len(results)
                logger.info(f"[试运行] 分析了 {len(results)} 条内容")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            stats["error"] += len(data)

    return stats


def analyze_comment(
    predictor: SentimentPredictor,
    batch_size: int = 100,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    分析数据库中的评论

    Args:
        predictor: 预测器实例
        batch_size: 每批处理数量
        dry_run: 是否试运行

    Returns:
        处理统计信息
    """
    from ..database import SentimentCommentRepo

    stats = {"total": 0, "success": 0, "error": 0}

    while True:
        # 获取未分析的评论
        data = SentimentCommentRepo.get_unanalyzed(limit=batch_size)

        if not data:
            break

        stats["total"] += len(data)

        # 预测
        try:
            results = predictor.predict_from_db(data)

            if not dry_run:
                # 更新数据库
                updated = SentimentCommentRepo.batch_update_sentiment(results)
                stats["success"] += updated
            else:
                stats["success"] += len(results)
                logger.info(f"[试运行] 分析了 {len(results)} 条评论")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            stats["error"] += len(data)

    return stats
