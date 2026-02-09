# -*- coding: utf-8 -*-
"""
模型训练器

负责模型训练、验证和保存
"""

from typing import Optional, Dict, Any, List
import os
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from ..models import BertSentimentClassifier
from ..config import TrainingConfig, ModelConfig, get_settings
from .metrics import compute_metrics, MetricResult


logger = logging.getLogger(__name__)


class SentimentTrainer:
    """
    情感分析模型训练器
    """

    def __init__(
        self,
        model: Optional[BertSentimentClassifier] = None,
        training_config: Optional[TrainingConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        初始化训练器

        Args:
            model: 模型实例 (可选，不提供则自动创建)
            training_config: 训练配置
            model_config: 模型配置
        """
        self.training_config = training_config or get_settings().training
        self.model_config = model_config or get_settings().model

        # 设置设备
        self.device = self._get_device()
        logger.info(f"使用设备: {self.device}")

        # 创建或使用提供的模型
        if model is None:
            self.model = BertSentimentClassifier(
                model_name=self.model_config.name,
                num_labels=self.model_config.num_labels,
                dropout=self.model_config.dropout
            )
        else:
            self.model = model

        self.model.to(self.device)

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)

        # 训练状态
        self.global_step = 0
        self.best_f1 = 0.0
        self.training_history: List[Dict[str, Any]] = []

    def _get_device(self) -> torch.device:
        """获取计算设备"""
        device_str = self.model_config.device

        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_str)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器 (可选)
            epochs: 训练轮数 (可选，使用配置中的值)

        Returns:
            训练结果字典
        """
        epochs = epochs or self.training_config.epochs
        config = self.training_config

        # 创建输出目录
        output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)

        # 优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # 混合精度训练
        scaler = GradScaler() if config.fp16 and self.device.type == "cuda" else None

        logger.info(f"开始训练: {epochs} epochs, {len(train_dataloader)} batches/epoch")

        for epoch in range(epochs):
            # 训练一个 epoch
            train_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler, scaler, epoch + 1
            )

            # 验证
            val_result = None
            if val_dataloader is not None:
                val_result = self.evaluate(val_dataloader)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val F1: {val_result.f1_macro:.4f}, "
                    f"Val Acc: {val_result.accuracy:.4f}"
                )

                # 保存最佳模型
                if val_result.f1_macro > self.best_f1:
                    self.best_f1 = val_result.f1_macro
                    best_path = os.path.join(output_dir, "best_model.pt")
                    self.model.save(best_path)
                    logger.info(f"保存最佳模型到 {best_path}")
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

            # 记录历史
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_metrics": val_result.to_dict() if val_result else None
            })

        # 保存最终模型
        final_path = os.path.join(output_dir, "final_model.pt")
        self.model.save(final_path)
        logger.info(f"保存最终模型到 {final_path}")

        return {
            "output_dir": output_dir,
            "best_f1": self.best_f1,
            "history": self.training_history
        }

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: Optional[GradScaler],
        epoch: int
    ) -> float:
        """
        训练一个 epoch

        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        config = self.training_config

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播
            if scaler is not None:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs["loss"] / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"] / config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * config.gradient_accumulation_steps

            # 梯度累积
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item() * config.gradient_accumulation_steps)

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> MetricResult:
        """
        评估模型

        Args:
            dataloader: 验证/测试数据加载器

        Returns:
            评估结果
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                predictions, _, _ = self.model.predict(input_ids, attention_mask)

                all_preds.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        return compute_metrics(all_labels, all_preds)

    def save_tokenizer(self, path: str):
        """保存分词器"""
        self.tokenizer.save_pretrained(path)
