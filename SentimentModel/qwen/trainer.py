# -*- coding: utf-8 -*-
"""
Qwen2.5 LoRA 微调训练器

使用 PEFT 库进行高效微调
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from .data_prepare import SYSTEM_PROMPT, load_finetune_data

logger = logging.getLogger(__name__)


@dataclass
class QwenTrainingConfig:
    """Qwen 训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 512

    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # 训练配置
    output_dir: str = "models/qwen_sentiment"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3

    # 其他
    seed: int = 42
    fp16: bool = False  # Qwen2.5 建议使用 bf16
    bf16: bool = True


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建对话格式
        messages = [
            {"role": "system", "content": item.get("system", SYSTEM_PROMPT)},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]}
        ]

        # 使用 apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # 创建标签 (对于因果语言模型，labels 等于 input_ids)
        labels = input_ids.clone()

        # 将 padding token 的标签设为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class QwenSentimentTrainer:
    """Qwen2.5 情感分析微调训练器"""

    def __init__(self, config: Optional[QwenTrainingConfig] = None):
        self.config = config or QwenTrainingConfig()
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def setup(self):
        """初始化模型和分词器"""
        logger.info(f"加载模型: {self.config.model_name}")

        # 量化配置
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
        )

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 准备模型进行量化训练
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.peft_model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        trainable_params, all_params = self._count_parameters()
        logger.info(
            f"可训练参数: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )

    def _count_parameters(self):
        """统计参数数量"""
        trainable_params = 0
        all_params = 0
        for _, param in self.peft_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return trainable_params, all_params

    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            resume_from_checkpoint: 恢复训练的检查点路径

        Returns:
            训练结果
        """
        if self.peft_model is None:
            self.setup()

        # 加载数据
        logger.info("加载训练数据...")
        train_data = load_finetune_data(train_data_path)
        train_dataset = SentimentDataset(
            train_data, self.tokenizer, self.config.max_length
        )

        val_dataset = None
        if val_data_path:
            val_data = load_finetune_data(val_data_path)
            val_dataset = SentimentDataset(
                val_data, self.tokenizer, self.config.max_length
            )

        logger.info(f"训练集: {len(train_dataset)} 条")
        if val_dataset:
            logger.info(f"验证集: {len(val_dataset)} 条")

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.eval_steps if val_dataset else None,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if val_dataset else False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            report_to="none",  # 禁用 wandb 等
            remove_unused_columns=False,
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        # 创建 Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        logger.info("开始训练...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 保存模型
        logger.info("保存模型...")
        final_model_path = output_dir / "final_model"
        self.peft_model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        # 保存训练配置
        config_path = output_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            config_dict = {
                k: v if not isinstance(v, list) else v
                for k, v in vars(self.config).items()
            }
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        result = {
            "output_dir": str(output_dir),
            "final_model_path": str(final_model_path),
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
        }

        logger.info(f"训练完成! 模型保存到: {final_model_path}")

        return result

    def merge_and_save(self, adapter_path: str, output_path: str):
        """
        合并 LoRA 权重并保存完整模型

        Args:
            adapter_path: LoRA 适配器路径
            output_path: 输出路径
        """
        from peft import PeftModel

        logger.info(f"加载基础模型: {self.config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        logger.info(f"加载 LoRA 适配器: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        logger.info("合并权重...")
        merged_model = model.merge_and_unload()

        logger.info(f"保存合并后的模型到: {output_path}")
        merged_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)

        logger.info("完成!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试训练
    config = QwenTrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_length=256,
    )

    trainer = QwenSentimentTrainer(config)
    result = trainer.train(
        train_data_path="data/qwen_finetune/train.json",
        val_data_path="data/qwen_finetune/val.json"
    )
    print(f"训练结果: {result}")
