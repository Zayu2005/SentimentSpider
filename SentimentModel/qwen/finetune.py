# -*- coding: utf-8 -*-
"""
Qwen2.5 情感分析微调脚本

使用方法:
    # 1. 准备数据
    python -m SentimentModel.qwen.finetune prepare --max-samples 10000

    # 2. 训练模型
    python -m SentimentModel.qwen.finetune train --epochs 3

    # 3. 测试模型
    python -m SentimentModel.qwen.finetune test --model models/qwen_sentiment/xxx/final_model

    # 4. 合并权重 (可选)
    python -m SentimentModel.qwen.finetune merge --adapter models/qwen_sentiment/xxx/final_model --output models/qwen_merged
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_prepare(args):
    """准备数据命令"""
    from .data_prepare import prepare_sentiment_data

    logger.info("准备微调数据...")

    paths = prepare_sentiment_data(
        input_file=args.input,
        output_dir=args.output,
        max_samples=args.max_samples,
        seed=args.seed
    )

    print(f"\n数据准备完成:")
    for split, path in paths.items():
        print(f"  {split}: {path}")


def cmd_train(args):
    """训练命令"""
    from .trainer import QwenSentimentTrainer, QwenTrainingConfig

    # 检查数据是否存在
    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "val.json")

    if not os.path.exists(train_path):
        logger.error(f"训练数据不存在: {train_path}")
        logger.info("请先运行: python -m SentimentModel.qwen.finetune prepare")
        sys.exit(1)

    # 配置
    config = QwenTrainingConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
    )

    logger.info("训练配置:")
    logger.info(f"  模型: {config.model_name}")
    logger.info(f"  轮数: {config.num_train_epochs}")
    logger.info(f"  批次大小: {config.per_device_train_batch_size}")
    logger.info(f"  梯度累积: {config.gradient_accumulation_steps}")
    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  LoRA r: {config.lora_r}")
    logger.info(f"  4bit 量化: {config.use_4bit}")

    # 训练
    trainer = QwenSentimentTrainer(config)
    result = trainer.train(
        train_data_path=train_path,
        val_data_path=val_path if os.path.exists(val_path) else None
    )

    print(f"\n训练完成!")
    print(f"  输出目录: {result['output_dir']}")
    print(f"  训练损失: {result['train_loss']:.4f}")
    print(f"  模型路径: {result['final_model_path']}")


def cmd_test(args):
    """测试命令"""
    from .predictor import QwenSentimentPredictor

    logger.info(f"加载模型: {args.model}")

    predictor = QwenSentimentPredictor(
        model_path=args.model,
        base_model_name=args.base_model,
        is_merged=args.merged,
        load_in_4bit=args.load_4bit
    )

    # 测试文本
    if args.text:
        texts = [args.text]
    else:
        texts = [
            "这个产品真的太好用了！强烈推荐！",
            "我很喜欢这个，感觉很满意",
            "太棒了，超出预期",
            "好开心，终于买到了",
            "垃圾产品，千万别买",
            "太差了，非常失望",
            "不好用，退货了",
            "很失望，和描述不符",
            "还行吧，一般般",
            "没什么特别的感觉",
        ]

    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    for text in texts:
        result = predictor.predict(text)
        print(f"\n文本: {text}")
        print(f"  情感: {result.sentiment}")
        print(f"  分数: {result.sentiment_score:.2f}")
        print(f"  情绪: {', '.join(result.emotion_tags)}")


def cmd_merge(args):
    """合并权重命令"""
    from .trainer import QwenSentimentTrainer, QwenTrainingConfig

    config = QwenTrainingConfig(model_name=args.base_model)
    trainer = QwenSentimentTrainer(config)
    trainer.merge_and_save(args.adapter, args.output)

    print(f"\n合并完成! 模型保存到: {args.output}")


def cmd_analyze(args):
    """分析数据库数据命令"""
    from .predictor import analyze_with_qwen

    logger.info("开始分析数据库数据...")

    stats = analyze_with_qwen(
        model_path=args.model,
        base_model_name=args.base_model,
        is_merged=args.merged,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

    print(f"\n分析完成!")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5 情感分析微调工具"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # prepare 命令
    prepare_parser = subparsers.add_parser("prepare", help="准备微调数据")
    prepare_parser.add_argument("--input", help="输入 CSV 文件 (可选，默认从 HuggingFace 下载)")
    prepare_parser.add_argument("--output", default="data/qwen_finetune", help="输出目录")
    prepare_parser.add_argument("--max-samples", type=int, help="最大样本数")
    prepare_parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型")
    train_parser.add_argument("--data-dir", default="data/qwen_finetune", help="数据目录")
    train_parser.add_argument("--output", default="models/qwen_sentiment", help="输出目录")
    train_parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    train_parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    train_parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    train_parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    train_parser.add_argument("--no-4bit", action="store_true", help="禁用 4bit 量化")

    # test 命令
    test_parser = subparsers.add_parser("test", help="测试模型")
    test_parser.add_argument("--model", required=True, help="模型路径")
    test_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型")
    test_parser.add_argument("--merged", action="store_true", help="是否是合并后的模型")
    test_parser.add_argument("--load-4bit", action="store_true", help="使用 4bit 加载")
    test_parser.add_argument("--text", help="测试文本 (可选)")

    # merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并 LoRA 权重")
    merge_parser.add_argument("--adapter", required=True, help="LoRA 适配器路径")
    merge_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型")
    merge_parser.add_argument("--output", required=True, help="输出路径")

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="分析数据库数据")
    analyze_parser.add_argument("--model", required=True, help="模型路径")
    analyze_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型")
    analyze_parser.add_argument("--merged", action="store_true", help="是否是合并后的模型")
    analyze_parser.add_argument("--batch-size", type=int, default=50, help="批次大小")
    analyze_parser.add_argument("--dry-run", action="store_true", help="试运行")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "prepare": cmd_prepare,
        "train": cmd_train,
        "test": cmd_test,
        "merge": cmd_merge,
        "analyze": cmd_analyze,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
