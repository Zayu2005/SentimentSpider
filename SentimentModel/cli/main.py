# -*- coding: utf-8 -*-
"""
命令行工具

提供模型训练、评估、预测等命令
"""

import argparse
import logging
import sys

from ..config import get_settings, LABEL_NAMES_CN


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_stats(args):
    """统计信息命令"""
    from ..database import SentimentContentRepo, SentimentCommentRepo

    print("\n" + "=" * 60)
    print("情感分析统计信息")
    print("=" * 60)

    # 内容统计
    content_total = SentimentContentRepo.count_all()
    content_analyzed = SentimentContentRepo.count_analyzed()
    content_unanalyzed = SentimentContentRepo.count_unanalyzed()

    print(f"\n📝 内容统计:")
    print(f"   总数:       {content_total}")
    print(f"   已分析:     {content_analyzed}")
    print(f"   待分析:     {content_unanalyzed}")

    # 评论统计
    comment_total = SentimentCommentRepo.count_all()
    comment_analyzed = SentimentCommentRepo.count_analyzed()
    comment_unanalyzed = SentimentCommentRepo.count_unanalyzed()

    print(f"\n💬 评论统计:")
    print(f"   总数:       {comment_total}")
    print(f"   已分析:     {comment_analyzed}")
    print(f"   待分析:     {comment_unanalyzed}")

    # 情感分布
    if content_analyzed > 0:
        print(f"\n📊 内容情感分布:")
        stats = SentimentContentRepo.get_sentiment_stats()
        for row in stats:
            print(f"   {row['platform']} - {row['sentiment']}: {row['count']} (平均分: {row['avg_score']:.2f})")

    if comment_analyzed > 0:
        print(f"\n📊 评论情感分布:")
        stats = SentimentCommentRepo.get_sentiment_stats()
        for row in stats:
            print(f"   {row['platform']} - {row['sentiment']}: {row['count']} (平均分: {row['avg_score']:.2f})")

    print("\n" + "=" * 60)


def cmd_train(args):
    """训练命令"""
    from transformers import AutoTokenizer
    from ..models import BertSentimentClassifier
    from ..training import SentimentTrainer
    from ..data import load_public_dataset, split_dataset, create_dataloader
    from ..config import get_settings

    settings = get_settings()

    print(f"\n开始训练模型...")
    print(f"数据集: {args.dataset}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"输出目录: {args.output}")

    # 加载数据集
    print(f"\n加载数据集...")
    texts, labels = load_public_dataset(args.dataset)
    print(f"共 {len(texts)} 条数据")

    # 划分数据集
    splits = split_dataset(texts, labels)
    print(f"训练集: {len(splits['train'][0])}, 验证集: {len(splits['val'][0])}, 测试集: {len(splits['test'][0])}")

    # 加载分词器
    print(f"\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(settings.model.name)

    # 创建数据加载器
    train_loader = create_dataloader(
        splits["train"][0], splits["train"][1],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=settings.model.max_length,
        shuffle=True
    )
    val_loader = create_dataloader(
        splits["val"][0], splits["val"][1],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=settings.model.max_length,
        shuffle=False
    )

    # 创建模型和训练器
    print(f"\n创建模型...")
    model = BertSentimentClassifier(
        model_name=settings.model.name,
        num_labels=settings.model.num_labels,
        dropout=settings.model.dropout
    )

    # 更新训练配置
    settings.training.epochs = args.epochs
    settings.training.batch_size = args.batch_size
    settings.training.output_dir = args.output

    trainer = SentimentTrainer(model=model)

    # 训练
    print(f"\n开始训练...")
    result = trainer.train(train_loader, val_loader, epochs=args.epochs)

    print(f"\n训练完成!")
    print(f"最佳 F1: {result['best_f1']:.4f}")
    print(f"模型保存到: {result['output_dir']}")


def cmd_evaluate(args):
    """评估命令"""
    from transformers import AutoTokenizer
    from ..models import BertSentimentClassifier
    from ..training import SentimentTrainer
    from ..data import load_public_dataset, split_dataset, create_dataloader
    from ..config import get_settings

    settings = get_settings()

    print(f"\n评估模型: {args.model}")

    # 加载模型
    model = BertSentimentClassifier.load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    # 加载测试数据
    if args.test_data:
        import pandas as pd
        df = pd.read_csv(args.test_data)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
    else:
        # 使用默认数据集的测试集
        texts, labels = load_public_dataset(args.dataset)
        splits = split_dataset(texts, labels)
        texts, labels = splits["test"]

    print(f"测试数据: {len(texts)} 条")

    # 创建数据加载器
    test_loader = create_dataloader(
        texts, labels,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=settings.model.max_length,
        shuffle=False
    )

    # 评估
    trainer = SentimentTrainer(model=model)
    result = trainer.evaluate(test_loader)

    print(result.summary())


def cmd_analyze(args):
    """分析数据库数据命令"""
    from ..inference import SentimentPredictor, analyze_content, analyze_comment

    print(f"\n开始情感分析...")
    print(f"模型: {args.model}")
    print(f"批次大小: {args.batch_size}")
    if args.dry_run:
        print("[试运行模式]")

    # 创建预测器
    predictor = SentimentPredictor(model_path=args.model)

    # 分析内容
    if args.type in ["all", "content"]:
        print(f"\n分析内容...")
        stats = analyze_content(predictor, batch_size=args.batch_size, dry_run=args.dry_run)
        print(f"内容分析完成: 总数 {stats['total']}, 成功 {stats['success']}, 失败 {stats['error']}")

    # 分析评论
    if args.type in ["all", "comment"]:
        print(f"\n分析评论...")
        stats = analyze_comment(predictor, batch_size=args.batch_size, dry_run=args.dry_run)
        print(f"评论分析完成: 总数 {stats['total']}, 成功 {stats['success']}, 失败 {stats['error']}")

    print(f"\n分析完成!")


def cmd_predict(args):
    """预测单条文本命令"""
    from ..inference import SentimentPredictor

    predictor = SentimentPredictor(model_path=args.model)
    result = predictor.predict(args.text)

    print(f"\n输入文本: {args.text}")
    print(f"\n预测结果:")
    print(f"  情感: {result.label} ({LABEL_NAMES_CN[result.label_id]})")
    print(f"  分数: {result.score:.4f}")
    print(f"  概率分布:")
    for label, prob in result.probs.items():
        cn_name = LABEL_NAMES_CN[["negative", "neutral", "positive"].index(label)]
        print(f"    {cn_name} ({label}): {prob:.4f}")


def cmd_download(args):
    """下载预训练模型命令"""
    from transformers import AutoModel, AutoTokenizer

    print(f"\n下载预训练模型: {args.model}")

    print("下载模型...")
    AutoModel.from_pretrained(args.model)

    print("下载分词器...")
    AutoTokenizer.from_pretrained(args.model)

    print(f"\n下载完成! 模型已缓存到本地。")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="SentimentModel",
        description="中文情感分析模型"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # stats 命令
    stats_parser = subparsers.add_parser("stats", help="查看统计信息")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--dataset", "-d",
        default="weibo_senti_100k",
        help="训练数据集 (默认: weibo_senti_100k)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="批次大小 (默认: 32)"
    )
    train_parser.add_argument(
        "--output", "-o",
        default="models",
        help="输出目录 (默认: models)"
    )

    # evaluate 命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型路径"
    )
    eval_parser.add_argument(
        "--test-data",
        help="测试数据 CSV 文件"
    )
    eval_parser.add_argument(
        "--dataset",
        default="weibo_senti_100k",
        help="使用的数据集"
    )
    eval_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="批次大小"
    )

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="分析数据库中的数据")
    analyze_parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型路径"
    )
    analyze_parser.add_argument(
        "--type", "-t",
        choices=["all", "content", "comment"],
        default="all",
        help="分析类型 (默认: all)"
    )
    analyze_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="批次大小 (默认: 100)"
    )
    analyze_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行 (不保存结果)"
    )

    # predict 命令
    predict_parser = subparsers.add_parser("predict", help="预测单条文本")
    predict_parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型路径"
    )
    predict_parser.add_argument(
        "--text", "-t",
        required=True,
        help="要预测的文本"
    )

    # download 命令
    download_parser = subparsers.add_parser("download", help="下载预训练模型")
    download_parser.add_argument(
        "--model", "-m",
        default="hfl/chinese-roberta-wwm-ext-base",
        help="模型名称 (默认: hfl/chinese-roberta-wwm-ext-base)"
    )

    return parser


# 命令映射
COMMANDS = {
    "stats": cmd_stats,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "analyze": cmd_analyze,
    "predict": cmd_predict,
    "download": cmd_download,
}


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command in COMMANDS:
        try:
            COMMANDS[args.command](args)
        except KeyboardInterrupt:
            print("\n操作已取消")
            sys.exit(1)
        except Exception as e:
            logger.error(f"执行失败: {e}")
            raise
    else:
        parser.print_help()
        sys.exit(1)


# 为了兼容性
app = main


if __name__ == "__main__":
    main()
