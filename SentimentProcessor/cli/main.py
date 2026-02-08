# =====================================================
# SentimentProcessor - CLI
# 命令行工具
# =====================================================

import argparse
import logging
import sys

from ..processor import ContentProcessor, CommentProcessor


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_process_content(args):
    """处理内容命令"""
    processor = ContentProcessor()

    if args.stats:
        stats = processor.get_stats()
        print("\n=== 内容预处理统计 ===")
        print(f"总内容数: {stats['total_content']}")
        print(f"未处理数: {stats['unprocessed']}")
        print(f"已处理数: {stats['processed']}")
        print(f"  - 成功: {stats['completed']}")
        print(f"  - 失败: {stats['error']}")
        return

    result = processor.run(
        batch_size=args.batch_size,
        max_items=args.limit,
        save_to_db=not args.dry_run,
    )

    print("\n=== 处理结果 ===")
    print(f"总计处理: {result['total']}")
    print(f"成功: {result['success']}")
    print(f"失败: {result['error']}")


def cmd_process_comment(args):
    """处理评论命令"""
    processor = CommentProcessor()

    if args.stats:
        stats = processor.get_stats()
        print("\n=== 评论预处理统计 ===")
        print(f"总评论数: {stats['total_comment']}")
        print(f"未处理数: {stats['unprocessed']}")
        print(f"已处理数: {stats['processed']}")
        print(f"  - 成功: {stats['completed']}")
        print(f"  - 失败: {stats['error']}")
        return

    result = processor.run(
        batch_size=args.batch_size,
        max_items=args.limit,
        save_to_db=not args.dry_run,
    )

    print("\n=== 处理结果 ===")
    print(f"总计处理: {result['total']}")
    print(f"成功: {result['success']}")
    print(f"失败: {result['error']}")


def cmd_process_all(args):
    """处理所有数据命令"""
    print("=== 开始处理内容 ===")
    content_processor = ContentProcessor()
    content_result = content_processor.run(
        batch_size=args.batch_size,
        max_items=args.limit,
        save_to_db=not args.dry_run,
    )

    print("\n=== 开始处理评论 ===")
    comment_processor = CommentProcessor()
    comment_result = comment_processor.run(
        batch_size=args.batch_size,
        max_items=args.limit,
        save_to_db=not args.dry_run,
    )

    print("\n=== 总体结果 ===")
    print(f"内容: 处理 {content_result['total']}, 成功 {content_result['success']}, 失败 {content_result['error']}")
    print(f"评论: 处理 {comment_result['total']}, 成功 {comment_result['success']}, 失败 {comment_result['error']}")


def cmd_stats(args):
    """统计信息命令"""
    content_processor = ContentProcessor()
    comment_processor = CommentProcessor()

    content_stats = content_processor.get_stats()
    comment_stats = comment_processor.get_stats()

    print("\n" + "=" * 50)
    print("预处理统计信息")
    print("=" * 50)

    print("\n【内容】")
    print(f"  总数: {content_stats['total_content']}")
    print(f"  未处理: {content_stats['unprocessed']}")
    print(f"  已处理: {content_stats['processed']}")
    print(f"    - 成功: {content_stats['completed']}")
    print(f"    - 失败: {content_stats['error']}")

    print("\n【评论】")
    print(f"  总数: {comment_stats['total_comment']}")
    print(f"  未处理: {comment_stats['unprocessed']}")
    print(f"  已处理: {comment_stats['processed']}")
    print(f"    - 成功: {comment_stats['completed']}")
    print(f"    - 失败: {comment_stats['error']}")

    print("\n" + "=" * 50)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="SentimentProcessor - 情感分析数据预处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # content 子命令
    content_parser = subparsers.add_parser("content", help="处理内容")
    content_parser.add_argument("-b", "--batch-size", type=int, default=100, help="批处理大小")
    content_parser.add_argument("-l", "--limit", type=int, help="最大处理数量")
    content_parser.add_argument("--dry-run", action="store_true", help="试运行（不保存到数据库）")
    content_parser.add_argument("--stats", action="store_true", help="显示统计信息")
    content_parser.set_defaults(func=cmd_process_content)

    # comment 子命令
    comment_parser = subparsers.add_parser("comment", help="处理评论")
    comment_parser.add_argument("-b", "--batch-size", type=int, default=100, help="批处理大小")
    comment_parser.add_argument("-l", "--limit", type=int, help="最大处理数量")
    comment_parser.add_argument("--dry-run", action="store_true", help="试运行（不保存到数据库）")
    comment_parser.add_argument("--stats", action="store_true", help="显示统计信息")
    comment_parser.set_defaults(func=cmd_process_comment)

    # all 子命令
    all_parser = subparsers.add_parser("all", help="处理所有数据")
    all_parser.add_argument("-b", "--batch-size", type=int, default=100, help="批处理大小")
    all_parser.add_argument("-l", "--limit", type=int, help="最大处理数量")
    all_parser.add_argument("--dry-run", action="store_true", help="试运行（不保存到数据库）")
    all_parser.set_defaults(func=cmd_process_all)

    # stats 子命令
    stats_parser = subparsers.add_parser("stats", help="显示统计信息")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # 执行命令
    args.func(args)


if __name__ == "__main__":
    main()
