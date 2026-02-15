# -*- coding: utf-8 -*-
"""
命令行工具

知识图谱构建的各种操作命令
"""

import argparse
import logging
import sys

from ..config import get_kg_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_extract(args):
    """实体关系抽取命令"""
    from ..extraction import EntityRelationExtractor

    mode = getattr(args, 'mode', 'quick') or 'quick'
    print(f"\n开始实体关系抽取 (OneKE {mode} 模式)...")
    print(f"话题ID: {args.topic_id}")
    if args.limit:
        print(f"最大内容数: {args.limit}")
    if args.dry_run:
        print("[试运行模式]")

    extractor = EntityRelationExtractor(mode=mode)
    result = extractor.extract_for_topic(
        topic_id=args.topic_id,
        limit=args.limit or 200,
        dry_run=args.dry_run,
    )

    print(f"\n抽取结果:")
    print(f"  内容总数:   {result.total_content}")
    print(f"  成功抽取:   {result.extracted}")
    print(f"  空结果:     {result.empty}")
    print(f"  失败:       {result.failed}")
    print(f"  跳过(已有): {result.skipped}")
    print(f"  实体总数:   {result.total_entities}")
    print(f"  关系总数:   {result.total_relations}")
    # OneKE Pipeline 不直接暴露 token 用量


def cmd_build(args):
    """构建 Neo4j 图命令"""
    from ..graph import GraphBuilder

    print(f"\n开始构建知识图谱...")
    print(f"话题ID: {args.topic_id}")
    if args.clear:
        print("[清除已有图数据]")

    builder = GraphBuilder()
    result = builder.build_for_topic(
        topic_id=args.topic_id,
        clear_existing=args.clear,
    )

    print(f"\n构建结果:")
    print(f"  原始实体:   {result.raw_entities}")
    print(f"  去重实体:   {result.deduplicated_entities}")
    print(f"  原始关系:   {result.raw_relations}")
    print(f"  去重关系:   {result.merged_relations}")
    print(f"  Neo4j节点:  {result.neo4j_nodes_created}")
    print(f"  Neo4j关系:  {result.neo4j_rels_created}")


def cmd_pipeline(args):
    """完整流水线: 抽取 + 构建"""
    mode = getattr(args, 'mode', 'quick') or 'quick'
    print(f"\n{'='*50}")
    print(f"知识图谱完整流水线 (OneKE {mode} 模式)")
    print(f"{'='*50}")
    print(f"话题ID: {args.topic_id}")

    # Step 1: 抽取
    print(f"\n[Step 1/2] 实体关系抽取 (OneKE {mode} 模式)")
    from ..extraction import EntityRelationExtractor

    extractor = EntityRelationExtractor(mode=mode)
    ext_result = extractor.extract_for_topic(
        topic_id=args.topic_id,
        limit=args.limit or 200,
        dry_run=args.dry_run,
    )
    print(
        f"  抽取: {ext_result.extracted} 成功, "
        f"{ext_result.total_entities} 实体, "
        f"{ext_result.total_relations} 关系"
    )

    if args.dry_run:
        print(f"\n[试运行模式] 跳过图构建")
        return

    # Step 2: 构建
    print(f"\n[Step 2/2] Neo4j 图构建")
    from ..graph import GraphBuilder

    builder = GraphBuilder()
    build_result = builder.build_for_topic(
        topic_id=args.topic_id,
        clear_existing=args.clear,
    )
    print(
        f"  构建: {build_result.neo4j_nodes_created} 节点, "
        f"{build_result.neo4j_rels_created} 关系"
    )

    print(f"\n{'='*50}")
    print(f"流水线完成!")
    print(f"{'='*50}")


def cmd_query(args):
    """查询图谱信息"""
    from ..graph import GraphBuilder

    builder = GraphBuilder()
    info = builder.query_topic_graph(args.topic_id)

    print(f"\n{'='*60}")
    print(f"话题 {args.topic_id} 知识图谱信息")
    print(f"{'='*60}")

    node_stats = info.get("node_stats", [])
    if node_stats:
        print(f"\n实体类型分布:")
        for row in node_stats:
            examples = ", ".join(row.get("examples", [])[:3])
            print(f"  {row['type']:15s} {row['cnt']:4d} 个  (如: {examples})")

    rel_stats = info.get("rel_stats", [])
    if rel_stats:
        print(f"\n关系类型分布:")
        for row in rel_stats:
            print(f"  {row['type']:20s} {row['cnt']:4d} 条")

    top_entities = info.get("top_entities", [])
    if top_entities:
        print(f"\n高提及实体 Top 10:")
        for row in top_entities:
            print(
                f"  {row['name']:20s} [{row['type']:12s}] "
                f"提及 {row['mentions']} 次"
            )

    top_relations = info.get("top_relations", [])
    if top_relations:
        print(f"\n关键关系 Top 10:")
        for row in top_relations:
            print(
                f"  {row['head']} --[{row['relation']}]--> {row['tail']}  "
                f"(置信度: {row['confidence']:.2f}, 来源: {row['sources']})"
            )

    if not node_stats and not rel_stats:
        print(f"\n话题 {args.topic_id} 暂无知识图谱数据")

    print(f"\n{'='*60}")


def cmd_stats(args):
    """全局统计信息"""
    from ..database import KGExtractionRepo, KGBuildLogRepo

    print(f"\n{'='*60}")
    print(f"知识图谱全局统计")
    print(f"{'='*60}")

    ext_stats = KGExtractionRepo.get_stats()
    if ext_stats:
        print(f"\n抽取统计:")
        print(f"  涉及话题:   {ext_stats.get('topic_count') or 0}")
        print(f"  总抽取数:   {ext_stats.get('total_extractions') or 0}")
        print(f"  成功:       {ext_stats.get('completed') or 0}")
        print(f"  失败:       {ext_stats.get('failed') or 0}")
        print(f"  空结果:     {ext_stats.get('empty_results') or 0}")
        print(f"  总Token:    {ext_stats.get('total_tokens') or 0}")

    build_stats = KGBuildLogRepo.get_stats()
    if build_stats:
        print(f"\n构建统计:")
        print(f"  已构建话题: {build_stats.get('topics_built') or 0}")
        print(f"  总实体数:   {build_stats.get('total_entities') or 0}")
        print(f"  总关系数:   {build_stats.get('total_relations') or 0}")
        print(f"  成功:       {build_stats.get('success_count') or 0}")
        print(f"  失败:       {build_stats.get('failed_count') or 0}")

    print(f"\n{'='*60}")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="KnowledgeGraph",
        description="知识图谱构建 - 实体关系抽取 + Neo4j 图存储",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # extract 命令
    ext_parser = subparsers.add_parser("extract", help="实体关系抽取")
    ext_parser.add_argument(
        "--topic-id", "-t", type=int, required=True,
        help="话题ID"
    )
    ext_parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="最大内容数 (默认: 200)"
    )
    ext_parser.add_argument(
        "--mode", "-m", choices=["quick", "standard"], default="quick",
        help="OneKE 模式: quick(快速) / standard(含Reflection) (默认: quick)"
    )
    ext_parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行 (不保存结果)"
    )

    # build 命令
    build_parser = subparsers.add_parser("build", help="构建Neo4j图")
    build_parser.add_argument(
        "--topic-id", "-t", type=int, required=True,
        help="话题ID"
    )
    build_parser.add_argument(
        "--clear", action="store_true",
        help="清除已有图数据后重建"
    )

    # pipeline 命令
    pipe_parser = subparsers.add_parser(
        "pipeline", help="完整流水线(抽取+构建)"
    )
    pipe_parser.add_argument(
        "--topic-id", "-t", type=int, required=True,
        help="话题ID"
    )
    pipe_parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="最大内容数 (默认: 200)"
    )
    pipe_parser.add_argument(
        "--mode", "-m", choices=["quick", "standard"], default="quick",
        help="OneKE 模式: quick(快速) / standard(含Reflection) (默认: quick)"
    )
    pipe_parser.add_argument(
        "--clear", action="store_true",
        help="清除已有图数据后重建"
    )
    pipe_parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行 (仅抽取, 不构建)"
    )

    # query 命令
    query_parser = subparsers.add_parser("query", help="查询图谱信息")
    query_parser.add_argument(
        "--topic-id", "-t", type=int, required=True,
        help="话题ID"
    )

    # stats 命令
    subparsers.add_parser("stats", help="全局统计信息")

    return parser


# 命令映射
COMMANDS = {
    "extract": cmd_extract,
    "build": cmd_build,
    "pipeline": cmd_pipeline,
    "query": cmd_query,
    "stats": cmd_stats,
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


if __name__ == "__main__":
    main()
