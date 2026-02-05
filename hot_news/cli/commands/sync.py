# =====================================================
# Hot News Module - Sync Command
# 数据同步命令 - 将各平台数据同步到统一表
# =====================================================

import typer
import asyncio
from typing import List, Optional


def sync_data(
    platforms: Optional[List[str]] = None,
    no_comments: bool = False,
    full: bool = False,
    batch_size: int = 500,
):
    """
    同步各平台数据到统一表

    Args:
        platforms: 平台列表 (xhs/dy/wb/bili/ks)
        no_comments: 不同步评论
        full: 全量同步（清空后重建）
        batch_size: 批量大小
    """
    from hot_news.sync import UnifiedDataSync

    syncer = UnifiedDataSync()

    async def _run():
        if full:
            print("[Sync] 开始全量同步（将清空现有数据）...")
            stats = await syncer.full_resync(
                platforms=platforms,
                sync_comments=not no_comments,
            )
        else:
            print("[Sync] 开始增量同步...")
            stats = await syncer.sync_all_platforms(
                platforms=platforms,
                sync_comments=not no_comments,
                batch_size=batch_size,
                incremental=True,
            )

        print("\n[Sync] 同步完成！统计:")
        for key, count in stats.items():
            print(f"  - {key}: {count} 条")

        # 显示总体统计
        total_stats = await syncer.get_sync_stats()
        print("\n[Sync] 统一表总数据量:")
        print("  内容表:")
        for platform, count in total_stats["content"].items():
            print(f"    - {platform}: {count} 条")
        print("  评论表:")
        for platform, count in total_stats["comment"].items():
            print(f"    - {platform}: {count} 条")

    asyncio.run(_run())


def sync_stats():
    """查看同步统计信息"""
    from hot_news.sync import UnifiedDataSync

    syncer = UnifiedDataSync()

    async def _run():
        stats = await syncer.get_sync_stats()

        print("[Sync Stats] 统一表数据统计:")
        print("\n内容表 (unified_content):")
        total_content = 0
        for platform, count in stats["content"].items():
            print(f"  - {platform}: {count} 条")
            total_content += count
        print(f"  总计: {total_content} 条")

        print("\n评论表 (unified_comment):")
        total_comment = 0
        for platform, count in stats["comment"].items():
            print(f"  - {platform}: {count} 条")
            total_comment += count
        print(f"  总计: {total_comment} 条")

    asyncio.run(_run())
