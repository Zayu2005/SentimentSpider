# =====================================================
# Hot News Module - Unified Data Sync
# 统一数据同步 - 将各平台数据同步到统一表
# =====================================================

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import asdict

from ..database.connection import get_async_db_connection
from .adapters import ContentAdapter, CommentAdapter, UnifiedContent, UnifiedComment


class UnifiedDataSync:
    """
    统一数据同步器

    功能：
    1. 从各平台原始表读取数据
    2. 通过适配器转换为统一格式
    3. 写入统一内容表和统一评论表
    """

    def __init__(self):
        self.content_adapter = ContentAdapter
        self.comment_adapter = CommentAdapter

    async def sync_all_platforms(
        self,
        platforms: Optional[List[str]] = None,
        sync_comments: bool = True,
        batch_size: int = 500,
        incremental: bool = True,
        since_ts: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        同步所有平台数据

        Args:
            platforms: 要同步的平台列表，None表示全部
            sync_comments: 是否同步评论
            batch_size: 批量处理大小
            incremental: 是否增量同步（只同步新数据）
            since_ts: 指定起始时间戳（毫秒），只同步此时间之后的数据

        Returns:
            各平台同步数量统计
        """
        if platforms is None:
            platforms = list(ContentAdapter.PLATFORM_TABLES.keys())

        stats = {}
        for platform in platforms:
            content_count = await self.sync_platform_content(
                platform, batch_size, incremental, since_ts
            )
            stats[f"{platform}_content"] = content_count

            if sync_comments:
                comment_count = await self.sync_platform_comments(
                    platform, batch_size, incremental, since_ts
                )
                stats[f"{platform}_comment"] = comment_count

        return stats

    async def sync_platform_content(
        self,
        platform: str,
        batch_size: int = 500,
        incremental: bool = True,
        since_ts: Optional[int] = None,
    ) -> int:
        """
        同步单个平台的内容数据

        Args:
            platform: 平台代码
            batch_size: 批量大小
            incremental: 增量同步
            since_ts: 指定起始时间戳（毫秒），优先于incremental

        Returns:
            同步的记录数
        """
        source_table = ContentAdapter.PLATFORM_TABLES.get(platform)
        if not source_table:
            print(f"[UnifiedSync] 未知平台: {platform}")
            return 0

        async with get_async_db_connection() as conn:
            async with conn.cursor() as cursor:
                # 确定同步起始时间戳
                if since_ts is not None:
                    # 使用指定的时间戳
                    last_sync_ts = since_ts
                elif incremental:
                    # 增量同步：从统一表中获取最新时间戳
                    await cursor.execute(
                        """
                        SELECT COALESCE(MAX(add_ts), 0)
                        FROM unified_content
                        WHERE platform = %s
                        """,
                        (platform,),
                    )
                    result = await cursor.fetchone()
                    last_sync_ts = result[0] if result and result[0] else 0
                else:
                    # 全量同步
                    last_sync_ts = 0

                # 查询源表数据
                query = f"""
                    SELECT * FROM {source_table}
                    WHERE add_ts > %s
                    ORDER BY add_ts ASC
                """
                await cursor.execute(query, (last_sync_ts,))

                total_synced = 0
                while True:
                    rows = await cursor.fetchmany(batch_size)
                    if not rows:
                        break

                    # 获取列名
                    columns = [desc[0] for desc in cursor.description]

                    # 转换并插入
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        try:
                            unified = self.content_adapter.adapt(platform, row_dict)
                            await self._upsert_content(conn, unified)
                            total_synced += 1
                        except Exception as e:
                            print(f"[UnifiedSync] 转换失败 {platform}: {e}")
                            continue

                    await conn.commit()
                    print(f"[UnifiedSync] {platform} 内容已同步 {total_synced} 条")

                return total_synced

    async def sync_platform_comments(
        self,
        platform: str,
        batch_size: int = 500,
        incremental: bool = True,
        since_ts: Optional[int] = None,
    ) -> int:
        """
        同步单个平台的评论数据

        Args:
            platform: 平台代码
            batch_size: 批量大小
            incremental: 增量同步
            since_ts: 指定起始时间戳（毫秒），优先于incremental

        Returns:
            同步的记录数
        """
        source_table = CommentAdapter.PLATFORM_TABLES.get(platform)
        if not source_table:
            print(f"[UnifiedSync] 未知评论表: {platform}")
            return 0

        async with get_async_db_connection() as conn:
            async with conn.cursor() as cursor:
                # 确定同步起始时间戳
                if since_ts is not None:
                    # 使用指定的时间戳
                    last_sync_ts = since_ts
                elif incremental:
                    # 增量同步：从统一表中获取最新时间戳
                    await cursor.execute(
                        """
                        SELECT COALESCE(MAX(add_ts), 0)
                        FROM unified_comment
                        WHERE platform = %s
                        """,
                        (platform,),
                    )
                    result = await cursor.fetchone()
                    last_sync_ts = result[0] if result and result[0] else 0
                else:
                    # 全量同步
                    last_sync_ts = 0

                # 查询源表数据
                query = f"""
                    SELECT * FROM {source_table}
                    WHERE add_ts > %s
                    ORDER BY add_ts ASC
                """
                await cursor.execute(query, (last_sync_ts,))

                total_synced = 0
                while True:
                    rows = await cursor.fetchmany(batch_size)
                    if not rows:
                        break

                    columns = [desc[0] for desc in cursor.description]

                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        try:
                            unified = self.comment_adapter.adapt(platform, row_dict)
                            await self._upsert_comment(conn, unified)
                            total_synced += 1
                        except Exception as e:
                            print(f"[UnifiedSync] 评论转换失败 {platform}: {e}")
                            continue

                    await conn.commit()
                    print(f"[UnifiedSync] {platform} 评论已同步 {total_synced} 条")

                return total_synced

    async def _upsert_content(self, conn, content: UnifiedContent):
        """插入或更新统一内容"""
        async with conn.cursor() as cursor:
            sql = """
                INSERT INTO unified_content (
                    platform, content_id, content_type,
                    user_id, nickname, avatar, ip_location, gender,
                    title, content, content_url,
                    media_type, cover_url, video_url, video_download_url,
                    image_list, music_url, tag_list,
                    liked_count, comment_count, share_count, collect_count,
                    view_count, coin_count, danmaku_count,
                    source_keyword, original_created_at, add_ts
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    content_type = VALUES(content_type),
                    nickname = VALUES(nickname),
                    avatar = VALUES(avatar),
                    ip_location = VALUES(ip_location),
                    title = VALUES(title),
                    content = VALUES(content),
                    media_type = VALUES(media_type),
                    cover_url = VALUES(cover_url),
                    video_url = VALUES(video_url),
                    video_download_url = VALUES(video_download_url),
                    image_list = VALUES(image_list),
                    music_url = VALUES(music_url),
                    tag_list = VALUES(tag_list),
                    liked_count = VALUES(liked_count),
                    comment_count = VALUES(comment_count),
                    share_count = VALUES(share_count),
                    collect_count = VALUES(collect_count),
                    view_count = VALUES(view_count),
                    coin_count = VALUES(coin_count),
                    danmaku_count = VALUES(danmaku_count),
                    updated_at = CURRENT_TIMESTAMP
            """
            await cursor.execute(
                sql,
                (
                    content.platform,
                    content.content_id,
                    content.content_type,
                    content.user_id,
                    content.nickname,
                    content.avatar,
                    content.ip_location,
                    content.gender,
                    content.title,
                    content.content,
                    content.content_url,
                    content.media_type,
                    content.cover_url,
                    content.video_url,
                    content.video_download_url,
                    content.image_list,
                    content.music_url,
                    content.tag_list,
                    content.liked_count,
                    content.comment_count,
                    content.share_count,
                    content.collect_count,
                    content.view_count,
                    content.coin_count,
                    content.danmaku_count,
                    content.source_keyword,
                    content.original_created_at,
                    content.add_ts,
                ),
            )

    async def _upsert_comment(self, conn, comment: UnifiedComment):
        """插入或更新统一评论"""
        async with conn.cursor() as cursor:
            sql = """
                INSERT INTO unified_comment (
                    platform, comment_id, content_id, parent_comment_id,
                    user_id, nickname, avatar, ip_location, gender,
                    content, pictures,
                    liked_count, reply_count,
                    original_created_at, add_ts
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    pictures = VALUES(pictures),
                    liked_count = VALUES(liked_count),
                    reply_count = VALUES(reply_count),
                    updated_at = CURRENT_TIMESTAMP
            """
            await cursor.execute(
                sql,
                (
                    comment.platform,
                    comment.comment_id,
                    comment.content_id,
                    comment.parent_comment_id,
                    comment.user_id,
                    comment.nickname,
                    comment.avatar,
                    comment.ip_location,
                    comment.gender,
                    comment.content,
                    comment.pictures,
                    comment.liked_count,
                    comment.reply_count,
                    comment.original_created_at,
                    comment.add_ts,
                ),
            )

    async def get_sync_stats(self) -> Dict[str, Dict[str, int]]:
        """获取同步统计信息"""
        stats = {"content": {}, "comment": {}}

        async with get_async_db_connection() as conn:
            async with conn.cursor() as cursor:
                # 统一内容表统计
                await cursor.execute(
                    """
                    SELECT platform, COUNT(*) as count
                    FROM unified_content
                    GROUP BY platform
                    """
                )
                for row in await cursor.fetchall():
                    stats["content"][row[0]] = row[1]

                # 统一评论表统计
                await cursor.execute(
                    """
                    SELECT platform, COUNT(*) as count
                    FROM unified_comment
                    GROUP BY platform
                    """
                )
                for row in await cursor.fetchall():
                    stats["comment"][row[0]] = row[1]

        return stats

    async def full_resync(
        self,
        platforms: Optional[List[str]] = None,
        sync_comments: bool = True,
    ) -> Dict[str, int]:
        """
        全量重新同步（清空后重建）

        警告：这会删除统一表中的现有数据！
        """
        if platforms is None:
            platforms = list(ContentAdapter.PLATFORM_TABLES.keys())

        async with get_async_db_connection() as conn:
            async with conn.cursor() as cursor:
                for platform in platforms:
                    await cursor.execute(
                        "DELETE FROM unified_content WHERE platform = %s",
                        (platform,),
                    )
                    if sync_comments:
                        await cursor.execute(
                            "DELETE FROM unified_comment WHERE platform = %s",
                            (platform,),
                        )
                await conn.commit()

        return await self.sync_all_platforms(
            platforms=platforms,
            sync_comments=sync_comments,
            incremental=False,
        )
