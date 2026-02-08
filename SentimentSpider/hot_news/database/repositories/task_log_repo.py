# =====================================================
# Hot News Module - Task Log Repository
# =====================================================

from typing import List
from datetime import datetime
from ..connection import get_db


class TaskLogRepository:
    """任务日志数据访问层"""

    def __init__(self):
        self.db = get_db()

    def start_task(self, task_name: str) -> int:
        """开始任务"""
        sql = """
            INSERT INTO task_execution_log 
            (task_name, status, started_at)
            VALUES (%s, 'running', NOW())
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (task_name,))
            cursor.execute("SELECT LAST_INSERT_ID()")
            return cursor.fetchone()[0]

    def complete_task(
        self,
        log_id: int,
        status: str,
        hot_count: int = 0,
        matched_count: int = 0,
        keyword_count: int = 0,
        crawl_triggered: int = 0,
        error_message: str = None,
    ) -> bool:
        """完成任务"""
        sql = """
            UPDATE task_execution_log 
            SET status = %s, hot_count = %s, matched_count = %s, 
                keyword_count = %s, crawl_triggered = %s, 
                error_message = %s, completed_at = NOW()
            WHERE id = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql,
                (
                    status,
                    hot_count,
                    matched_count,
                    keyword_count,
                    crawl_triggered,
                    error_message,
                    log_id,
                ),
            )
            return cursor.rowcount > 0

    def get_recent(self, task_name: str = None, limit: int = 20) -> List[dict]:
        """获取最近的日志"""
        sql = "SELECT * FROM task_execution_log"
        params = []
        if task_name:
            sql += " WHERE task_name = %s"
            params.append(task_name)
        sql += " ORDER BY started_at DESC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_running(self, task_name: str = None) -> List[dict]:
        """获取运行中的任务"""
        sql = "SELECT * FROM task_execution_log WHERE status = 'running'"
        params = []
        if task_name:
            sql += " AND task_name = %s"
            params.append(task_name)

        with self.db.get_cursor(commit=False) as cursor:
            if params:
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def cleanup_old(self, days: int = 7) -> int:
        """清理旧日志"""
        sql = "DELETE FROM task_execution_log WHERE started_at < DATE_SUB(NOW(), INTERVAL %s DAY)"
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (days,))
            return cursor.rowcount
