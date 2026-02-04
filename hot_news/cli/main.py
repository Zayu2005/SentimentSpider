# =====================================================
# Hot News Module - CLI Main Entry
# =====================================================

import typer
from typing import Optional
from pathlib import Path

from .commands.fetch import cmd_fetch
from .commands.analyze import cmd_analyze
from .commands.extract import cmd_extract
from .commands.crawl import cmd_crawl
from .commands.run import cmd_run
from .commands.config import cmd_config
from .commands.show import cmd_show

app = typer.Typer(name="hot-news", add_completion=False, help="热点新闻获取与分析模块")

app.add_typer(cmd_fetch, name="fetch", help="获取热点新闻")
app.add_typer(cmd_analyze, name="analyze", help="分析热点领域匹配")
app.add_typer(cmd_extract, name="extract", help="提取关键词")
app.add_typer(cmd_crawl, name="crawl", help="触发爬虫")
app.add_typer(cmd_run, name="run", help="一键执行完整流程")
app.add_typer(cmd_config, name="config", help="配置管理")
app.add_typer(cmd_show, name="show", help="查看数据")


@app.command()
def init_db():
    """初始化数据库表"""
    import sys
    from pathlib import Path

    sql_file = (
        Path(__file__).parent.parent / "database" / "migrations" / "001_initial.sql"
    )

    if not sql_file.exists():
        print(f"[Error] SQL文件不存在: {sql_file}")
        raise typer.Exit(1)

    from ..config.settings import get_settings
    import pymysql

    settings = get_settings()

    print(f"[Init] 连接数据库: {settings._db_config.host}:{settings._db_config.port}")

    try:
        conn = pymysql.connect(**settings._db_config.connection_kwargs)
        with conn.cursor() as cursor:
            with open(sql_file, "r", encoding="utf-8") as f:
                sql = f.read()
                for statement in sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
            conn.commit()
        conn.close()

        print("[Success] 数据库初始化完成！")

    except Exception as e:
        print(f"[Error] 初始化失败: {e}")
        raise typer.Exit(1)


def cli():
    """CLI入口点"""
    app()


if __name__ == "__main__":
    cli()
