# =====================================================
# Hot News Module - Show Command
# =====================================================

import typer
from typing import Optional
from hot_news.database import HotNewsRepository, KeywordRepository, TaskLogRepository

cmd_show = typer.Typer(help="查看数据")


@cmd_show.command("hot-news")
def show_hot_news(
    limit: int = typer.Option(20, "--limit", "-l", help="显示数量"),
    platform: Optional[str] = typer.Option(None, "--platform", "-p", help="平台筛选"),
):
    """查看热点新闻"""
    repo = HotNewsRepository()
    news_list = repo.get_recent(limit, platform)

    print(f"\n热点新闻 (共 {len(news_list)} 条):")
    print("-" * 80)
    for idx, news in enumerate(news_list, 1):
        title = news["title"][:40] + "..." if len(news["title"]) > 40 else news["title"]
        print(f"  {idx:2}. [{news['platform_code']:10}] {title}")
    print("-" * 80)


@cmd_show.command("keywords")
def show_keywords(
    limit: int = typer.Option(50, "--limit", "-l", help="显示数量"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="领域筛选"),
):
    """查看关键词"""
    from hot_news.config import get_settings

    settings = get_settings()
    domain_configs = settings.get_domains()

    repo = KeywordRepository()

    if domain:
        domain_id = next(
            (d.id for d in domain_configs if d.domain_name == domain), None
        )
        if domain_id:
            keywords = repo.get_by_domain(domain_id, limit)
        else:
            print(f"[Error] 未找到领域: {domain}")
            return
    else:
        keywords = repo.get_all(limit)

    print(f"\n关键词 (共 {len(keywords)} 条):")
    print("-" * 60)
    for idx, kw in enumerate(keywords, 1):
        print(
            f"  {idx:3}. {kw['keyword']:30} 搜索:{kw['search_count']} 置信度:{kw['confidence']}"
        )
    print("-" * 60)


@cmd_show.command("logs")
def show_logs(
    limit: int = typer.Option(10, "--limit", "-l", help="显示数量"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="任务名称筛选"),
):
    """查看任务日志"""
    repo = TaskLogRepository()
    logs = repo.get_recent(task, limit)

    print(f"\n任务执行日志 (共 {len(logs)} 条):")
    print("-" * 100)
    for log in logs:
        status = (
            "✓"
            if log["status"] == "success"
            else "✗"
            if log["status"] == "failed"
            else "▶"
        )
        started = log["started_at"].strftime("%m-%d %H:%M") if log["started_at"] else ""
        print(
            f"  {status} {log['task_name']:20} {started} | 热点:{log['hot_count']:3} 匹配:{log['matched_count']:3} 关键词:{log['keyword_count']:3} 爬取:{log['crawl_triggered']:3}"
        )
    print("-" * 100)


@cmd_show.command("matched")
def show_matched(
    domain: str = typer.Argument(..., help="领域名称"),
    limit: int = typer.Option(20, "--limit", "-l", help="显示数量"),
):
    """查看匹配的热点"""
    from hot_news.config import get_settings
    from hot_news.database import AnalysisRepository

    settings = get_settings()
    domain_configs = settings.get_domains()

    domain_config = next((d for d in domain_configs if d.domain_name == domain), None)
    if not domain_config:
        print(f"[Error] 未找到领域: {domain}")
        return

    repo = AnalysisRepository()
    matched = repo.get_matched_news(domain_config.id, limit)

    print(f"\n匹配的热点 - {domain} (共 {len(matched)} 条):")
    print("-" * 80)
    for idx, news in enumerate(matched, 1):
        title = news["title"][:35] + "..." if len(news["title"]) > 35 else news["title"]
        print(f"  {idx:2}. {title} (置信度:{news['confidence']:.2f})")
    print("-" * 80)
