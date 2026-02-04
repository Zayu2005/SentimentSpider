# =====================================================
# Hot News Module - Run Command (一键执行)
# =====================================================

import typer
from typing import List, Optional
import asyncio
from hot_news.config import get_settings
from hot_news.fetcher import HotNewsFactory
from hot_news.analyzer import DomainChecker, KeywordExtractor
from hot_news.crawler import CrawlTrigger
from hot_news.database import (
    HotNewsRepository,
    AnalysisRepository,
    KeywordRepository,
    TaskLogRepository,
)
from hot_news.models.entities import HotNewsItem

cmd_run = typer.Typer(help="一键执行完整流程")


async def _analyze_domain(news_list, selected_domains):
    """异步分析领域匹配"""
    checker = DomainChecker()
    matched_count = 0

    for idx, news_row in enumerate(news_list):
        news = HotNewsItem(
            news_id=news_row["news_id"],
            platform_code=news_row["platform_code"],
            title=news_row["title"],
            url=news_row.get("url", ""),
            description=news_row.get("description", ""),
        )

        for domain in selected_domains:
            try:
                result = await checker.check_domain(news, domain)
                AnalysisRepository().save_analysis(
                    news_id=result.news_id,
                    domain_id=result.domain_id,
                    is_match=result.is_match,
                    llm_provider="",
                    analysis_content=result.reason,
                    confidence=result.confidence,
                )
                if result.is_match:
                    matched_count += 1
            except Exception:
                pass

    return matched_count


async def _extract_keywords(selected_domains, keyword_limit):
    """异步提取关键词"""
    extractor = KeywordExtractor()
    keyword_repo = KeywordRepository()
    total_keywords = 0

    for domain in selected_domains:
        matched_news = AnalysisRepository().get_matched_news(domain.id, keyword_limit)
        for news_row in matched_news:
            news = HotNewsItem(
                news_id=news_row["news_id"],
                platform_code=news_row["platform_code"],
                title=news_row["title"],
                url=news_row.get("url", ""),
                description=news_row.get("description", ""),
            )

            try:
                keywords = await extractor.extract_keywords(news, domain)
                if keywords:
                    keyword_repo.bulk_save(keywords)
                    total_keywords += len(keywords)
            except Exception:
                pass

    return total_keywords


async def _run_pipeline_inner(
    platforms, domains, crawl_platforms, hot_limit, keyword_limit, no_llm, no_crawl
):
    """内部异步执行函数"""
    settings = get_settings()
    log_repo = TaskLogRepository()

    log_id = log_repo.start_task("hot_pipeline")

    hot_count = 0
    matched_count = 0
    keyword_count = 0
    crawl_count = 0

    try:
        print("=" * 60)
        print("  热点新闻获取与分析流程")
        print("=" * 60)

        # 1. 获取热点
        print("\n[Step 1/4] 获取热点新闻...")

        if platforms:
            selected_platforms = platforms
        else:
            selected_platforms = [p.platform_code for p in settings.get_hot_platforms()]

        repo = HotNewsRepository()

        total = 0
        for platform in selected_platforms:
            try:
                fetcher = HotNewsFactory.get_fetcher(platform)
                news_list = await fetcher.fetch(hot_limit)
                if news_list:
                    repo.bulk_save(news_list)
                    total += len(news_list)
                    print(f"  {platform}: {len(news_list)} 条")
            except Exception as e:
                print(f"  {platform}: 错误 - {e}")
        await HotNewsFactory.close_all()
        hot_count = total
        print(f"  -> 共获取 {hot_count} 条热点")

        # 2. 领域分析
        domain_configs = settings.get_domains()
        if domains:
            selected_domains = [d for d in domain_configs if d.domain_name in domains]
        else:
            selected_domains = domain_configs

        if selected_domains and not no_llm:
            print(f"\n[Step 2/4] 分析领域匹配...")
            news_list = repo.get_recent(hot_limit)
            matched_count = await _analyze_domain(news_list, selected_domains)
            print(f"  -> 匹配 {matched_count} 条")

        # 3. 提取关键词
        if not no_llm and selected_domains:
            print(f"\n[Step 3/4] 提取关键词...")
            keyword_count = await _extract_keywords(selected_domains, keyword_limit)
            print(f"  -> 提取 {keyword_count} 个关键词")

        # 4. 触发爬虫
        if not no_crawl:
            print(f"\n[Step 4/4] 触发爬虫...")

            trigger = CrawlTrigger()
            keyword_repo = KeywordRepository()

            if crawl_platforms:
                crawl_platform_list = crawl_platforms
            else:
                crawl_platform_list = [
                    p.platform_code for p in settings.get_crawler_platforms()
                ]

            keywords = keyword_repo.get_never_crawled(limit=20)
            for kw in keywords:
                for platform in crawl_platform_list:
                    if trigger.trigger_crawl(kw["keyword"], platform, 30, 10):
                        keyword_repo.increment_search_count(kw["id"])
                        crawl_count += 1

            print(f"  -> 触发 {crawl_count} 次爬取")

        print("\n" + "=" * 60)
        print(f"  执行完成!")
        print(
            f"  热点: {hot_count} | 匹配: {matched_count} | 关键词: {keyword_count} | 爬取: {crawl_count}"
        )
        print("=" * 60)

        log_repo.complete_task(
            log_id, "success", hot_count, matched_count, keyword_count, crawl_count
        )

    except Exception as e:
        print(f"\n[Error] {e}")
        log_repo.complete_task(
            log_id,
            "failed",
            hot_count,
            matched_count,
            keyword_count,
            crawl_count,
            str(e),
        )
        raise


@cmd_run.command("run")
def run_pipeline(
    platforms: Optional[List[str]] = typer.Argument(None, help="热点平台列表"),
    domains: Optional[List[str]] = typer.Argument(None, help="领域列表"),
    crawl_platforms: Optional[List[str]] = typer.Option(
        None, "--crawl-platforms", "-cp", help="爬虫平台"
    ),
    hot_limit: int = typer.Option(50, "--hot-limit", "-hl", help="获取热点数量"),
    keyword_limit: int = typer.Option(
        20, "--keyword-limit", "-kl", help="提取关键词数量"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过LLM分析，直接提取关键词"),
    no_crawl: bool = typer.Option(False, "--no-crawl", help="跳过爬虫触发"),
):
    """
    一键执行完整流程：获取热点 -> 分析匹配 -> 提取关键词 -> 触发爬虫

    示例:
        hot-news run weibo zhihu 科技 金融
        hot-news run --hot-limit 100 --crawl-platforms xhs dy
    """
    asyncio.run(
        _run_pipeline_inner(
            platforms,
            domains,
            crawl_platforms,
            hot_limit,
            keyword_limit,
            no_llm,
            no_crawl,
        )
    )
