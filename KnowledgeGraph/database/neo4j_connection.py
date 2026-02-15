# -*- coding: utf-8 -*-
"""
Neo4j 数据库连接管理
"""

from contextlib import contextmanager

from ..config import get_kg_settings

# 模块级驱动单例
_driver = None


def get_neo4j_driver():
    """
    获取 Neo4j 驱动 (单例)

    Returns:
        neo4j.Driver instance
    """
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        settings = get_kg_settings()
        cfg = settings.neo4j
        _driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.user, cfg.password),
        )
        # 验证连通性
        _driver.verify_connectivity()
    return _driver


@contextmanager
def neo4j_session(access_mode=None):
    """
    Neo4j session 上下文管理器

    Usage:
        with neo4j_session() as session:
            session.run("MATCH (n) RETURN n LIMIT 10")
    """
    driver = get_neo4j_driver()
    settings = get_kg_settings()
    kwargs = {"database": settings.neo4j.database}
    if access_mode:
        kwargs["default_access_mode"] = access_mode
    session = driver.session(**kwargs)
    try:
        yield session
    finally:
        session.close()


def close_neo4j_driver():
    """关闭 Neo4j 驱动"""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
