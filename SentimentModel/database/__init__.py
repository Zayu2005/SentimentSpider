# -*- coding: utf-8 -*-
"""数据库模块"""

from .connection import get_connection, execute_query, execute_many, execute_update
from .repository import SentimentContentRepo, SentimentCommentRepo

__all__ = [
    'get_connection',
    'execute_query',
    'execute_many',
    'execute_update',
    'SentimentContentRepo',
    'SentimentCommentRepo'
]
