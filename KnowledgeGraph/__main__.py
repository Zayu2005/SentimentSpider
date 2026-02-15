# -*- coding: utf-8 -*-
"""
KnowledgeGraph 命令行入口

使用方法:
    python -m KnowledgeGraph extract --topic-id 42
    python -m KnowledgeGraph build --topic-id 42
    python -m KnowledgeGraph pipeline --topic-id 42
    python -m KnowledgeGraph query --topic-id 42
    python -m KnowledgeGraph stats
"""

from .cli import main

if __name__ == "__main__":
    main()
