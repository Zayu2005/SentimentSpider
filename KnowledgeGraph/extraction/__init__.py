# -*- coding: utf-8 -*-
"""实体关系抽取模块"""

from .schema import KG_ENTITY_TYPES, KG_RELATION_TYPES, EXTRACTION_SCHEMA
from .extractor import EntityRelationExtractor

__all__ = [
    'KG_ENTITY_TYPES',
    'KG_RELATION_TYPES',
    'EXTRACTION_SCHEMA',
    'EntityRelationExtractor',
]
