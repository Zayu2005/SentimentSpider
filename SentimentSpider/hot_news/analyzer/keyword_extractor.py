# =====================================================
# Hot News Module - Keyword Extractor
# =====================================================

import json
from typing import List, Dict, Optional
from ..config.settings import get_settings, DomainConfig
from ..models.entities import HotNewsItem, KeywordResult, DomainMatchResult
from .llm_client import LLMClientFactory


KEYWORD_EXTRACT_PROMPT = """你是一个社交媒体关键词提取专家。请从热点新闻中提取适合在各平台搜索的关键词。

## 任务要求
- 根据新闻内容提取适量的关键词，注重质量而非数量
- 关键词应具有搜索价值和讨论热度
- 保留完整的专有名词、人名、品牌名

## 目标领域
- 领域名称：{domain_name}
- 领域关键词：{domain_keywords}

## 热点新闻内容
- 标题：{title}
- 描述：{description}
- 来源平台：{platform}

## 关键词提取原则
1. **适度性**：根据内容丰富程度提取适量关键词，不要过多也不要遗漏重要词
2. **话题性**：优先选择具有讨论热度的话题词
3. **精确性**：选择能精确定位内容的关键词
4. **去重性**：相似的关键词只保留一个（如"华为"和"华为公司"）
5. **避免**：虚词、停用词、太泛泛的词（如"最新"、"热点"、"重磅"）
    
## 输出要求
请以JSON格式输出：
{{
    "keyword_count": 提取的关键词数量,
    "primary_keyword": "最核心的1个关键词",
    "all_keywords": ["关键词1", "关键词2", "..."],
    "topic_categories": ["话题类别1", "类别2"],
    "extraction_reason": "简要说明提取逻辑"
}}
"""


class KeywordExtractor:
    """关键词提取器"""

    def __init__(self):
        self.settings = get_settings()

    async def extract_keywords(
        self,
        news: HotNewsItem,
        domain: DomainConfig,
        llm_provider: Optional[str] = None,
    ) -> List[KeywordResult]:
        """提取关键词"""
        client = LLMClientFactory.get_client(llm_provider)

        messages = [
            {
                "role": "system",
                "content": "你是一个社交媒体关键词提取专家，请严格按照JSON格式输出结果。",
            },
            {
                "role": "user",
                "content": KEYWORD_EXTRACT_PROMPT.format(
                    domain_name=domain.domain_name,
                    domain_keywords=domain.domain_keywords,
                    title=news.title,
                    description=news.description or "",
                    platform=news.platform_code,
                ),
            },
        ]

        try:
            response = await client.async_chat(messages, temperature=0.3)
            result = self._parse_response(response)

            keywords = result.get("all_keywords", [])
            primary = result.get("primary_keyword", "")

            results = []
            for kw in keywords:
                results.append(
                    KeywordResult(
                        keyword=kw,
                        source_news_id=news.news_id,
                        domain_id=domain.id,
                        llm_provider=llm_provider or "",
                        confidence=result.get("keyword_count", len(keywords))
                        / len(keywords)
                        if keywords
                        else 0,
                        primary_keyword=kw if kw == primary else "",
                    )
                )

            return results
        except Exception as e:
            print(f"[KeywordExtractor] 提取失败: {e}")
            return []

    async def batch_extract(
        self,
        news_list: List[HotNewsItem],
        domains: List[DomainConfig],
        llm_provider: Optional[str] = None,
    ) -> Dict[str, List[KeywordResult]]:
        """批量提取关键词，按新闻ID分组"""
        results = {}

        for news in news_list:
            for domain in domains:
                keywords = await self.extract_keywords(news, domain, llm_provider)
                if keywords:
                    if news.news_id not in results:
                        results[news.news_id] = []
                    results[news.news_id].extend(keywords)

        return results

    def _parse_response(self, response: str) -> dict:
        """解析LLM响应"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                return json.loads(response.strip())
            except json.JSONDecodeError:
                return {
                    "keyword_count": 0,
                    "primary_keyword": "",
                    "all_keywords": [],
                    "topic_categories": [],
                    "extraction_reason": f"解析失败: {response[:100]}",
                }

    def get_all_keywords(
        self, results: Dict[str, List[KeywordResult]]
    ) -> List[KeywordResult]:
        """获取所有关键词"""
        all_keywords = []
        for keyword_list in results.values():
            all_keywords.extend(keyword_list)
        return all_keywords

    def deduplicate_keywords(
        self, keywords: List[KeywordResult]
    ) -> Dict[str, KeywordResult]:
        """去重关键词，保留置信度最高的"""
        unique = {}
        for kw in keywords:
            if (
                kw.keyword not in unique
                or kw.confidence > unique[kw.keyword].confidence
            ):
                unique[kw.keyword] = kw
        return unique
