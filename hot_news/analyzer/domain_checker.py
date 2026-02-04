# =====================================================
# Hot News Module - Domain Checker
# =====================================================

import json
from typing import List, Dict, Optional
from ..config.settings import get_settings, DomainConfig
from ..models.entities import HotNewsItem, DomainMatchResult
from .llm_client import LLMClientFactory


DOMAIN_MATCH_PROMPT = """你是一个内容分类专家。请分析以下热点新闻是否属于指定领域。

## 任务要求
- 根据领域名称和关键词进行判断
- 评估新闻与领域的相关程度
- 输出JSON格式结果

## 领域信息
- 领域名称：{domain_name}
- 领域关键词：{domain_keywords}

## 热点新闻内容
- 标题：{title}
- 描述：{description}
- 原文链接：{url}
- 来源平台：{platform}

## 判断标准
1. **高匹配**：新闻核心内容直接涉及领域关键词或密切相关话题
2. **中匹配**：新闻部分内容涉及领域话题，但非核心
3. **低匹配**：新闻仅略微提及领域相关内容
4. **不匹配**：新闻内容与领域无关

## 输出要求
请以JSON格式输出分析结果：
{{
    "is_match": true/false,
    "match_level": "高/中/低/无",
    "confidence": 0.0-1.0,
    "reason": "详细说明判断理由"
}}
"""


class DomainChecker:
    """领域匹配检查器"""

    def __init__(self):
        self.settings = get_settings()

    async def check_domain(
        self,
        news: HotNewsItem,
        domain: DomainConfig,
        llm_provider: Optional[str] = None,
    ) -> DomainMatchResult:
        """检查热点是否符合领域"""
        client = LLMClientFactory.get_client(llm_provider)

        messages = [
            {
                "role": "system",
                "content": "你是一个内容分类专家，请严格按照JSON格式输出分析结果。",
            },
            {
                "role": "user",
                "content": DOMAIN_MATCH_PROMPT.format(
                    domain_name=domain.domain_name,
                    domain_keywords=domain.domain_keywords,
                    title=news.title,
                    description=news.description or "",
                    url=news.url or "",
                    platform=news.platform_code,
                ),
            },
        ]

        try:
            response = await client.async_chat(messages, temperature=0.2)
            result = self._parse_response(response)

            return DomainMatchResult(
                news_id=news.news_id,
                domain_id=domain.id,
                domain_name=domain.domain_name,
                is_match=result.get("is_match", False),
                confidence=result.get("confidence", 0.0),
                match_level=result.get("match_level", "无"),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            print(f"[DomainChecker] 检查失败: {e}")
            return DomainMatchResult(
                news_id=news.news_id,
                domain_id=domain.id,
                domain_name=domain.domain_name,
                is_match=False,
                confidence=0.0,
                match_level="无",
                reason=f"LLM调用失败: {str(e)}",
            )

    async def batch_check(
        self,
        news_list: List[HotNewsItem],
        domains: List[DomainConfig],
        llm_provider: Optional[str] = None,
    ) -> List[DomainMatchResult]:
        """批量检查多个领域"""
        results = []

        for news in news_list:
            for domain in domains:
                result = await self.check_domain(news, domain, llm_provider)
                results.append(result)

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
                    "is_match": False,
                    "match_level": "无",
                    "confidence": 0.0,
                    "reason": f"解析失败: {response[:100]}",
                }

    def get_matched_domains(
        self, results: List[DomainMatchResult], min_confidence: float = 0.5
    ) -> Dict[str, List[DomainMatchResult]]:
        """获取匹配的领域结果"""
        matched = {}
        for result in results:
            if result.is_match and result.confidence >= min_confidence:
                if result.news_id not in matched:
                    matched[result.news_id] = []
                matched[result.news_id].append(result)
        return matched
