# =====================================================
# Hot News Module - LLM Client
# =====================================================

from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI, AsyncOpenAI
from ..config.settings import get_settings, LLMConfig


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    @abstractmethod
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """发送对话请求"""
        pass

    @abstractmethod
    async def async_chat(self, messages: list, temperature: float = 0.7) -> str:
        """异步发送对话请求"""
        pass


class DeepSeekClient(BaseLLMClient):
    """DeepSeek 客户端"""

    def __init__(self, config: LLMConfig):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base or "https://api.deepseek.com",
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base or "https://api.deepseek.com",
        )
        self.model_name = config.model_name or "deepseek-chat"
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def chat(self, messages: list, temperature: float = None) -> str:
        """同步对话"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def async_chat(self, messages: list, temperature: float = None) -> str:
        """异步对话"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class QwenClient(BaseLLMClient):
    """阿里Qwen客户端"""

    def __init__(self, config: LLMConfig):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = config.model_name or "qwen-turbo"
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def chat(self, messages: list, temperature: float = None) -> str:
        """同步对话"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def async_chat(self, messages: list, temperature: float = None) -> str:
        """异步对话"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class LLMClientFactory:
    """LLM客户端工厂"""

    _clients = {}

    @classmethod
    def get_client(
        cls, provider: str = "deepseek", config: LLMConfig = None
    ) -> BaseLLMClient:
        """获取LLM客户端"""
        if config:
            key = f"{provider}:{config.id}"
        else:
            settings = get_settings()
            default_config = settings.get_default_llm()
            if default_config:
                key = f"{default_config.provider}:{default_config.id}"
            else:
                key = provider

        if key not in cls._clients:
            if config:
                cfg = config
            else:
                settings = get_settings()
                cfg = settings.get_default_llm()

            if cfg is None:
                raise ValueError("未配置LLM，请先配置LLM")

            if provider == "deepseek" or cfg.provider == "deepseek":
                cls._clients[key] = DeepSeekClient(cfg)
            elif provider == "qwen" or cfg.provider == "qwen":
                cls._clients[key] = QwenClient(cfg)
            else:
                raise ValueError(f"不支持的LLM提供商: {provider}")

        return cls._clients[key]

    @classmethod
    def clear_cache(cls):
        """清除客户端缓存"""
        cls._clients.clear()
