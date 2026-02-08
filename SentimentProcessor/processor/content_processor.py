# =====================================================
# SentimentProcessor - Content Processor
# 内容预处理器
# =====================================================

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..config import get_settings
from ..database import UnifiedContentRepo, ProcessedContentRepo
from .cleaner import TextCleaner
from .segmenter import Segmenter
from .extractor import KeywordExtractor


logger = logging.getLogger(__name__)


class ContentProcessor:
    """内容预处理器"""

    def __init__(self):
        """初始化内容预处理器"""
        self.settings = get_settings()
        self.cleaner = TextCleaner()
        self.segmenter = Segmenter()
        self.extractor = KeywordExtractor(topK=10)

    def process_single(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单条内容

        Args:
            content: 原始内容字典，包含 id, platform, content_id, title, content

        Returns:
            预处理结果字典
        """
        try:
            title = content.get("title", "") or ""
            text = content.get("content", "") or ""

            # 清洗文本
            title_cleaned = self.cleaner.clean(title)
            content_cleaned = self.cleaner.clean(text)

            # 分词
            title_segmented = self.segmenter.segment(title_cleaned) if title_cleaned else []
            content_segmented = self.segmenter.segment(content_cleaned) if content_cleaned else []

            # 提取关键词（合并标题和内容）
            combined_text = f"{title_cleaned} {content_cleaned}".strip()
            keywords = self.extractor.extract(combined_text, method="tfidf", topK=10)

            # 统计信息
            char_count = len(content_cleaned)
            word_count = len(content_segmented)

            return {
                "unified_id": content["id"],
                "platform": content["platform"],
                "content_id": content["content_id"],
                "original_title": title,
                "original_content": text,
                "title_cleaned": title_cleaned,
                "content_cleaned": content_cleaned,
                "title_segmented": title_segmented,
                "content_segmented": content_segmented,
                "keywords": keywords,
                "char_count": char_count,
                "word_count": word_count,
                "process_status": "completed",
                "preprocessed_at": datetime.now(),
            }
        except Exception as e:
            logger.error(f"处理内容失败 [id={content.get('id')}]: {e}")
            return {
                "unified_id": content["id"],
                "platform": content.get("platform", ""),
                "content_id": content.get("content_id", ""),
                "original_title": content.get("title", ""),
                "original_content": content.get("content", ""),
                "title_cleaned": "",
                "content_cleaned": "",
                "title_segmented": [],
                "content_segmented": [],
                "keywords": [],
                "char_count": 0,
                "word_count": 0,
                "process_status": "error",
                "error_message": str(e),
                "preprocessed_at": datetime.now(),
            }

    def process_batch(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理内容

        Args:
            contents: 原始内容列表

        Returns:
            预处理结果列表
        """
        results = []
        for content in contents:
            result = self.process_single(content)
            results.append(result)
        return results

    def run(
        self,
        batch_size: Optional[int] = None,
        max_items: Optional[int] = None,
        save_to_db: bool = True
    ) -> Dict[str, int]:
        """
        运行预处理流程

        Args:
            batch_size: 批处理大小（None使用配置值）
            max_items: 最大处理数量（None处理全部）
            save_to_db: 是否保存到数据库

        Returns:
            处理统计信息
        """
        batch_size = batch_size or self.settings.processor.batch_size
        offset = 0
        total_processed = 0
        total_success = 0
        total_error = 0

        logger.info("开始预处理内容...")

        while True:
            # 获取未处理的内容
            contents = UnifiedContentRepo.get_unprocessed(limit=batch_size, offset=0)
            if not contents:
                break

            # 检查是否达到最大处理数量
            if max_items and total_processed >= max_items:
                break

            # 限制本批次数量
            if max_items:
                remaining = max_items - total_processed
                contents = contents[:remaining]

            # 批量处理
            results = self.process_batch(contents)

            # 统计结果
            success_results = [r for r in results if r["process_status"] == "completed"]
            error_results = [r for r in results if r["process_status"] == "error"]

            total_processed += len(results)
            total_success += len(success_results)
            total_error += len(error_results)

            # 保存到数据库
            if save_to_db and results:
                try:
                    ProcessedContentRepo.batch_insert(results)
                    logger.info(f"已处理 {total_processed} 条内容 (成功: {total_success}, 失败: {total_error})")
                except Exception as e:
                    logger.error(f"保存预处理结果失败: {e}")

            # 检查是否处理完毕
            if len(contents) < batch_size:
                break

        logger.info(f"内容预处理完成: 总计 {total_processed}, 成功 {total_success}, 失败 {total_error}")

        return {
            "total": total_processed,
            "success": total_success,
            "error": total_error,
        }

    def get_stats(self) -> Dict[str, int]:
        """获取预处理统计信息"""
        return {
            "total_content": UnifiedContentRepo.count_all(),
            "unprocessed": UnifiedContentRepo.count_unprocessed(),
            "processed": ProcessedContentRepo.count_all(),
            "completed": ProcessedContentRepo.count_by_status("completed"),
            "error": ProcessedContentRepo.count_by_status("error"),
        }
