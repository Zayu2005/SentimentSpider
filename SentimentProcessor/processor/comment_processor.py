# =====================================================
# SentimentProcessor - Comment Processor
# 评论预处理器
# =====================================================

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..config import get_settings
from ..database import UnifiedCommentRepo, ProcessedCommentRepo
from .cleaner import TextCleaner
from .segmenter import Segmenter
from .extractor import KeywordExtractor


logger = logging.getLogger(__name__)


class CommentProcessor:
    """评论预处理器"""

    def __init__(self):
        """初始化评论预处理器"""
        self.settings = get_settings()
        self.cleaner = TextCleaner()
        self.segmenter = Segmenter()
        self.extractor = KeywordExtractor(topK=5)  # 评论关键词较少

    def process_single(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单条评论

        Args:
            comment: 原始评论字典，包含 id, platform, comment_id, content_id, content

        Returns:
            预处理结果字典
        """
        try:
            text = comment.get("content", "") or ""

            # 清洗文本
            content_cleaned = self.cleaner.clean(text)

            # 分词
            content_segmented = self.segmenter.segment(content_cleaned) if content_cleaned else []

            # 提取关键词（评论较短，提取5个）
            keywords = self.extractor.extract(content_cleaned, method="tfidf", topK=5) if content_cleaned else []

            # 统计信息
            char_count = len(content_cleaned)
            word_count = len(content_segmented)

            return {
                "unified_id": comment["id"],
                "platform": comment["platform"],
                "comment_id": comment["comment_id"],
                "content_id": comment["content_id"],
                "original_content": text,
                "content_cleaned": content_cleaned,
                "content_segmented": content_segmented,
                "keywords": keywords,
                "char_count": char_count,
                "word_count": word_count,
                "process_status": "completed",
                "preprocessed_at": datetime.now(),
            }
        except Exception as e:
            logger.error(f"处理评论失败 [id={comment.get('id')}]: {e}")
            return {
                "unified_id": comment["id"],
                "platform": comment.get("platform", ""),
                "comment_id": comment.get("comment_id", ""),
                "content_id": comment.get("content_id", ""),
                "original_content": comment.get("content", ""),
                "content_cleaned": "",
                "content_segmented": [],
                "keywords": [],
                "char_count": 0,
                "word_count": 0,
                "process_status": "error",
                "error_message": str(e),
                "preprocessed_at": datetime.now(),
            }

    def process_batch(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理评论

        Args:
            comments: 原始评论列表

        Returns:
            预处理结果列表
        """
        results = []
        for comment in comments:
            result = self.process_single(comment)
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
        total_processed = 0
        total_success = 0
        total_error = 0

        logger.info("开始预处理评论...")

        while True:
            # 获取未处理的评论
            comments = UnifiedCommentRepo.get_unprocessed(limit=batch_size, offset=0)
            if not comments:
                break

            # 检查是否达到最大处理数量
            if max_items and total_processed >= max_items:
                break

            # 限制本批次数量
            if max_items:
                remaining = max_items - total_processed
                comments = comments[:remaining]

            # 批量处理
            results = self.process_batch(comments)

            # 统计结果
            success_results = [r for r in results if r["process_status"] == "completed"]
            error_results = [r for r in results if r["process_status"] == "error"]

            total_processed += len(results)
            total_success += len(success_results)
            total_error += len(error_results)

            # 保存到数据库
            if save_to_db and results:
                try:
                    ProcessedCommentRepo.batch_insert(results)
                    logger.info(f"已处理 {total_processed} 条评论 (成功: {total_success}, 失败: {total_error})")
                except Exception as e:
                    logger.error(f"保存预处理结果失败: {e}")

            # 检查是否处理完毕
            if len(comments) < batch_size:
                break

        logger.info(f"评论预处理完成: 总计 {total_processed}, 成功 {total_success}, 失败 {total_error}")

        return {
            "total": total_processed,
            "success": total_success,
            "error": total_error,
        }

    def get_stats(self) -> Dict[str, int]:
        """获取预处理统计信息"""
        return {
            "total_comment": UnifiedCommentRepo.count_all(),
            "unprocessed": UnifiedCommentRepo.count_unprocessed(),
            "processed": ProcessedCommentRepo.count_all(),
            "completed": ProcessedCommentRepo.count_by_status("completed"),
            "error": ProcessedCommentRepo.count_by_status("error"),
        }
