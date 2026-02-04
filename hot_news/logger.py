# =====================================================
# Hot News Module - Logging Configuration
# =====================================================

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "hot_news", log_dir: str = None) -> logging.Logger:
    """
    设置logger，同时输出到控制台和文件

    Args:
        name: logger名称
        log_dir: 日志文件目录，默认为 hot_news/logs

    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 确定日志目录
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"hot_news_{timestamp}.log"

    # ===== 格式化器 =====
    # 文件日志格式：详细信息
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台日志格式：简化，保留emoji和颜色感
    console_formatter = logging.Formatter(
        fmt="%(message)s"
    )

    # ===== 文件Handler (所有日志) =====
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,  # 保留10个日志文件
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ===== 控制台Handler (INFO及以上) =====
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# 创建默认logger
logger = setup_logger()


# ===== 便利函数（直接调用无需获取logger) =====
def debug(msg, *args, **kwargs):
    """输出DEBUG级别日志"""
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """输出INFO级别日志"""
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """输出WARNING级别日志"""
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """输出ERROR级别日志"""
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """输出CRITICAL级别日志"""
    logger.critical(msg, *args, **kwargs)
