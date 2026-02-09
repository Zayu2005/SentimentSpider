# -*- coding: utf-8 -*-
"""
SentimentModel 命令行入口

使用方法:
    python -m SentimentModel stats
    python -m SentimentModel train --dataset weibo_senti_100k
    python -m SentimentModel analyze --model models/best_model.pt
    python -m SentimentModel predict --model models/best_model.pt --text "这个产品很好"
"""

from .cli import main

if __name__ == "__main__":
    main()
