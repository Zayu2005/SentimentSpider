# -*- coding: utf-8 -*-
"""测试预测结果"""
from SentimentModel.inference import SentimentPredictor

predictor = SentimentPredictor(model_path='models/20260209_193815/best_model.pt')

test_texts = [
    '这个产品真的太好用了',
    '我很喜欢这个',
    '太棒了',
    '好开心',
    '垃圾产品',
    '太差了',
    '不好用',
    '很失望',
]

print('测试结果:')
print('-' * 60)
for text in test_texts:
    result = predictor.predict(text)
    print(f'{text:20} -> {result.label:10} (P_neg={result.probs["negative"]:.3f}, P_pos={result.probs["positive"]:.3f})')
