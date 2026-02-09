# SentimentModel 情感分析模块设计文档

## 1. 模块概述

SentimentModel 是情感分析系统的核心模块，负责对预处理后的中文社交媒体文本进行情感分析，输出情感倾向、情感分数和情绪标签。

### 1.1 功能目标

- 对内容（帖子/笔记）进行情感分析
- 对评论进行情感分析
- 支持批量处理和实时推理
- 支持模型训练和微调

### 1.2 输入输出

**输入**:
- 清洗后的中文文本 (`content_cleaned`)
- 来源: `processed_content` / `processed_comment` 表

**输出**:
- `sentiment`: 情感倾向 (positive/negative/neutral)
- `sentiment_score`: 情感分数 (-1.0 ~ 1.0)
- `emotion_tags`: 情绪标签 (可选，如：喜悦/愤怒/悲伤)

---

## 2. 技术选型

### 2.1 模型选择

| 方案 | 模型 | 参数量 | 推理速度 | 效果 | 推荐 |
|------|------|--------|----------|------|------|
| A | chinese-roberta-wwm-ext-base | 102M | 中等 | 优秀 | ⭐ 推荐 |
| B | bert-base-chinese | 102M | 中等 | 良好 | 备选 |
| C | DistilBERT (蒸馏) | 66M | 快 | 良好 | 资源受限 |
| D | TextCNN | 1M | 很快 | 一般 | 轻量需求 |

**推荐方案 A**: `hfl/chinese-roberta-wwm-ext-base`

理由：
1. 哈工大讯飞实验室出品，中文效果最佳
2. 全词遮罩 (Whole Word Masking) 更适合中文
3. HuggingFace 生态完善，易于使用
4. 可在 RTX 3060 (12GB) 及以上 GPU 上训练

### 2.2 技术栈

```
核心框架:
├── transformers (HuggingFace)    # 预训练模型
├── torch                          # 深度学习框架
├── datasets                       # 数据集处理
└── scikit-learn                   # 评估指标

数据处理:
├── pandas                         # 数据操作
└── pymysql                        # 数据库连接

工具:
├── tqdm                           # 进度条
├── tensorboard                    # 训练可视化
└── pydantic                       # 配置管理
```

---

## 3. 模块架构

### 3.1 目录结构

```
SentimentModel/
├── __init__.py                    # 模块入口
├── __main__.py                    # CLI 入口
├── requirements.txt               # 依赖列表
│
├── config/
│   ├── __init__.py
│   └── settings.py                # 配置管理
│
├── models/
│   ├── __init__.py
│   ├── bert_classifier.py         # BERT 分类模型
│   └── base.py                    # 模型基类
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset
│   ├── dataloader.py              # 数据加载器
│   └── augmentation.py            # 数据增强 (可选)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # 训练器
│   ├── metrics.py                 # 评估指标
│   └── callbacks.py               # 训练回调
│
├── inference/
│   ├── __init__.py
│   └── predictor.py               # 推理预测器
│
├── database/
│   ├── __init__.py
│   └── repository.py              # 数据库操作
│
├── cli/
│   ├── __init__.py
│   └── main.py                    # 命令行工具
│
└── utils/
    ├── __init__.py
    └── logger.py                  # 日志工具
```

### 3.2 核心类设计

```python
# models/bert_classifier.py
class BertSentimentClassifier(nn.Module):
    """BERT 情感分类器"""

    def __init__(self, model_name: str, num_labels: int = 3):
        # 加载预训练 BERT
        # 添加分类头

    def forward(self, input_ids, attention_mask):
        # 返回 logits 和 情感分数

# inference/predictor.py
class SentimentPredictor:
    """情感预测器"""

    def __init__(self, model_path: str, device: str = "auto"):
        # 加载模型

    def predict(self, text: str) -> SentimentResult:
        # 单条预测

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        # 批量预测

# training/trainer.py
class SentimentTrainer:
    """模型训练器"""

    def train(self, train_data, val_data, epochs: int):
        # 训练循环

    def evaluate(self, test_data) -> MetricResult:
        # 模型评估
```

---

## 4. 数据流设计

### 4.1 训练数据流

```
┌─────────────────────────────────────────────────────────────┐
│                      训练数据准备                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  公开数据集                    自有数据 (标注)                 │
│  ├── weibo_senti_100k         ├── processed_content          │
│  ├── ChnSentiCorp             └── processed_comment          │
│  └── 外卖评论数据集                  (人工标注)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据预处理                               │
│  ├── 文本截断/填充 (max_length=128)                          │
│  ├── Tokenization (BERT Tokenizer)                          │
│  ├── 标签编码 (0=负面, 1=中性, 2=正面)                        │
│  └── 训练/验证/测试集划分 (8:1:1)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型训练                                 │
│  ├── 加载预训练模型 (chinese-roberta-wwm-ext)                │
│  ├── 添加分类头 (768 -> 3)                                   │
│  ├── 训练参数: lr=2e-5, batch=32, epochs=3                   │
│  └── 保存最佳模型 (按 F1-score)                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 推理数据流

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ processed_content│     │  SentimentModel  │     │ unified_content  │
│ processed_comment│ ──▶ │                  │ ──▶ │ unified_comment  │
│ (content_cleaned)│     │  predict_batch() │     │ (sentiment 字段) │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         │                        │                        │
    读取清洗后文本            批量推理预测              更新情感字段
```

### 4.3 批量处理流程

```python
# 伪代码
def run_sentiment_analysis(batch_size=100):
    predictor = SentimentPredictor("models/best_model")

    while True:
        # 1. 获取未分析的内容
        contents = get_unanalyzed_contents(limit=batch_size)
        if not contents:
            break

        # 2. 提取文本
        texts = [c["content_cleaned"] for c in contents]

        # 3. 批量预测
        results = predictor.predict_batch(texts)

        # 4. 更新数据库
        for content, result in zip(contents, results):
            update_sentiment(
                unified_id=content["unified_id"],
                sentiment=result.label,          # positive/negative/neutral
                sentiment_score=result.score,    # -1.0 ~ 1.0
                emotion_tags=result.emotions     # 可选
            )
```

---

## 5. 模型细节

### 5.1 模型架构

```
输入文本: "这个产品真的太好用了，强烈推荐！"
                    │
                    ▼
┌─────────────────────────────────────────┐
│           BERT Tokenizer                │
│  [CLS] 这个 产品 真的 太 好用 了 ... [SEP]  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│     chinese-roberta-wwm-ext-base        │
│         (12层 Transformer)               │
│         Hidden Size: 768                 │
└─────────────────────────────────────────┘
                    │
                    ▼ [CLS] token embedding
┌─────────────────────────────────────────┐
│           Dropout (0.1)                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Linear (768 -> 3)                  │
│       分类头 (3分类)                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Softmax                        │
│  [0.05, 0.10, 0.85] (负/中/正)           │
└─────────────────────────────────────────┘
                    │
                    ▼
输出: sentiment="positive", score=0.85
```

### 5.2 情感分数计算

```python
def calculate_sentiment_score(logits):
    """
    将三分类 logits 转换为 -1 到 1 的情感分数

    logits: [negative_prob, neutral_prob, positive_prob]
    score = positive_prob - negative_prob
    """
    probs = softmax(logits)
    score = probs[2] - probs[0]  # positive - negative
    return score  # 范围: -1.0 ~ 1.0
```

### 5.3 训练配置

```yaml
model:
  name: "hfl/chinese-roberta-wwm-ext-base"
  num_labels: 3
  max_length: 128
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true  # 混合精度训练

optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: "linear"
  warmup_steps: 500
```

---

## 6. CLI 命令设计

```bash
# 查看统计信息
python -m SentimentModel stats

# 训练模型
python -m SentimentModel train \
    --dataset weibo_senti_100k \
    --epochs 3 \
    --batch-size 32 \
    --output models/sentiment_v1

# 评估模型
python -m SentimentModel evaluate \
    --model models/sentiment_v1 \
    --test-data data/test.csv

# 分析数据库中的数据
python -m SentimentModel analyze \
    --batch-size 100 \
    --model models/sentiment_v1

# 单条文本预测
python -m SentimentModel predict \
    --model models/sentiment_v1 \
    --text "这个产品真的太好用了！"

# 导出模型 (ONNX格式，用于部署)
python -m SentimentModel export \
    --model models/sentiment_v1 \
    --format onnx \
    --output models/sentiment_v1.onnx
```

---

## 7. 数据库交互

### 7.1 读取待分析数据

```sql
-- 获取未进行情感分析的内容
SELECT
    pc.unified_id,
    pc.platform,
    pc.content_cleaned,
    pc.title_cleaned
FROM processed_content pc
JOIN unified_content uc ON pc.unified_id = uc.id
WHERE uc.sentiment IS NULL
  AND pc.process_status = 'completed'
  AND pc.content_cleaned IS NOT NULL
  AND pc.content_cleaned != ''
ORDER BY pc.id
LIMIT 100;
```

### 7.2 更新情感分析结果

```sql
-- 更新情感分析结果
UPDATE unified_content
SET
    sentiment = %s,
    sentiment_score = %s,
    emotion_tags = %s,
    sentiment_analyzed_at = NOW()
WHERE id = %s;
```

### 7.3 统计查询

```sql
-- 情感分布统计
SELECT
    platform,
    sentiment,
    COUNT(*) as count,
    AVG(sentiment_score) as avg_score
FROM unified_content
WHERE sentiment IS NOT NULL
GROUP BY platform, sentiment;
```

---

## 8. 配置管理

### 8.1 配置文件结构

```python
# config/settings.py

class ModelConfig(BaseModel):
    """模型配置"""
    name: str = "hfl/chinese-roberta-wwm-ext-base"
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1
    device: str = "auto"  # auto/cpu/cuda

class TrainingConfig(BaseModel):
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    fp16: bool = True
    output_dir: str = "models"

class InferenceConfig(BaseModel):
    """推理配置"""
    batch_size: int = 64
    model_path: str = "models/best_model"
    threshold: float = 0.5  # 置信度阈值
```

---

## 9. 评估指标

### 9.1 分类指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| Accuracy | 准确率 | > 85% |
| Macro F1 | 宏平均 F1 | > 80% |
| Weighted F1 | 加权 F1 | > 85% |
| Per-class F1 | 各类别 F1 | > 75% |

### 9.2 混淆矩阵示例

```
              Predicted
            Neg   Neu   Pos
Actual Neg  850    50   100
       Neu   80   800   120
       Pos   70    50   880
```

---

## 10. 部署方案

### 10.1 本地部署

```bash
# 安装依赖
pip install -r SentimentModel/requirements.txt

# 下载预训练模型 (首次运行自动下载)
python -m SentimentModel download --model hfl/chinese-roberta-wwm-ext-base

# 运行分析
python -m SentimentModel analyze --batch-size 100
```

### 10.2 GPU 要求

| 操作 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 推理 | 4GB VRAM | 8GB VRAM |
| 训练 | 8GB VRAM | 12GB+ VRAM |
| CPU 模式 | 16GB RAM | 32GB RAM |

### 10.3 性能预估

| 配置 | 推理速度 | 训练速度 |
|------|----------|----------|
| RTX 3060 (12GB) | ~200 条/秒 | ~3小时/epoch |
| RTX 3080 (10GB) | ~350 条/秒 | ~2小时/epoch |
| CPU (i7) | ~10 条/秒 | 不推荐 |

---

## 11. 开发计划

### Phase 1: 基础框架 (第1周)
- [ ] 创建模块目录结构
- [ ] 实现配置管理
- [ ] 实现数据库 Repository
- [ ] 实现基础 CLI

### Phase 2: 模型开发 (第2周)
- [ ] 实现 BERT 分类器
- [ ] 实现数据集加载
- [ ] 实现训练器
- [ ] 使用公开数据集训练

### Phase 3: 推理集成 (第3周)
- [ ] 实现推理预测器
- [ ] 集成到 CLI
- [ ] 批量处理数据库数据
- [ ] 性能优化

### Phase 4: 测试优化 (第4周)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 模型调优
- [ ] 文档完善

---

## 12. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| GPU 显存不足 | 无法训练 | 降低 batch_size，使用梯度累积 |
| 训练数据不足 | 效果差 | 使用公开数据集预训练，再微调 |
| 推理速度慢 | 处理效率低 | 使用 ONNX 导出，批量处理 |
| 中性类别难分 | F1 低 | 调整阈值，增加中性样本 |

---

## 附录: 公开数据集

### A. 微博情感数据集 (weibo_senti_100k)
- 规模: 119,988 条
- 类别: 正面/负面 (二分类)
- 来源: https://github.com/SophonPlus/ChineseNlpCorpus

### B. ChnSentiCorp
- 规模: 9,600 条
- 类别: 正面/负面
- 来源: 酒店评论

### C. 外卖评论数据集
- 规模: 11,987 条
- 类别: 正面/负面
- 来源: 外卖平台

### D. 自建数据集方案
从已采集的小红书数据中随机抽样 1000-2000 条，人工标注情感标签，用于领域微调。
