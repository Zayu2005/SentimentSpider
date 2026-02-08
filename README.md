# SentimentAnalysis

中文社交媒体情感分析系统，支持多平台数据采集、文本预处理和情感分析。

## 项目结构

```
SentimentAnalysis/
├── SentimentSpider/          # 数据采集模块
│   ├── MediaCrawler/         # 多平台社交媒体爬虫
│   │   ├── media_platform/   # 各平台爬虫实现
│   │   ├── store/            # 数据存储
│   │   └── ...
│   └── hot_news/             # 热点新闻采集
│
├── SentimentProcessor/       # 数据预处理模块
│   ├── config/               # 配置管理
│   ├── database/             # 数据库操作
│   ├── processor/            # 预处理核心
│   │   ├── cleaner.py        # 文本清洗
│   │   ├── segmenter.py      # 中文分词
│   │   └── extractor.py      # 关键词提取
│   ├── utils/                # 工具类
│   │   ├── stopwords.py      # 停用词管理
│   │   └── slang.py          # 网络用语规范化
│   └── cli/                  # 命令行工具
│
└── README.md
```

## 功能特性

### 数据采集 (SentimentSpider)

支持多个主流社交媒体平台的数据采集：

- **小红书** (Xiaohongshu)
- **抖音** (Douyin)
- **快手** (Kuaishou)
- **微博** (Weibo)
- **B站** (Bilibili)
- **贴吧** (Tieba)
- **知乎** (Zhihu)

### 数据预处理 (SentimentProcessor)

- **文本清洗**
  - 移除 URL、邮箱、HTML 标签
  - 移除 @提及 和表情符号
  - 处理平台特定表情格式 (如 `[笑哭R]`)
  - 提取话题标签内容

- **文本规范化**
  - 繁体转简体 (OpenCC)
  - 网络用语规范化 (如 "yyds" → "永远的神")
  - 重复字符压缩

- **中文分词**
  - 基于 jieba 分词
  - 支持自定义词典
  - 停用词过滤

- **关键词提取**
  - TF-IDF 算法
  - TextRank 算法

## 环境要求

- Python 3.10+
- MySQL 5.7+
- Conda (推荐)

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/Zayy2005x/SentimentAnalysis.git
cd SentimentAnalysis
```

### 2. 创建 Conda 环境

```bash
conda create -n sentiment python=3.10
conda activate sentiment
```

### 3. 安装依赖

```bash
# 安装爬虫模块依赖
cd SentimentSpider/MediaCrawler
pip install -r requirements.txt

# 安装预处理模块依赖
cd ../../SentimentProcessor
pip install -r requirements.txt
```

### 4. 配置数据库

在 `SentimentSpider/MediaCrawler/.env` 中配置 MySQL 连接：

```env
MYSQL_DB_HOST=localhost
MYSQL_DB_PORT=3306
MYSQL_DB_USER=root
MYSQL_DB_PWD=your_password
MYSQL_DB_NAME=sentiment
```

## 使用方法

### 数据采集

```bash
cd SentimentSpider/MediaCrawler

# 采集小红书数据
python main.py --platform xhs --keywords "教育"

# 采集抖音数据
python main.py --platform douyin --keywords "教育"
```

### 数据预处理

```bash
cd SentimentAnalysis

# 查看统计信息
python -m SentimentProcessor stats

# 处理所有未处理的数据
python -m SentimentProcessor all

# 只处理内容
python -m SentimentProcessor content

# 只处理评论
python -m SentimentProcessor comment

# 试运行 (不保存到数据库)
python -m SentimentProcessor all --dry-run

# 指定批处理大小
python -m SentimentProcessor all -b 200
```

### Python API

```python
from SentimentProcessor import ContentProcessor, CommentProcessor

# 处理内容
content_processor = ContentProcessor()
result = content_processor.run(batch_size=100)
print(f"处理完成: {result['success']} 成功, {result['error']} 失败")

# 处理评论
comment_processor = CommentProcessor()
result = comment_processor.run(batch_size=100)

# 单独使用文本清洗器
from SentimentProcessor import TextCleaner, Segmenter

cleaner = TextCleaner()
cleaned = cleaner.clean("这是一条测试文本[笑哭R] #话题#")

segmenter = Segmenter()
words = segmenter.segment(cleaned)
```

## 数据库表结构

### 原始数据表

| 表名 | 说明 |
|------|------|
| `unified_content` | 统一内容表 (各平台帖子/笔记) |
| `unified_comment` | 统一评论表 (各平台评论) |
| `xhs_note` | 小红书笔记 |
| `xhs_note_comment` | 小红书评论 |
| `douyin_aweme` | 抖音视频 |
| `douyin_aweme_comment` | 抖音评论 |
| ... | 其他平台表 |

### 预处理数据表

| 表名 | 说明 |
|------|------|
| `processed_content` | 预处理后的内容 |
| `processed_comment` | 预处理后的评论 |

预处理表包含以下字段：
- `original_*` - 原始内容
- `*_cleaned` - 清洗后的内容
- `*_segmented` - 分词结果 (JSON)
- `keywords` - 关键词 (JSON)
- `char_count` / `word_count` - 统计信息

## 项目规划

- [x] 数据采集模块 (SentimentSpider)
- [x] 数据预处理模块 (SentimentProcessor)
- [ ] 情感分析模型 (SentimentModel)
- [ ] API 服务 (SentimentAPI)
- [ ] 可视化仪表板 (SentimentDashboard)

## 作者

**Zayy2005x** - [GitHub](https://github.com/Zayy2005x)

## 许可证

MIT License
