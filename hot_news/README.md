# Hot News Module

热点新闻获取与分析模块，从 orz.ai 获取热点，使用 LLM 进行领域匹配和关键词提取。

## 快速开始

### 1. 数据库初始化

```bash
python -m hot_news.cli.main init-db
```

### 2. 配置领域

```bash
# 添加领域
python -m hot_news.cli.main config domain add --name "科技" --keywords "AI,编程,技术,软件"
python -m hot_news.cli.main config domain add --name "金融" --keywords "投资,股票,基金,理财"
python -m hot_news.cli.main config domain add --name "医疗" --keywords "医药,健康,医院,医保"

# 查看领域
python -m hot_news.cli.main config domain list
```

### 3. 配置 LLM

```bash
# 添加 DeepSeek 配置
python -m hot_news.cli.main config llm add --provider deepseek --api-key "your-api-key" --model "deepseek-chat"

# 添加 Qwen 配置
python -m hot_news.cli.main config llm add --provider qwen --api-key "your-api-key" --model "qwen-turbo"

# 设置默认
python -m hot_news.cli.main config llm set-default --provider deepseek

# 查看配置
python -m hot_news.cli.main config llm list
```

### 4. 一键执行完整流程

```bash
python -m hot_news.cli.main run weibo zhihu 科技 金融
```

## 命令说明

### fetch - 获取热点

```bash
# 从所有启用的平台获取热点
python -m hot_news.cli.main fetch

# 从指定平台获取
python -m hot_news.cli.main fetch weibo zhihu bilibili

# 限制数量
python -m hot_news.cli.main fetch --limit 100
```

### analyze - 分析领域匹配

```bash
# 分析所有配置的领域
python -m hot_news.cli.main analyze

# 分析指定领域
python -m hot_news.cli.main analyze 科技 金融

# 设置最小置信度
python -m hot_news.cli.main analyze --min-confidence 0.7
```

### extract - 提取关键词

```bash
# 从匹配的热点中提取关键词
python -m hot_news.cli.main extract

# 从指定领域提取
python -m hot_news.cli.main extract 科技
```

### crawl - 触发爬虫

```bash
# 使用关键词触发爬虫
python -m hot_news.cli.main crawl xhs dy --keyword "AI 编程"

# 使用数据库中的关键词
python -m hot_news.cli.main crawl xhs dy --limit 10
```

### run - 一键执行

```bash
# 执行完整流程
python -m hot_news.cli.main run weibo zhihu 科技 金融

# 跳过 LLM 分析
python -m hot_news.cli.main run --no-llm

# 跳过爬虫触发
python -m hot_news.cli.main run --no-crawl
```

### config - 配置管理

```bash
# 领域配置
python -m hot_news.cli.main config domain list
python -m hot_news.cli.main config domain add --name "科技" --keywords "AI,编程"
python -m hot_news.cli.main config domain enable --name "科技"
python -m hot_news.cli.main config domain delete --name "科技"

# LLM配置
python -m hot_news.cli.main config llm list
python -m hot_news.cli.main config llm add --provider deepseek --api-key "xxx"

# 平台配置
python -m hot_news.cli.main config platform list
python -m hot_news.cli.main config platform enable --codes "weibo,zhihu"
```

### show - 查看数据

```bash
# 查看热点
python -m hot_news.cli.main show hot-news --limit 20

# 查看关键词
python -m hot_news.cli.main show keywords --limit 50

# 查看匹配热点
python -m hot_news.cli.main show matched 科技

# 查看任务日志
python -m hot_news.cli.main show logs --limit 10
```

## 数据库表说明

| 表名 | 说明 |
|------|------|
| `hot_platform_config` | 热点平台配置 |
| `domain_config` | 领域配置 |
| `llm_config` | LLM配置 |
| `crawler_platform_config` | 爬虫平台配置 |
| `task_schedule_config` | 任务调度配置 |
| `hot_news` | 热点新闻数据 |
| `hot_news_analysis` | 热点分析结果 |
| `extracted_keywords` | 提取的关键词 |
| `keyword_crawl_log` | 关键词爬取记录 |
| `task_execution_log` | 任务执行日志 |

## 支持的热点平台

- 百度热搜 (baidu)
- 微博热搜 (weibo)
- 知乎热榜 (zhihu)
- B站热门 (bilibili)
- 抖音热点 (douyin)
- 掘金技术 (juejin)
- GitHub Trending (github)
- Hacker News (hackernews)
- 新浪财经 (sina_finance)
- 雪球 (xueqiu)

## 依赖

```
httpx>=0.27.0
openai>=1.0.0
typer>=0.12.3
pymysql>=1.1.0
pydantic>=2.5.0
```

## License

MIT
