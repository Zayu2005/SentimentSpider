# -*- coding: utf-8 -*-
"""
测试 Qwen2.5 基础模型的情感分析能力 (无需微调)
"""

import os
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv
env_file = Path(__file__).parent / ".env"
load_dotenv(str(env_file))

# 设置 HuggingFace 镜像 (如果环境变量未设置则使用默认值)
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# 模型名称 (可选: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的情感分析助手。请分析用户提供的文本，输出以下信息：
1. sentiment: 情感倾向 (positive/negative/neutral)
2. sentiment_score: 情感分数 (-1.0 到 1.0，负面为负，正面为正)
3. emotion_tags: 情绪标签列表 (可选: 喜悦、兴奋、满足、感激、爱、愤怒、厌恶、悲伤、恐惧、失望、惊讶、困惑、好奇、期待、焦虑、平静、无聊、冷漠)

请只输出 JSON 格式结果，不要有其他内容。"""


def load_model():
    """加载模型"""
    print(f"正在加载模型: {MODEL_NAME}")
    print("首次运行需要下载模型，请耐心等待...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # 使用 fp16 节省显存
        device_map="auto",
        trust_remote_code=True
    )

    print("模型加载完成！")
    return model, tokenizer


def predict(model, tokenizer, text):
    """预测单条文本"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下文本的情感：\n\n{text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    # 加载模型
    model, tokenizer = load_model()

    # 测试文本
    test_texts = [
        "这个产品真的太好用了！强烈推荐！",
        "我很喜欢这个，感觉很满意",
        "太棒了，超出预期",
        "好开心，终于买到了",
        "垃圾产品，千万别买",
        "太差了，非常失望",
        "不好用，退货了",
        "很失望，和描述不符",
        "还行吧，一般般",
        "没什么特别的感觉",
    ]

    print("\n" + "=" * 70)
    print("Qwen2.5 情感分析测试")
    print("=" * 70)

    for text in test_texts:
        print(f"\n📝 文本: {text}")
        response = predict(model, tokenizer, text)
        print(f"🤖 输出: {response}")

        # 尝试解析 JSON
        try:
            # 提取 JSON 部分
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"   ✅ 情感: {result.get('sentiment', 'N/A')}")
                print(f"   ✅ 分数: {result.get('sentiment_score', 'N/A')}")
                print(f"   ✅ 情绪: {result.get('emotion_tags', 'N/A')}")
        except:
            print("   ⚠️ JSON 解析失败")

        print("-" * 50)


if __name__ == "__main__":
    main()
