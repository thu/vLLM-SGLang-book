#!/usr/bin/env python3
# ch06/load_test/locustfile.py
# Locust 压测脚本
#
# 依赖：Locust 2.32.x
# 运行（Web UI 模式）：
#   locust -f ch06/load_test/locustfile.py --host http://localhost:8001
# 运行（无头模式，用于自动化）：
#   locust -f ch06/load_test/locustfile.py --host http://localhost:8001 \
#          --headless -u 500 -r 50 -t 5m \
#          --csv results/baseline

import json
import random
from pathlib import Path

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# 预加载测试数据，避免每次请求时读取文件
# 数据集格式：每行一个 JSON 对象，包含 "question" 字段
_DATA_FILE = Path(__file__).parent.parent / "data" / "customer_service_queries.jsonl"

def _load_queries() -> list[str]:
    if not _DATA_FILE.exists():
        # 数据文件不存在时使用内置样本，确保脚本可独立运行
        return [
            "这个商品有库存吗？",
            "我的订单什么时候发货？",
            "如何申请退款？",
            "商品质量有问题怎么办？",
            "支持哪些支付方式？",
            "快递一般几天到？",
            "这款商品有什么颜色可以选？",
            "可以开增值税发票吗？",
            "商品有质保吗？质保期多久？",
            "如何查看物流信息？",
            "优惠券怎么使用？",
            "商品参数是什么？",
        ]
    queries = []
    with open(_DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries.append(obj["question"])
    return queries

QUERIES = _load_queries()

SYSTEM_PROMPT = (
    "你是一个专业的电商客服助手，负责解答用户关于商品、订单、物流和售后的问题。"
    "回答请简洁准确，不超过200字。"
)
MODEL_NAME = "Qwen2.5-7B-Instruct-AWQ"


class CustomerServiceUser(HttpUser):
    """
    模拟客服用户行为。
    wait_time = between(0, 0) 表示收到响应后立即发出下一个请求，
    适合测试系统的极限吞吐能力。
    若要模拟更真实的用户行为（思考时间），可改为 between(1, 3)。
    """
    wait_time = between(0, 0)

    @task
    def ask_question(self):
        question = random.choice(QUERIES)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": False,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=30,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                # 限流响应不计为失败，视为正常的背压信号
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
