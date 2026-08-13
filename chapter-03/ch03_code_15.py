"""
第3章 代码示例 15
主题：传统JSON生成方式的痛点——生成-验证-重试循环
场景：演示不使用约束解码时获取合法JSON输出的传统做法，
      量化其失败率、重试开销与代码复杂度问题

运行前提：
    pip install openai
    export OPENAI_API_KEY="your-key"

说明：
    如未配置API密钥，程序将进入模拟模式，
    通过概率模型模拟真实的失败/重试行为，
    无需实际API调用即可观察统计结果。
"""

import json
import os
import random
import time
from typing import Optional, Tuple

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "")
)


def generate_json_traditional(
    prompt: str, max_retries: int = 3
) -> dict:
    """
    传统方式：生成-验证-重试循环。
    每次重试都是一次完整的LLM调用，延迟和成本线性增加。
    成功率通常在60-80%之间，无法保证100%。
    """
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"{prompt}\n请以JSON格式返回。"
            }]
        )
        output = response.choices[0].message.content
        # 剥离模型常见的markdown代码块包裹
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        try:
            return json.loads(output.strip())
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                # 失败后追加提示词重试，浪费算力且延迟翻倍
                prompt += "\n注意：请确保输出为严格合法的JSON格式。"
    raise ValueError(f"经{max_retries}次尝试仍无法生成合法JSON")


# ── 模拟模式（无需API密钥）──────────────────────────────────

class SimulatedJSONGenerator:
    """
    模拟传统JSON生成器的行为，无需实际API调用。
    用于在没有API密钥的环境中演示统计效果。
    每次尝试按 success_rate 概率成功。
    """

    def __init__(
        self,
        success_rate: float = 0.72,
        base_latency_ms: float = 450.0,
        seed: int = 42,
    ):
        self.success_rate = success_rate
        self.base_latency_ms = base_latency_ms
        self._rng = random.Random(seed)
        self._total_attempts = 0
        self._total_successes = 0
        self._total_retries = 0

    def generate(
        self, prompt: str, max_retries: int = 3
    ) -> Tuple[Optional[dict], dict]:
        """
        模拟生成过程。
        返回 (结果或None, 统计信息字典)。
        """
        total_latency = 0.0
        attempt_details = []

        for attempt in range(max_retries):
            self._total_attempts += 1
            # 模拟API调用延迟（加入随机抖动）
            latency = self.base_latency_ms + self._rng.gauss(0, 40)
            total_latency += latency

            if self._rng.random() < self.success_rate:
                self._total_successes += 1
                attempt_details.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "latency_ms": latency,
                })
                return {"simulated": True, "attempt": attempt + 1}, {
                    "success": True,
                    "attempts": attempt + 1,
                    "retries": attempt,
                    "total_latency_ms": total_latency,
                    "attempt_details": attempt_details,
                }
            else:
                self._total_retries += 1
                attempt_details.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "latency_ms": latency,
                    "error": "json.JSONDecodeError（模拟）",
                })

        return None, {
            "success": False,
            "attempts": max_retries,
            "retries": max_retries - 1,
            "total_latency_ms": total_latency,
            "attempt_details": attempt_details,
        }

    def get_stats(self) -> dict:
        total = self._total_attempts
        success_rate = (
            self._total_successes / total if total > 0 else 0.0
        )
        avg_latency = (
            self.base_latency_ms
            * (1 + (1 - self.success_rate))
        )
        return {
            "total_requests": total,
            "successful": self._total_successes,
            "success_rate": success_rate,
            "total_retries": self._total_retries,
            "avg_retries_per_request": (
                self._total_retries / total if total > 0 else 0.0
            ),
            "estimated_avg_latency_ms": avg_latency,
        }


def run_simulation(n: int = 50):
    """运行n次模拟请求，统计整体表现。"""
    gen = SimulatedJSONGenerator(success_rate=0.72)
    prompts = [f"生成JSON示例请求#{i}" for i in range(n)]

    retry_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for prompt in prompts:
        _, stats = gen.generate(prompt, max_retries=3)
        retries = stats["retries"]
        if retries in retry_dist:
            retry_dist[retries] += 1

    return gen.get_stats(), retry_dist


def main():
    use_api = bool(os.environ.get("OPENAI_API_KEY"))

    if use_api:
        print("=" * 60)
        print("【真实API模式：传统JSON生成重试演示】")
        print("=" * 60)
        print()

        test_prompts = [
            "生成一个包含name(字符串)、age(整数)、city(字符串)的人员信息",
            "生成一个包含title、year(整数)、rating(小数)的电影信息",
            "生成一个包含product、price(数字)、in_stock(布尔)的商品信息",
        ]

        for prompt in test_prompts:
            start = time.time()
            try:
                result = generate_json_traditional(
                    prompt, max_retries=3
                )
                elapsed = (time.time() - start) * 1000
                print(f"✓ 成功  耗时：{elapsed:.0f}ms  "
                      f"结果：{result}")
            except ValueError as e:
                elapsed = (time.time() - start) * 1000
                print(f"✗ 失败  耗时：{elapsed:.0f}ms  {e}")
            print()
    else:
        print("=" * 60)
        print("【模拟模式：传统JSON生成统计演示】")
        print("（未检测到OPENAI_API_KEY，使用概率模拟）")
        print("=" * 60)
        print()
        print("模拟参数：每次尝试成功率72%，基础延迟450ms\n")

        stats, retry_dist = run_simulation(n=100)

        print(f"模拟100次JSON生成请求的统计结果：\n")
        print(f"  总请求数        ：{stats['total_requests']}")
        print(f"  成功次数        ：{stats['successful']}")
        print(f"  最终成功率      ：{stats['success_rate']:.1%}")
        print(f"  总重试次数      ：{stats['total_retries']}")
        print(
            f"  平均重试次数/请求：{stats['avg_retries_per_request']:.2f}"
        )
        print(
            f"  估算平均端到端延迟：{stats['estimated_avg_latency_ms']:.0f}ms"
        )
        print()
        print("重试次数分布：")
        for retries, count in sorted(retry_dist.items()):
            bar = "█" * count
            print(f"  重试{retries}次：{count:3d}次  {bar}")

    # ── 痛点总结 ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("【传统方式的系统性痛点】")
    print("=" * 60)
    print()
    print("痛点1：成功率不稳定（65-80%），严重依赖提示词质量")
    print("痛点2：每次重试为完整LLM调用，延迟和成本线性叠加")
    print("痛点3：补救提示词增加额外token消耗")
    print("痛点4：无法从根本上保证100%成功率")
    print("痛点5：大量try/except和格式处理代码，维护成本高")
    print()
    print("SGLang约束解码的解决方案：")
    print("  在token生成阶段实时过滤，成功率100%，零重试，")
    print("  彻底消除格式验证代码（详见后续示例）。")


if __name__ == "__main__":
    main()
