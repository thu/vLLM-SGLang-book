#!/usr/bin/env python3
# scripts/verify_radix_attention.py
# 验证 RadixAttention 前缀缓存效果
# 使用方法：python scripts/verify_radix_attention.py
#
# 实验设计：
# - 基线组（pre_flush=True）：每次测试前清空 KV Cache，模拟无缓存状态
# - 实验组（pre_flush=False）：正常批量流程，缓存在批次内持续累积
# - 两组使用相同的 100 份简历文本，消除内容差异的干扰
#
# 注意：
# - /flush_cache 端点在 SGLang v0.4 中可用
#   若遇到 404，先执行 curl http://127.0.0.1:30000/ 确认可用端点
# - 指标名称以 curl http://127.0.0.1:30000/metrics 的实际输出为准
#   若下列名称不匹配，请对照实际输出修改 METRIC_NAMES

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.sglang_extractor import extract_resume

SGLANG_BASE = "http://127.0.0.1:30000"

# SGLang v0.4 的 Prometheus 指标名称（以实际 /metrics 输出为准）
METRIC_NAMES = {
    "cache_hit_rate":   "sglang:cache_hit_rate",
    "num_running_reqs": "sglang:num_running_reqs",
    "token_usage":      "sglang:token_usage",
}


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def flush_cache() -> bool:
    """
    清空 KV Cache。
    返回 True 表示成功，False 表示端点不可用。
    """
    try:
        resp = requests.post(f"{SGLANG_BASE}/flush_cache", timeout=5)
        if resp.status_code == 200:
            time.sleep(2)   # 等待缓存完全清空
            return True
        else:
            print(f"  警告：/flush_cache 返回 {resp.status_code}")
            print("  提示：先执行 curl http://127.0.0.1:30000/ 确认可用端点列表")
            return False
    except Exception as e:
        print(f"  警告：无法连接 /flush_cache：{e}")
        return False


def read_metric(name: str) -> float:
    """
    从 /metrics 端点读取指定指标的当前值。
    若指标不存在返回 -1.0（用于区分"值为0"和"未找到"）。
    """
    try:
        text = requests.get(f"{SGLANG_BASE}/metrics", timeout=5).text
    except Exception:
        return -1.0

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                continue
    return -1.0


def print_available_metrics():
    """打印 /metrics 端点的原始输出，帮助确认实际指标名称"""
    try:
        text = requests.get(f"{SGLANG_BASE}/metrics", timeout=5).text
        print("\n── /metrics 原始输出（前 40 行）──")
        for line in text.splitlines()[:40]:
            print(f"  {line}")
    except Exception as e:
        print(f"  无法获取 /metrics：{e}")


# ──────────────────────────────────────────────
# 测试简历生成
# ──────────────────────────────────────────────

def generate_test_resumes(n: int = 100) -> list[str]:
    """
    生成 n 份差异化的测试简历文本。
    每份简历的系统提示前缀完全相同（由 extract_resume 内部固定），
    只有简历正文部分不同，确保 RadixAttention 能够命中前缀缓存。
    """
    templates = [
        """\
{name}
电话：138-{phone:04d}-{phone2:04d}
邮箱：{name_lower}{idx}@example.com

教育背景
2016-2020  北京大学  计算机科学  本科

工作经历
2020年7月 - 2022年6月  腾讯科技  后端工程师
- 负责社交平台后端服务开发
- 系统 QPS 优化至 10 万

技能
编程语言：Python、Go
框架：Django、Gin
数据库：MySQL、Redis
工具：Docker、Git""",

        """\
{name}
邮箱：{name_lower}{idx}@company.com
电话：159-{phone:04d}-{phone2:04d}

教育背景
2015-2019  复旦大学  软件工程  本科

工作经历
2019年6月至今  美团  算法工程师
- 负责推荐算法研发，召回率提升 18%
- 主导 A/B 测试平台建设

技能
编程语言：Python、Java
框架：TensorFlow、Spring Boot
数据库：MongoDB、Redis
工具：Kubernetes、Jenkins""",

        """\
{name}
地址：上海市浦东新区
电话：186-{phone:04d}-{phone2:04d}

教育背景
2017-2021  上海交通大学  信息安全  本科

工作经历
2021年8月 - 2023年12月  字节跳动  安全工程师
- 负责业务安全风控系统
- 拦截率提升 35%，误报率降低 20%

技能
编程语言：Python、C++
框架：Flask
数据库：PostgreSQL、Redis
工具：Docker、Git""",
    ]

    names_male   = ["张伟", "李明", "王强", "刘洋", "陈磊", "赵鑫", "孙浩", "周杰"]
    names_female = ["王芳", "李娜", "张丽", "刘梅", "陈静", "赵雪", "孙燕", "周晓"]
    all_names    = names_male + names_female

    resumes = []
    for i in range(n):
        tpl   = templates[i % len(templates)]
        name  = all_names[i % len(all_names)]
        phone = (i * 137 + 1000) % 9000 + 1000   # 4位伪随机数
        phone2 = (i * 251 + 2000) % 9000 + 1000

        text = tpl.format(
            name=name,
            name_lower=name,
            idx=i,
            phone=phone,
            phone2=phone2,
        )
        resumes.append(text)

    return resumes


# ──────────────────────────────────────────────
# 实验执行
# ──────────────────────────────────────────────

async def run_experiment(
    texts: list[str],
    label: str,
    pre_flush: bool,
) -> dict:
    """
    运行一组实验，返回耗时、吞吐量和缓存命中率。

    Args:
        texts:      测试简历文本列表
        label:      实验标签（用于日志）
        pre_flush:  是否在实验前清空 KV Cache
    """
    if pre_flush:
        success = flush_cache()
        status = "已清空" if success else "清空失败（将继续实验）"
        print(f"  [{label}] KV Cache {status}")
    else:
        print(f"  [{label}] 使用现有缓存")

    t0 = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks   = [extract_resume(t, session) for t in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed    = time.perf_counter() - t0
    success_n  = sum(1 for r in results if not isinstance(r, Exception))
    failed_n   = len(results) - success_n
    throughput = success_n / elapsed          # req/s
    per_hour   = throughput * 3600            # 份/小时

    # 读取缓存命中率
    hit_rate = read_metric(METRIC_NAMES["cache_hit_rate"])

    print(f"  [{label}] 完成：{success_n}/{len(texts)} 成功，{failed_n} 失败")
    print(f"  [{label}] 耗时：{elapsed:.1f}s | 吞吐：{throughput:.2f} req/s（{per_hour:.0f} 份/小时）")
    if hit_rate >= 0:
        print(f"  [{label}] 缓存命中率：{hit_rate:.1%}")
    else:
        print(f"  [{label}] 缓存命中率：无法读取（请检查指标名称）")

    return {
        "label":        label,
        "n_success":    success_n,
        "n_total":      len(texts),
        "elapsed_s":    round(elapsed, 2),
        "throughput_rps":   round(throughput, 3),
        "throughput_per_h": round(per_hour, 0),
        "cache_hit_rate":   hit_rate,
    }


async def main():
    print("=" * 60)
    print("RadixAttention 前缀缓存效果验证")
    print("=" * 60)
    print(f"SGLang 服务：{SGLANG_BASE}")

    # 打印实际指标名称，帮助确认
    print_available_metrics()

    # 生成测试简历
    print("\n生成测试简历...")
    texts = generate_test_resumes(n=50)   # 使用 50 份，加快验证速度
    print(f"生成完成：{len(texts)} 份")

    print("\n── 实验1：无缓存基线（每次测试前清空 KV Cache）──")
    result_baseline = await run_experiment(texts, "无缓存", pre_flush=True)

    print("\n── 实验2：启用缓存（正常批量流程）──")
    result_cached   = await run_experiment(texts, "有缓存", pre_flush=False)

    # ── 对比分析 ──
    print("\n── 对比分析 ──")
    baseline_tp = result_baseline["throughput_per_h"]
    cached_tp   = result_cached["throughput_per_h"]

    if baseline_tp > 0:
        improvement = (cached_tp - baseline_tp) / baseline_tp * 100
        print(f"  基线吞吐量：{baseline_tp:.0f} 份/小时")
        print(f"  缓存吞吐量：{cached_tp:.0f} 份/小时")
        print(f"  吞吐量提升：{improvement:+.1f}%")
    else:
        print("  基线吞吐量为零，无法计算提升比例")

    # ── 保存结果 ──
    output_path = Path(__file__).parent.parent / "results" / "radix_attention_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"baseline": result_baseline, "cached": result_cached},
            f, ensure_ascii=False, indent=2
        )
    print(f"\n结果已保存：{output_path}")

    print("\n" + "=" * 60)
    print("说明：具体数值受硬件配置和简历长度影响，")
    print("以上结果为本次实际运行数据，供参考。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
