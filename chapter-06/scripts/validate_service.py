#!/usr/bin/env python3
# ch06/scripts/validate_service.py
# 验证 vLLM 服务健康状态与基本推理能力
#
# 依赖：httpx（pip install httpx）
# 运行：python ch06/scripts/validate_service.py
#
# 返回码：0 表示全部检查通过，非零表示存在问题

import sys
import time
import httpx

VLLM_URL = "http://localhost:8000"
MODEL_NAME = "Qwen2.5-7B-Instruct-AWQ"
EXPECTED_MAX_MODEL_LEN = 4096


def check_health(client: httpx.Client) -> bool:
    """
    检查服务健康状态。
    vLLM 0.6.x 在 /health 端点返回 200 表示服务就绪。
    """
    try:
        resp = client.get("/health", timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  健康检查异常: {e}")
        return False


def check_model_config(client: httpx.Client) -> dict | None:
    """
    查询已加载的模型配置。

    关键检查项：max_model_len 是否与启动参数一致。
    当显存不足时，vLLM 会静默降低 max_model_len 并继续启动，
    不检查这个值可能导致运行期出现意外的序列截断。
    """
    try:
        resp = client.get("/v1/models", timeout=10)
        data = resp.json()
        models = data.get("data", [])
        if not models:
            return None
        return models[0]
    except Exception as e:
        print(f"  模型信息查询异常: {e}")
        return None


def warmup_inference(client: httpx.Client) -> float | None:
    """
    发送预热请求，测量冷启动延迟。

    vLLM 首次推理会触发 CUDA kernel 编译（torch.compile），
    这次延迟通常远高于稳态延迟，不应计入正式的延迟基线。
    预热完成后，正式压测前的首次请求延迟才能反映真实性能。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
        "max_tokens": 30,
        "temperature": 0.1,
    }
    try:
        t0 = time.perf_counter()
        resp = client.post("/v1/chat/completions", json=payload, timeout=120)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            print(f"  推理请求失败: {resp.status_code} {resp.text[:200]}")
            return None
        return latency_ms
    except Exception as e:
        print(f"  推理请求异常: {e}")
        return None


def main():
    all_passed = True

    with httpx.Client(base_url=VLLM_URL) as client:

        # --- 检查 1：健康状态 ---
        print("[1/3] 健康检查...")
        if check_health(client):
            print("  ✓ 服务健康")
        else:
            print("  ✗ 服务不健康或未就绪")
            all_passed = False

        # --- 检查 2：模型配置 ---
        print("[2/3] 模型配置验证...")
        model_info = check_model_config(client)
        if model_info is None:
            print("  ✗ 无法获取模型信息")
            all_passed = False
        else:
            actual_max_len = model_info.get("max_model_len")
            print(f"  模型 ID:          {model_info.get('id')}")
            print(f"  max_model_len:    {actual_max_len}")
            if actual_max_len is not None and actual_max_len < EXPECTED_MAX_MODEL_LEN:
                print(
                    f"  ✗ max_model_len ({actual_max_len}) 低于预期 "
                    f"({EXPECTED_MAX_MODEL_LEN})，"
                    f"可能是显存不足导致 vLLM 自动降级"
                )
                all_passed = False
            else:
                print("  ✓ 模型配置正常")

        # --- 检查 3：推理预热 ---
        print("[3/3] 推理预热...")
        warmup_latency = warmup_inference(client)
        if warmup_latency is None:
            print("  ✗ 预热推理失败")
            all_passed = False
        else:
            print(f"  ✓ 预热完成，冷启动延迟: {warmup_latency:.0f} ms")
            if warmup_latency > 30000:
                print(
                    f"  ⚠ 冷启动延迟超过 30s，请确认 CUDA kernel 编译是否完成"
                )

    print()
    if all_passed:
        print("=== 全部检查通过，服务就绪 ===")
        sys.exit(0)
    else:
        print("=== 存在检查失败项，请排查后重试 ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
