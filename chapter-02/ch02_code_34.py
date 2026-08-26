import requests
import time
from prometheus_client.parser import text_string_to_metric_families


def monitor_vllm_metrics(server_url: str = "http://localhost:8000"):
    """
    监控vLLM服务器指标
    功能：定期收集并展示性能指标
    """
    metrics_url = f"{server_url}/metrics"

    while True:
        try:
            # 获取指标
            response = requests.get(metrics_url)
            metrics_text = response.text

            # 解析指标
            metrics = {}
            for family in text_string_to_metric_families(metrics_text):
                for sample in family.samples:
                    metrics[sample.name] = sample.value

            # 显示关键指标
            print(f"\n=== vLLM监控 {time.strftime('%H:%M:%S')} ===")
            print(f"运行请求数：{metrics.get('vllm_num_requests_running', 0):.0f}")
            print(f"等待请求数：{metrics.get('vllm_num_requests_waiting', 0):.0f}")
            print(f"GPU缓存使用率：{metrics.get('vllm_gpu_cache_usage_perc', 0):.1%}")
            print(f"吞吐量：{metrics.get('vllm_avg_generation_throughput_toks_per_s', 0):.1f} tokens/s")

            time.sleep(5)  # 每5秒更新一次

        except Exception as e:
            print(f"监控错误：{e}")
            time.sleep(5)


# 使用示例
if __name__ == "__main__":
    print("开始监控vLLM服务器...")
    monitor_vllm_metrics()