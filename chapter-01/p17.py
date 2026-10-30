class HardwareAbstraction:
    """
    硬件抽象层示例：统一不同硬件平台的接口。
    """

    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        self.capabilities = self._detect_capabilities()

    def _detect_capabilities(self):
        """检测硬件能力。"""
        caps = {
            "device_type": self.device_type,
            "compute_capability": None,
            "memory_gb": 0,
            "supports_fp16": False,
            "supports_int8": False,
            "tensor_cores": False
        }

        if self.device_type == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)

                    caps["compute_capability"] = (
                        f"{props.major}.{props.minor}"
                    )
                    caps["memory_gb"] = (
                        props.total_memory / (1024 ** 3)
                    )
                    caps["supports_fp16"] = True
                    caps["supports_int8"] = props.major >= 7
                    caps["tensor_cores"] = props.major >= 7

            except ImportError:
                pass

        return caps

    def get_optimal_batch_size(self, model_size_gb, seq_length):
        """根据硬件能力推荐最优批处理大小。"""
        available_memory = self.capabilities["memory_gb"]

        # 预留 20% 内存，用于系统开销和模型运行时产生的临时数据
        usable_memory = available_memory * 0.8

        # 减去模型权重占用
        kv_cache_budget = usable_memory - model_size_gb

        # 估算单个请求的 KV 缓存（简化计算）
        kv_per_request = seq_length * 0.001  # GB（粗略估计）

        max_batch = int(kv_cache_budget / kv_per_request)

        # 考虑 Tensor Core 的限制（批大小最好是 8 的倍数）
        if self.capabilities["tensor_cores"]:
            max_batch = (max_batch // 8) * 8

        return max(1, min(max_batch, 256))  # 限制在 [1, 256]

    def select_dtype(self, precision_requirement="balanced"):
        """根据精度需求和硬件能力选择数据类型。"""
        if precision_requirement == "high":
            return "fp32"

        elif precision_requirement == "balanced":
            if self.capabilities["supports_fp16"]:
                return "fp16"
            else:
                return "fp32"

        elif precision_requirement == "low":
            if self.capabilities["supports_int8"]:
                return "int8"
            elif self.capabilities["supports_fp16"]:
                return "fp16"
            else:
                return "fp32"

        return "fp32"


def demonstrate_hardware_abstraction():
    """演示硬件抽象层的使用。"""
    hal = HardwareAbstraction("cuda")

    print("=== 硬件能力检测 ===\n")

    for key, value in hal.capabilities.items():
        print(f"{key}: {value}")

    print("\n=== 配置推荐 ===\n")

    # 假设使用 7B 参数规模的模型，在 FP16 精度下约占用 14GB 显存
    model_size = 14
    seq_len = 2048

    batch_size = hal.get_optimal_batch_size(model_size, seq_len)
    dtype = hal.select_dtype("balanced")

    print(f"模型大小：{model_size}GB")
    print(f"序列长度：{seq_len}")
    print(f"推荐批大小：{batch_size}")
    print(f"推荐数据类型：{dtype}")


if __name__ == "__main__":
    demonstrate_hardware_abstraction()
