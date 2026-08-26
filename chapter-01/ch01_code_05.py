from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def basic_inference_example():
    """
    使用HuggingFace Transformers进行基础推理
    """
    # 加载模型和分词器
    model_name = "meta-llama/Llama-2-7b-hf"  # 示例模型

    print("=== HuggingFace Transformers基础推理 ===\n")
    print(f"加载模型: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 准备输入
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成输出
        print(f"\n输入: {prompt}")
        print("生成中...")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n输出: {generated_text}")

        # 性能分析
        print("\n性能特点:")
        print("  ✓ 易于使用,代码简洁")
        print("  ✗ 无批处理优化")
        print("  ✗ KV Cache手动管理")
        print("  ✗ GPU利用率低 (~10-20%)")

    except Exception as e:
        print(f"注意: 此示例需要实际的模型文件: {e}")
        print("在生产环境中,请确保已下载模型")


# 运行示例(注意:需要实际模型)
# basic_inference_example()

# 性能对比模拟
def compare_framework_performance():
    """
    对比不同框架的推理性能
    """
    frameworks = {
        "原生PyTorch": {
            "吞吐量(tokens/s)": 100,
            "GPU利用率(%)": 15,
            "批处理支持": "手动",
            "KV Cache": "手动管理"
        },
        "HuggingFace": {
            "吞吐量(tokens/s)": 150,
            "GPU利用率(%)": 20,
            "批处理支持": "有限",
            "KV Cache": "自动"
        },
        "ONNX Runtime": {
            "吞吐量(tokens/s)": 300,
            "GPU利用率(%)": 35,
            "批处理支持": "静态",
            "KV Cache": "优化"
        },
        "TensorRT": {
            "吞吐量(tokens/s)": 450,
            "GPU利用率(%)": 50,
            "批处理支持": "静态",
            "KV Cache": "高度优化"
        },
        "vLLM": {
            "吞吐量(tokens/s)": 1000,
            "GPU利用率(%)": 85,
            "批处理支持": "连续",
            "KV Cache": "PagedAttention"
        },
        "SGLang": {
            "吞吐量(tokens/s)": 950,
            "GPU利用率(%)": 80,
            "批处理支持": "灵活",
            "KV Cache": "RadixAttention"
        }
    }

    print("=== 推理框架性能对比 ===\n")
    print(f"{'框架':<20} {'吞吐量':<15} {'GPU利用率':<12} {'批处理':<12} {'KV Cache':<15}")
    print("-" * 80)

    for name, metrics in frameworks.items():
        print(f"{name:<20} {metrics['吞吐量(tokens/s)']:<15} "
              f"{metrics['GPU利用率(%)']:<12} "
              f"{metrics['批处理支持']:<12} {metrics['KV Cache']:<15}")

    print("\n关键观察:")
    print("  1. 专用推理引擎相比通用框架有5-10倍性能提升")
    print("  2. 连续批处理是吞吐量提升的关键")
    print("  3. 高效KV Cache管理直接影响GPU利用率")


compare_framework_performance()