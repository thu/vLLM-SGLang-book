# 文件名: scripts/common/preprocess_datasets.py
# 功能: 将原始数据集转换为统一的测试格式
# 用法: python preprocess_datasets.py
# 依赖: transformers

import json
import random
from pathlib import Path
from typing import Dict, List


class DatasetPreprocessor:
    """
    数据集预处理器，负责将原始数据统一转换为测试用格式。

    生成两类数据集：
    1. ShareGPT 按对话轮次分组抽样（单轮、3轮、5轮、10+轮）
    2. 合成可控长度数据集（短短、短长、长短、长长）
    """

    def __init__(self, project_root: Path = None):
        if project_root is None:
            project_root = Path.home() / "llm-benchmark"
        self.raw_dir = project_root / "datasets" / "raw"
        self.out_dir = project_root / "datasets" / "processed"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root

    # ------------------------------------------------------------------
    # ShareGPT 处理
    # ------------------------------------------------------------------

    def process_sharegpt(self) -> Dict[str, List]:
        """
        处理 ShareGPT 数据集，按对话轮次分组后各抽取固定数量样本。

        轮次分组的意义：单轮对话测试基础延迟，多轮对话测试前缀
        缓存命中率对性能的影响。随着对话轮次增加，连续请求之间的
        公共前缀越长，RadixAttention 的优势应越明显。

        返回值为字典，键为场景名称，值为对应的对话列表。
        """
        print("处理 ShareGPT 数据集...")

        sharegpt_path = self.raw_dir / "sharegpt.json"
        if not sharegpt_path.exists():
            print(f"  ✗ 未找到 {sharegpt_path}，请先运行 download_sharegpt.sh")
            return {}

        with open(sharegpt_path, encoding="utf-8") as f:
            raw = json.load(f)

        # 按轮次分组，过滤不以 human 开头的对话
        by_turns: Dict[int, List] = {}
        skipped = 0

        for conv in raw:
            turns_raw = conv.get("conversations", [])
            if len(turns_raw) < 2:
                skipped += 1
                continue
            if turns_raw[0].get("from") != "human":
                skipped += 1
                continue

            # 将交替的 human/gpt 消息整理为 (user, assistant) 对列表
            pairs = []
            for i in range(0, len(turns_raw) - 1, 2):
                if (i + 1 < len(turns_raw)
                        and turns_raw[i].get("from") == "human"
                        and turns_raw[i + 1].get("from") == "gpt"):
                    pairs.append({
                        "user": turns_raw[i]["value"],
                        "assistant": turns_raw[i + 1]["value"],
                    })

            if pairs:
                n = len(pairs)
                by_turns.setdefault(n, []).append(pairs)

        print(f"  有效对话: {sum(len(v) for v in by_turns.values())}，"
              f"跳过: {skipped}")

        # 按场景抽样
        random.seed(42)  # 固定随机种子保证可重复性
        samples = {}

        sample_configs = [
            ("single_turn",   1,    100),
            ("multi_turn_3",  3,    100),
            ("multi_turn_5",  5,    100),
        ]

        for name, target_turns, count in sample_configs:
            pool = by_turns.get(target_turns, [])
            actual = min(count, len(pool))
            samples[name] = random.sample(pool, actual) if pool else []

        # 10 轮及以上的对话合并后抽取
        long_convs = [
            c for n, cs in by_turns.items() if n >= 10 for c in cs
        ]
        actual_long = min(50, len(long_convs))
        samples["multi_turn_10"] = (
            random.sample(long_convs, actual_long) if long_convs else []
        )

        # 写出各场景文件
        for name, data in samples.items():
            out = self.out_dir / f"sharegpt_{name}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {out.name}（{len(data)} 条）")

        return samples

    # ------------------------------------------------------------------
    # 合成数据集生成
    # ------------------------------------------------------------------

    def create_synthetic_dataset(self) -> None:
        """
        生成四类可控长度的合成数据集。

        通过精确控制输入/输出 token 数，可以将"输入长度"和
        "输出长度"作为自变量，分离它们对 TTFT 和总延迟的影响。

        四种场景：
        - 短输入短输出 (50/50): 调度开销主导
        - 短输入长输出 (50/500): 解码效率主导
        - 长输入短输出 (2000/50): 预填充效率主导（TTFT 关键场景）
        - 长输入长输出 (2000/500): 全流程综合
        """
        from transformers import AutoTokenizer

        model_dir = (
            self.project_root / "models" / "llama-3.1-8b-instruct"
        )
        if not model_dir.exists():
            print(f"  ✗ 模型目录不存在: {model_dir}")
            print("  请先运行 download_model.sh")
            return

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # 基础句子池（语义单一但语法完整，用于填充输入文本）
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming modern industries.",
            "Deep learning models require large amounts of training data.",
            "Natural language processing enables machines to understand text.",
            "Performance benchmarking requires careful experimental design.",
            "Cloud computing provides scalable infrastructure for AI workloads.",
            "Transformer architectures have revolutionized sequence modeling.",
            "Efficient memory management is critical for serving large models.",
        ]

        scenarios = [
            ("short_input_short_output",  50,   50),
            ("short_input_long_output",   50,  500),
            ("long_input_short_output",  2000,  50),
            ("long_input_long_output",   2000, 500),
        ]

        print("生成合成数据集...")
        random.seed(42)

        for name, in_len, out_len in scenarios:
            samples = []
            for idx in range(100):
                # 拼接句子直至超过目标输入 token 数，再截断
                text = ""
                while len(tokenizer.encode(text)) < in_len:
                    text += random.choice(sentences) + " "
                ids = tokenizer.encode(text)[:in_len]
                input_text = tokenizer.decode(
                    ids, skip_special_tokens=True
                )

                samples.append({
                    "id": f"{name}_{idx:03d}",
                    "input": input_text,
                    "target_output_tokens": out_len,
                    "actual_input_tokens": len(
                        tokenizer.encode(input_text)
                    ),
                })

            out = self.out_dir / f"synthetic_{name}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            actual_in = samples[0]["actual_input_tokens"]
            print(f"  ✓ {out.name}（100 条，"
                  f"输入≈{actual_in} tokens，"
                  f"输出限制 {out_len} tokens）")

    # ------------------------------------------------------------------
    # 统计报告
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """打印所有已生成数据集的统计信息。"""
        print("\n已生成的数据集:")
        print("-" * 50)
        for f in sorted(self.out_dir.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
            print(f"  {f.name}: {len(data)} 条")


def main():
    preprocessor = DatasetPreprocessor()
    preprocessor.process_sharegpt()
    preprocessor.create_synthetic_dataset()
    preprocessor.print_summary()
    print("\n✓ 全部数据集预处理完成")


if __name__ == "__main__":
    main()
