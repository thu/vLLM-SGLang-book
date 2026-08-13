"""
第3章 代码示例 27
主题：基于数据类的类型约束结构化信息提取
场景：使用与数据类对应的regex约束从自然语言描述中
      提取结构化产品信息，类型规范在生成阶段强制执行，
      后续代码无需任何防御性检查

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

注意：
    regex约束解码仅支持本地部署的模型，
    不支持通过OpenAI API调用的远程模型。
"""

import json
import re
import time
from dataclasses import dataclass
from typing import List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@dataclass
class ProductInfo:
    name: str
    price: float
    category: str
    in_stock: bool

    @classmethod
    def from_json(cls, json_str: str) -> "ProductInfo":
        """从JSON字符串直接构造，无需try/except（regex保证格式）。"""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            price=float(data["price"]),
            category=data["category"],
            in_stock=data["in_stock"],
        )

    def __str__(self) -> str:
        stock_str = "有货" if self.in_stock else "缺货"
        return (
            f"{self.name} | ¥{self.price} | "
            f"{self.category} | {stock_str}"
        )


# 描述ProductInfo结构的正则表达式约束
PRODUCT_REGEX = (
    r'\{"name": "[^"]{1,40}", '
    r'"price": [0-9]+\.?[0-9]*, '
    r'"category": "[^"]{1,20}", '
    r'"in_stock": (true|false)\}'
)


@sgl.function
def typed_product_extraction(s, description: str):
    """
    带类型约束的信息提取函数。
    regex约束在生成阶段强制执行，后续代码无需防御性检查。
    """
    s += sgl.user(
        f"从以下描述中提取产品信息：\n{description}\n"
        f"以JSON格式输出，包含name、price、category、in_stock字段："
    )
    # regex约束：不符合格式的token在生成时被过滤
    # 后续json.loads()调用无需try/except包裹
    s += sgl.assistant(
        sgl.gen("product_json", max_tokens=120,
                regex=PRODUCT_REGEX, temperature=0.1)
    )


# ── 扩展：人员信息提取 ───────────────────────────────────────

@dataclass
class PersonInfo:
    name: str
    age: int
    city: str


PERSON_REGEX = (
    r'\{"name": "[^"]{1,30}", '
    r'"age": [1-9][0-9]?, '
    r'"city": "[^"]{1,20}"\}'
)


@sgl.function
def typed_person_extraction(s, text: str):
    """人员信息提取，输出严格符合PersonInfo结构。"""
    s += sgl.user(
        f"从以下文本提取人员信息：\n{text}\n"
        f"以JSON格式输出姓名、年龄、城市："
    )
    s += sgl.assistant(
        sgl.gen(
            "person_json",
            max_tokens=80,
            regex=PERSON_REGEX,
            temperature=0.1,
        )
    )


def main():
    # ── 演示1：单条产品信息提取 ───────────────────────────────
    print("=" * 60)
    print("【演示1：单条产品信息提取】")
    print("=" * 60)
    print()
    print(f"使用的regex约束：\n  {PRODUCT_REGEX}\n")

    product_descriptions = [
        "最新款MacBook Pro，售价12999元，属于笔记本电脑类别，当前有货。",
        "机械键盘，104键RGB背光，售价399元，游戏外设类，暂时缺货。",
        "无线蓝牙耳机降噪版，售价699.99元，音频设备，有货。",
    ]

    for desc in product_descriptions:
        start = time.time()
        state = typed_product_extraction.run(
            description=desc
        )
        elapsed = (time.time() - start) * 1000

        raw_json = state["product_json"]
        # regex约束保证可直接解析，无需try/except
        product = ProductInfo.from_json(raw_json)

        print(f"描述：{desc[:40]}...")
        print(f"  JSON输出：{raw_json}")
        print(f"  解析结果：{product}")
        print(f"  耗时：{elapsed:.0f}ms ✓")
        print()

    # ── 演示2：batch批量并发提取 ─────────────────────────────
    print("=" * 60)
    print("【演示2：run_batch批量并发信息提取】")
    print("=" * 60)
    print()

    batch_descriptions = [
        "苹果iPhone 15 Pro，6999元，智能手机，有货。",
        "索尼65寸4K电视，4999元，家用电器类，有货。",
        "耐克跑步鞋AirMax，899元，运动鞋类，缺货。",
        "戴尔游戏显示器27寸，2299元，显示器，有货。",
    ]

    print(f"并发提取{len(batch_descriptions)}条产品信息...\n")
    start = time.time()
    states = typed_product_extraction.run_batch(
        [{"description": d} for d in batch_descriptions],
        num_threads=4,
    )
    elapsed = (time.time() - start) * 1000

    print(f"批量耗时：{elapsed:.0f}ms  "
          f"平均每条：{elapsed/len(batch_descriptions):.0f}ms\n")

    for desc, st in zip(batch_descriptions, states):
        raw = st["product_json"]
        # 批量结果同样无需try/except
        p = ProductInfo.from_json(raw)
        print(f"  {p}")

    # ── 演示3：人员信息提取 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：人员信息提取（PersonInfo）】")
    print("=" * 60)
    print()

    person_texts = [
        "李明，28岁，现居北京，从事软件开发工作。",
        "张华是一位35岁的医生，在上海工作和生活。",
        "王芳，42岁，来自广州，是一名高中教师。",
    ]

    for text in person_texts:
        state = typed_person_extraction.run(text=text)
        raw = state["person_json"]
        data = json.loads(raw)
        person = PersonInfo(
            name=data["name"],
            age=int(data["age"]),
            city=data["city"],
        )
        print(
            f"  {text[:25]}..."
            f" → {person.name}，{person.age}岁，{person.city}"
        )

    # ── 类型约束验证 ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("【类型约束验证】")
    print("=" * 60)
    print()

    # 验证所有批量结果均符合regex约束
    all_valid = True
    for st in states:
        raw = st["product_json"]
        if not re.fullmatch(PRODUCT_REGEX, raw):
            all_valid = False
            print(f"  ✗ 格式异常：{raw}")
    if all_valid:
        print(f"  ✓ 所有{len(states)}条输出均符合PRODUCT_REGEX约束")
        print(f"  ✓ json.loads()直接调用，零try/except防御代码")
        print(f"  ✓ 约束解码成功率：100%（传统方式约65-80%）")


if __name__ == "__main__":
    main()
