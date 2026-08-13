"""
第3章 代码示例 12
主题：编译时静态前缀分析——自动识别可缓存前缀模式
场景：通过解析SGLang函数的AST，在编译阶段识别
      纯字符串常量（static）和f-string模板（template），
      为运行时RadixAttention缓存提供静态分析依据

说明：
    本示例为教学性实现，展示编译时分析的核心逻辑。
    SGLang生产环境中的编译器与Python解释器深度集成。
    无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class PrefixPattern:
    # "static"表示纯常量，可直接跨请求缓存KV Cache
    # "template"表示含变量插值，骨架部分可参数化缓存
    pattern_type: str
    content: str
    variables: Set[str] = field(default_factory=set)

    @property
    def estimated_tokens(self) -> int:
        """粗略估算token数（中文约1.5/字符，英文约0.25/字符）。"""
        cn = sum(1 for c in self.content if '\u4e00' <= c <= '\u9fff')
        other = len(self.content) - cn
        return max(1, int(cn * 1.5 + other * 0.25))

    def __repr__(self) -> str:
        var_info = (
            f", vars={self.variables}" if self.variables else ""
        )
        preview = self.content[:40].replace('\n', '\\n')
        return (
            f"PrefixPattern(type={self.pattern_type!r}, "
            f"content={preview!r}"
            f"{var_info}, ~{self.estimated_tokens}tokens)"
        )


class SGLangCompiler:
    def __init__(self):
        self.prefix_patterns: List[PrefixPattern] = []

    def analyze_function(self, func):
        # 获取函数源码并解析为AST
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        self._walk(tree)
        return self.prefix_patterns

    def _walk(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign):
                self._analyze_expression(child.value)

    def _analyze_expression(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                # 纯字符串常量：所有请求共享，直接缓存KV Cache
                self.prefix_patterns.append(PrefixPattern(
                    pattern_type="static",
                    content=node.value
                ))
        elif isinstance(node, ast.JoinedStr):
            # f-string：提取骨架与变量，参数化缓存骨架部分
            parts, variables = [], set()
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    var_name = ast.unparse(value.value)
                    parts.append(f"{{{var_name}}}")
                    variables.add(var_name)
            self.prefix_patterns.append(PrefixPattern(
                pattern_type="template",
                content="".join(parts),
                variables=variables
            ))
        elif isinstance(node, ast.BinOp):
            # 字符串拼接（+运算符）：递归分析两侧
            self._analyze_expression(node.left)
            self._analyze_expression(node.right)

    def get_summary(self) -> dict:
        """返回分析结果的统计摘要。"""
        static_count = sum(
            1 for p in self.prefix_patterns
            if p.pattern_type == "static"
        )
        template_count = sum(
            1 for p in self.prefix_patterns
            if p.pattern_type == "template"
        )
        total_tokens = sum(
            p.estimated_tokens for p in self.prefix_patterns
        )
        return {
            "total_patterns": len(self.prefix_patterns),
            "static_patterns": static_count,
            "template_patterns": template_count,
            "estimated_cacheable_tokens": total_tokens,
        }


# ── 供分析演示用的模拟SGLang函数 ─────────────────────────────

def demo_translate(s, text: str, target_lang: str):
    """含纯静态常量和f-string模板的翻译函数（供AST分析）。"""
    # 纯静态字符串：跨所有请求共享
    s += "你是专业翻译，翻译准确自然，保持原文风格。\n"
    # f-string模板：骨架静态，变量动态
    s += f"请将以下文本翻译为{target_lang}：\n"
    s += f"原文：{text}\n译文："


def demo_few_shot(s, query: str):
    """纯静态示例 + 动态查询（供AST分析）。"""
    s += "以下是情感分类示例：\n\n"
    s += "输入：这产品非常好用！\n输出：积极\n\n"
    s += "输入：质量很差，失望。\n输出：消极\n\n"
    s += "输入：一般般，没特别。\n输出：中性\n\n"
    s += f"输入：{query}\n输出："


def demo_multi_step(s, question: str, context: str):
    """多步骤函数：静态指令 + 多个f-string（供AST分析）。"""
    s += "你是一位专业的问答助手。\n"
    s += f"背景信息：{context}\n\n"
    s += f"问题：{question}\n"
    s += "请给出详细回答：\n"


def main():
    print("=" * 60)
    print("【演示1：翻译函数的编译时分析】")
    print("=" * 60)
    print()

    compiler1 = SGLangCompiler()
    patterns1 = compiler1.analyze_function(demo_translate)
    summary1 = compiler1.get_summary()

    print(f"识别到 {summary1['total_patterns']} 个前缀模式：")
    print(f"  静态模式：{summary1['static_patterns']} 个")
    print(f"  模板模式：{summary1['template_patterns']} 个")
    print(
        f"  预估可缓存token数："
        f"{summary1['estimated_cacheable_tokens']}"
    )
    print("\n详细模式：")
    for i, p in enumerate(patterns1, 1):
        print(f"  {i}. {p}")

    print()
    print("=" * 60)
    print("【演示2：Few-shot函数的编译时分析】")
    print("=" * 60)
    print()

    compiler2 = SGLangCompiler()
    patterns2 = compiler2.analyze_function(demo_few_shot)
    summary2 = compiler2.get_summary()

    print(f"识别到 {summary2['total_patterns']} 个前缀模式：")
    print(
        f"  静态模式（示例部分）："
        f"{summary2['static_patterns']} 个"
    )
    print(
        f"  模板模式（查询输入）："
        f"{summary2['template_patterns']} 个"
    )
    print(
        f"  预估可缓存token数："
        f"{summary2['estimated_cacheable_tokens']}"
    )

    # 计算静态比例
    total_tokens = sum(p.estimated_tokens for p in patterns2)
    static_tokens = sum(
        p.estimated_tokens
        for p in patterns2
        if p.pattern_type == "static"
    )
    if total_tokens > 0:
        ratio = static_tokens / total_tokens
        print(f"  静态内容占比：{ratio:.1%}（缓存效率极高）")

    print()
    print("=" * 60)
    print("【演示3：多步骤函数的编译时分析】")
    print("=" * 60)
    print()

    compiler3 = SGLangCompiler()
    patterns3 = compiler3.analyze_function(demo_multi_step)
    summary3 = compiler3.get_summary()

    print(f"识别到 {summary3['total_patterns']} 个前缀模式：")
    print("\n详细模式：")
    for i, p in enumerate(patterns3, 1):
        type_label = (
            "【静态】" if p.pattern_type == "static"
            else f"【模板，变量={p.variables}】"
        )
        preview = p.content[:35].replace('\n', '\\n')
        print(f"  {i}. {type_label} {preview!r}")

    print()
    print("=" * 60)
    print("【编译时分析的工程价值】")
    print("=" * 60)
    print()
    print("零运行时开销：分析在函数定义时执行，")
    print("  运行时直接使用分析结果，无额外计算。")
    print()
    print("自动发现优化机会：开发者无需手动标注，")
    print("  编译器自动识别所有可缓存前缀。")
    print()
    print("两类可缓存模式：")
    print("  static  ：完全相同，跨所有请求直接共享KV Cache")
    print("  template：骨架相同，可参数化缓存骨架部分")


if __name__ == "__main__":
    main()
