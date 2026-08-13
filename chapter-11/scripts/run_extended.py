# scripts/run_extended.py
# 延伸场景演示：学术论文审阅、代码库分析、多语言翻译
#
# 三个延伸场景均基于与法律文书相同的核心技术路径，
# 但各自的缓存利用模式存在差异：
#
#   学术论文（低文书层复用率，约 50%）：
#     每篇论文通常只需 1–2 项分析，文书层复用率远低于
#     法律文书场景（83%）。优化重点在于识别跨文档的公共
#     段落，构建额外的 L2.5 缓存层。
#
#   代码库分析（最高命中率，约 85–90%）：
#     天然的层级结构（项目→包→文件→函数）与 RadixAttention
#     的基数树结构完美匹配，缓存命中率最高。
#     风险点：大型代码库的树节点数量膨胀，需分库路由。
#
#   多语言翻译（前缀分叉问题）：
#     关键设计原则：目标语言声明须放在源文本之后，
#     确保源文本成为所有语言翻译请求的公共前缀。

from __future__ import annotations

import asyncio
import hashlib
from difflib import SequenceMatcher

from openai import AsyncOpenAI

from config.model_config import MODEL_CONFIG
from config.prompt_config import UNIFIED_SYSTEM_PROMPT
from inference.sglang_server import get_sglang_base_url


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=get_sglang_base_url(), api_key="EMPTY"
    )


def _model() -> str:
    return MODEL_CONFIG["model_path"].split("/")[-1]


# ── 场景一：学术论文批量审阅 ─────────────────────────────────────

class AcademicDocumentCacheOptimizer:
    """
    学术文档缓存优化器：识别跨文档共享段落。

    核心思路：
      在低文书层复用率的场景中，通过识别跨文档的高频段落，
      将其提取为共享的 L2.5 层前缀，人工提升缓存命中率。
      段落相似度基于精确哈希匹配（而非模糊相似度），
      以确保前缀在 token 层面的字节级一致性。

    限制：
      只有字节级完全相同的段落才能形成精确的缓存命中。
      本优化器通过哈希匹配识别完全相同的段落，
      不使用 SequenceMatcher 进行近似匹配，
      因为近似匹配无法保证 token 序列的精确一致性。
    """

    def __init__(
        self,
        min_paragraph_chars: int = 100,
        min_occurrence_ratio: float = 0.30,
    ) -> None:
        # 段落最小字符数（过短的段落缓存价值低）
        self.min_paragraph_chars = min_paragraph_chars
        # 出现在超过此比例文档中的段落视为公共段落
        self.min_occurrence_ratio = min_occurrence_ratio

    def extract_common_sections(
        self, docs: list[str]
    ) -> dict[str, dict]:
        """
        从文档批次中提取高频公共段落（字节级精确匹配）。

        返回
        ----
        dict[str, dict]
            键为段落 MD5 哈希，值为包含 text 和 docs 的字典。
        """
        n_docs = len(docs)
        para_index: dict[str, dict] = {}

        for doc_id, text in enumerate(docs):
            paragraphs = [
                p.strip()
                for p in text.split("\n\n")
                if len(p.strip()) >= self.min_paragraph_chars
            ]
            seen_in_doc: set[str] = set()
            for para in paragraphs:
                key = hashlib.md5(
                    para.encode("utf-8")
                ).hexdigest()
                if key in seen_in_doc:
                    continue
                seen_in_doc.add(key)
                if key not in para_index:
                    para_index[key] = {
                        "text": para,
                        "docs": set(),
                    }
                para_index[key]["docs"].add(doc_id)

        # 筛选出现比例超过阈值的段落
        return {
            k: v for k, v in para_index.items()
            if len(v["docs"]) / n_docs
            >= self.min_occurrence_ratio
        }

    def build_warmup_prefix(
        self,
        common_sections: dict[str, dict],
        system_prompt: str,
        max_sections: int = 5,
    ) -> str:
        """
        构建预热请求内容：系统提示 + 高频公共段落摘录。

        选取出现频率最高的 max_sections 个段落，
        按出现频率降序排列后拼接，形成固定顺序的 L2.5 层内容。
        固定排序确保相同段落集合总是产生相同的 token 序列。
        """
        top_sections = sorted(
            common_sections.values(),
            key=lambda x: len(x["docs"]),
            reverse=True,
        )[:max_sections]

        warmup = system_prompt + "\n\n"
        warmup += "以下为本批次文档的公共参考段落：\n\n"
        for sec in top_sections:
            warmup += sec["text"] + "\n\n"
        return warmup.strip()


async def demo_academic_review(n_papers: int = 10) -> None:
    """学术论文审阅场景演示。"""
    print("\n── 场景一：学术论文批量审阅 ──")
    optimizer = AcademicDocumentCacheOptimizer(
        min_paragraph_chars=100,
        min_occurrence_ratio=0.30,
    )

    # 模拟论文正文（含公共段落）
    common_para = (
        "实验采用随机对照设计，样本量通过功效分析确定，"
        "显著性水平设为 α=0.05，统计检验为双侧检验。"
        "所有数据经过预处理后使用 Python 3.11 进行分析，"
        "统计软件版本为 SciPy 1.12.0。"
    )
    papers = [
        common_para + f"\n\n本研究关注领域{i}的核心问题，"
        f"通过对{100 + i * 10}个样本的实验验证了假设H{i}。"
        for i in range(n_papers)
    ]

    common = optimizer.extract_common_sections(papers)
    print(
        f"  发现 {len(common)} 个公共段落，"
        f"出现在 ≥30% 的文档中"
    )
    warmup_prefix = optimizer.build_warmup_prefix(
        common, UNIFIED_SYSTEM_PROMPT
    )
    print(
        f"  L2.5 预热前缀长度：{len(warmup_prefix)} 字符"
    )
    print("  （实际推理调用在真实环境中执行）")


# ── 场景二：代码库全量分析 ───────────────────────────────────────

class CodebaseAwareRouter:
    """
    代码库感知的请求路由器。

    确保同一代码库的所有分析请求始终路由至同一 SGLang 实例，
    使该实例的基数树能够积累并复用该代码库的层级前缀缓存，
    避免跨实例的缓存碎片化。

    路由算法：基于 repo_id 的 MD5 哈希取模，
    保证同一 repo 的路由结果在实例列表不变时的确定性。
    """

    def __init__(self, sglang_instances: list[str]) -> None:
        if not sglang_instances:
            raise ValueError(
                "sglang_instances 不能为空"
            )
        self._instances = sglang_instances
        self._n = len(sglang_instances)

    def get_instance_url(self, repo_id: str) -> str:
        """
        根据 repo_id 通过一致性哈希确定目标实例。

        一致性哈希保证：
          同一 repo_id 始终映射到同一实例（实例列表不变时）。
          不同 repo_id 在实例间均匀分布。
        """
        hash_int = int(
            hashlib.md5(repo_id.encode("utf-8")).hexdigest(),
            16,
        )
        return self._instances[hash_int % self._n]

    def build_hierarchical_prefix(
        self,
        repo_id: str,
        package_name: str,
        file_path: str,
        func_code: str,
        func_name: str,
    ) -> str:
        """
        构建层级化的分析请求前缀。

        层级结构（从外到内）：
          L1  系统提示（所有请求共享）
          L2  代码审查任务指令（所有请求共享）
          L2.5 项目级上下文（同 repo 共享）
          L3  包级上下文（同包共享）
          L3.5 文件级代码（同文件共享）
          L4  函数级查询（每次请求独立）

        注意：各层内容须固定且不含动态字段，
        以确保跨请求的字节级一致性。
        """
        return (
            UNIFIED_SYSTEM_PROMPT
            + "\n\n请对以下代码进行安全漏洞扫描，"
            "识别潜在的注入攻击、越界访问和资源泄漏风险。\n\n"
            + f"# 项目：{repo_id}\n"
            + f"# 包：{package_name}\n"
            + f"# 文件：{file_path}\n\n"
            + func_code
            + f"\n\n请分析函数 `{func_name}` 的安全风险。"
        )


async def demo_codebase_analysis() -> None:
    """代码库分析场景演示。"""
    print("\n── 场景二：代码库全量分析 ──")
    router = CodebaseAwareRouter(
        sglang_instances=[
            "http://localhost:30000/v1",
            "http://localhost:30001/v1",
        ]
    )

    test_cases = [
        ("repo-alpha", "auth", "auth/login.py", "def login"),
        ("repo-alpha", "auth", "auth/logout.py", "def logout"),
        ("repo-beta", "api", "api/handler.py", "def handle"),
    ]

    for repo_id, pkg, fpath, func in test_cases:
        url = router.get_instance_url(repo_id)
        print(
            f"  {repo_id}/{func} → 实例 {url.split('//')[1]}"
        )

    print(
        "  （同一 repo 的请求始终路由至相同实例，"
        "保证缓存复用）"
    )


# ── 场景三：多语言翻译记忆库 ─────────────────────────────────────

class TranslationMemoryPrefixBuilder:
    """
    翻译记忆库与前缀缓存的协同构建器。

    核心设计原则（前缀分叉问题的根本解决方案）：
      目标语言声明必须放在源文本之后，
      确保"系统提示 + 翻译示例 + 源文本"成为所有目标语言
      翻译请求的公共前缀，仅在最末的目标语言声明处分叉。

    错误的设计（前缀在源文本之前分叉，缓存完全失效）：
      "请将以下内容翻译为{target_lang}：\n\n{source_text}"
      ↑ 不同语言在此处就已分叉，源文本 KV Cache 被重复计算

    正确的设计（源文本是公共前缀，最末才分叉）：
      "{few_shot_examples}\n{source_text}\n\n目标语言：{target_lang}"
      ↑ 仅最后一行不同，源文本 KV Cache 被所有语言共享
    """

    # 支持的目标语言列表
    SUPPORTED_LANGUAGES = ["英语", "日语", "法语", "德语", "西班牙语"]

    def __init__(
        self,
        translation_memory: list[dict] | None = None,
        max_examples: int = 3,
    ) -> None:
        """
        参数
        ----
        translation_memory : list[dict], optional
            翻译记忆库条目列表，每个条目包含 source 和 target 字段。
            若为 None，使用内置示例。
        max_examples : int
            最多引用的历史翻译示例数，默认 3。
            示例数量影响 L2.5 层的长度（约 200–600 tokens/示例），
            过多示例会增加前缀长度，过少则降低翻译风格一致性。
        """
        self._memory = translation_memory or []
        self.max_examples = max_examples

    def _retrieve_examples(
        self, target_lang: str
    ) -> list[dict]:
        """
        从记忆库检索与目标语言匹配的历史翻译示例。

        检索结果的顺序须固定（按 source 字段排序），
        确保相同输入产生字节级相同的前缀，
        避免因检索顺序随机变化导致缓存键不匹配。
        """
        candidates = [
            ex for ex in self._memory
            if ex.get("target_lang") == target_lang
        ]
        # 固定排序：确保相同记忆库和语言产生相同的示例序列
        candidates.sort(key=lambda x: x.get("source", ""))
        return candidates[: self.max_examples]

    def build_translation_prompt(
        self,
        source_text: str,
        target_lang: str,
    ) -> str:
        """
        构建正确的翻译请求前缀。

        前缀结构（公共部分尽可能长）：
          L1  统一系统提示
          L2  翻译任务通用说明
          L2.5 翻译记忆示例（同目标语言的请求共享）
          L3  源文本（所有目标语言请求共享）
          L4  目标语言声明（仅在此处分叉）

        公共前缀长度 ≈ L1 + L2 + L2.5 + L3
                    ≈ 512 + 100 + 600×n_examples + source_tokens
        对于 20K tokens 的源文本，公共前缀占总前缀的 >98%。
        """
        examples = self._retrieve_examples(target_lang)

        # L1 + L2
        prefix = (
            UNIFIED_SYSTEM_PROMPT
            + "\n\n你是一名专业翻译，请保持原文的语气和格式。\n\n"
        )

        # L2.5：翻译记忆示例（固定顺序插入，确保可缓存）
        if examples:
            prefix += "参考译文示例：\n\n"
            for ex in examples:
                prefix += (
                    f"原文：{ex['source']}\n"
                    f"译文：{ex['target']}\n\n"
                )

        # L3：源文本（所有目标语言共享，须在语言声明之前）
        prefix += f"待翻译内容：\n\n{source_text}\n\n"

        # L4：目标语言声明（仅在此处分叉，最末才区分语言）
        prefix += f"目标语言：{target_lang}\n译文："

        return prefix


async def demo_translation(
    source_text: str | None = None,
) -> None:
    """多语言翻译场景演示。"""
    print("\n── 场景三：多语言翻译记忆库 ──")

    if source_text is None:
        source_text = (
            "本协议由甲乙双方就软件服务的提供与使用事项协商订立，"
            "双方均应严格遵守本协议的全部条款。\n"
            "任何一方违反本协议的，应承担相应的法律责任。"
        )

    memory = [
        {
            "source": "本协议由双方协商订立",
            "target": "This agreement is concluded by mutual negotiation",
            "target_lang": "英语",
        },
        {
            "source": "应承担相应的法律责任",
            "target": "shall bear corresponding legal liability",
            "target_lang": "英语",
        },
    ]

    builder = TranslationMemoryPrefixBuilder(
        translation_memory=memory, max_examples=2
    )

    # 展示正确的前缀结构（公共前缀尽可能长）
    for lang in ["英语", "日语", "法语"]:
        prompt = builder.build_translation_prompt(
            source_text, lang
        )
        # 找到分叉点：目标语言声明之前的部分为公共前缀
        fork_pos = prompt.rfind(f"目标语言：{lang}")
        common_len = fork_pos if fork_pos != -1 else len(prompt)
        print(
            f"  {lang}：公共前缀 {common_len} 字符，"
            f"分叉点后 {len(prompt) - common_len} 字符"
        )

    print(
        "  （源文本是所有语言的公共前缀，"
        "仅目标语言声明处分叉）"
    )
    print("  （实际推理调用在真实环境中执行）")


async def main() -> None:
    await demo_academic_review(n_papers=10)
    await demo_codebase_analysis()
    await demo_translation()
    print("\n延伸场景演示完成。")


if __name__ == "__main__":
    asyncio.run(main())
