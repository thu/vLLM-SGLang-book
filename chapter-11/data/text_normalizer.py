# data/text_normalizer.py
# 文书正文标准化流水线（ch11_code_07）
#
# 目的：消除因文档来源不同而产生的字节级差异，
# 确保"相同"文书在进入前缀缓存时能够生成完全一致的
# token 序列，从而实现最大化的缓存命中率。
#
# 常见的差异来源：
#   1. 换行符：Windows(\r\n) vs Linux(\n) vs Mac(\r)
#   2. Unicode 标准化形式：NFC / NFD / NFKC / NFKD 混用
#   3. 连续空白：多个连续空格或空行
#   4. 尾随空白：行末空格
#   5. BOM 头：UTF-8 BOM（\ufeff）偶见于 Windows 导出文件

import re
import unicodedata


def normalize_document_text(text: str) -> str:
    """
    对合同文书正文进行字节级标准化处理。

    流水线步骤（顺序固定，不可调换）：
        1. 去除 UTF-8 BOM 头
        2. Unicode NFC 标准化（统一字符的组合形式）
        3. 统一换行符为 \\n
        4. 去除每行行末尾随空白
        5. 压缩连续空行（超过两个空行压缩为两个）
        6. 压缩行内连续空格（超过一个空格压缩为一个）
        7. 去除首尾空白

    参数
    ----
    text : str
        原始文书文本。

    返回
    ----
    str
        标准化后的文本，保证相同来源文书产生相同字节序列。
    """
    # 步骤 1：去除 BOM
    text = text.lstrip("\ufeff")

    # 步骤 2：Unicode NFC 标准化
    # NFC 将分解形式（如 é = e + 组合重音符）还原为预合成形式
    text = unicodedata.normalize("NFC", text)

    # 步骤 3：统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 步骤 4：去除行末尾随空白
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 步骤 5：压缩连续空行（3 个及以上压缩为 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 步骤 6：压缩行内连续空格
    text = re.sub(r" {2,}", " ", text)

    # 步骤 7：去除首尾空白
    return text.strip()


def normalize_system_prompt(prompt: str) -> str:
    """
    对系统提示词进行轻量级标准化。

    系统提示词通常由代码字符串直接定义，
    产生差异的可能性低于外部导入的文书，
    但仍需消除潜在的尾随换行符和 BOM。
    """
    prompt = prompt.lstrip("\ufeff")
    prompt = unicodedata.normalize("NFC", prompt)
    prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
    return prompt.strip()
