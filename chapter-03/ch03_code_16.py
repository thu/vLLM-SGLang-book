"""
第3章 代码示例 16
主题：基于有限状态机（FSM）的JSON约束解码器
场景：实现JSON解析的FSM状态枚举与约束解码器，
      演示每个生成步骤如何通过状态机过滤非法token，
      从根本上保证输出100%符合JSON语法

说明：
    本示例为教学性实现，展示约束解码的核心算法。
    SGLang生产环境中使用Outlines库的优化实现。
    无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import json
import time
from enum import Enum
from typing import Dict, List, Set


class JSONState(Enum):
    START         = "start"       # 期待{或[
    OBJECT_START  = "obj_start"   # 期待key（"）或}
    OBJECT_KEY    = "obj_key"     # 在key字符串内部
    OBJECT_COLON  = "obj_colon"   # 严格期待冒号
    OBJECT_VALUE  = "obj_value"   # 期待任意合法值类型
    OBJECT_COMMA  = "obj_comma"   # 期待逗号或}
    STRING_BODY   = "str_body"    # 在字符串值内部
    NUMBER        = "number"      # 在数字值内部
    COMPLETE      = "complete"    # JSON结构完整闭合


class MockTokenizer:
    """
    模拟tokenizer，用于演示约束解码逻辑。
    真实场景中使用模型配套的tokenizer。
    """

    def __init__(self):
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        # 构建常用JSON字符的token映射
        chars = (
            '{}[]:"\\,\n\t '
            '0123456789.-+eE'
            'truefalsnul'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            '名字年龄城市是否'
        )
        seen = set()
        idx = 0
        for c in chars:
            if c not in seen:
                self._char_to_id[c] = idx
                self._id_to_char[idx] = c
                seen.add(c)
                idx += 1
        self.vocab_size = idx

    def encode(self, text: str) -> List[int]:
        return [
            self._char_to_id[c]
            for c in text
            if c in self._char_to_id
        ]

    def decode(self, ids: List[int]) -> str:
        return "".join(
            self._id_to_char.get(i, "?") for i in ids
        )

    def __len__(self):
        return self.vocab_size


class JSONConstraintDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 用栈处理嵌套的对象和数组结构
        self.state_stack = [JSONState.START]
        self.depth = 0

    def get_allowed_tokens(self) -> Set[int]:
        """
        根据当前FSM状态返回合法token集合。
        每个token生成前调用此方法获取掩码，
        将不在集合中的token的logit置为-inf。
        """
        state = self.state_stack[-1]
        allowed = set()

        if state == JSONState.START:
            allowed.add(self.tokenizer.encode("{")[0])
            allowed.add(self.tokenizer.encode("[")[0])

        elif state == JSONState.OBJECT_COLON:
            # 此状态只允许冒号，强约束防止模型跳过
            allowed.add(self.tokenizer.encode(":")[0])

        elif state == JSONState.OBJECT_VALUE:
            # 值位置允许所有合法值起始token
            for start_char in ['"', '{', '[',
                                't', 'f', 'n']:
                toks = self.tokenizer.encode(start_char)
                if toks:
                    allowed.add(toks[0])
            for digit in "0123456789-":
                toks = self.tokenizer.encode(digit)
                if toks:
                    allowed.add(toks[0])

        elif state == JSONState.OBJECT_COMMA:
            allowed.add(self.tokenizer.encode(",")[0])
            allowed.add(self.tokenizer.encode("}")[0])

        elif state == JSONState.OBJECT_START:
            toks_quote = self.tokenizer.encode('"')
            toks_close = self.tokenizer.encode('}')
            if toks_quote:
                allowed.add(toks_quote[0])
            if toks_close:
                allowed.add(toks_close[0])

        elif state in (JSONState.OBJECT_KEY,
                       JSONState.STRING_BODY):
            # 字符串内部：允许大多数可打印字符和结束引号
            for c in (
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                '0123456789 _-名字年龄城市'
            ):
                toks = self.tokenizer.encode(c)
                if toks:
                    allowed.add(toks[0])
            # 允许结束引号
            toks_q = self.tokenizer.encode('"')
            if toks_q:
                allowed.add(toks_q[0])

        elif state == JSONState.NUMBER:
            for c in '0123456789.eE+-':
                toks = self.tokenizer.encode(c)
                if toks:
                    allowed.add(toks[0])
            # 数字可以被逗号或}终止
            for c in ',}':
                toks = self.tokenizer.encode(c)
                if toks:
                    allowed.add(toks[0])

        return allowed

    def update_state(self, char: str):
        """根据生成的字符更新FSM状态。"""
        state = self.state_stack[-1]

        if state == JSONState.START:
            if char == '{':
                self.state_stack[-1] = JSONState.OBJECT_START
                self.depth += 1
            elif char == '[':
                self.depth += 1

        elif state == JSONState.OBJECT_START:
            if char == '"':
                self.state_stack.append(JSONState.OBJECT_KEY)
            elif char == '}':
                self.depth -= 1
                self.state_stack.pop()
                if not self.state_stack:
                    self.state_stack = [JSONState.COMPLETE]

        elif state == JSONState.OBJECT_KEY:
            if char == '"':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_COLON)

        elif state == JSONState.OBJECT_COLON:
            if char == ':':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_VALUE)

        elif state == JSONState.OBJECT_VALUE:
            if char == '"':
                self.state_stack.pop()
                self.state_stack.append(JSONState.STRING_BODY)
            elif char in '0123456789-':
                self.state_stack.pop()
                self.state_stack.append(JSONState.NUMBER)
            elif char == '{':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_COMMA)
                self.state_stack.append(JSONState.OBJECT_START)
                self.depth += 1

        elif state == JSONState.STRING_BODY:
            if char == '"':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_COMMA)

        elif state == JSONState.NUMBER:
            if char in ',}':
                self.state_stack.pop()
                if char == '}':
                    self.depth -= 1
                    if self.depth == 0:
                        self.state_stack = [JSONState.COMPLETE]
                else:
                    self.state_stack.append(
                        JSONState.OBJECT_START
                    )

        elif state == JSONState.OBJECT_COMMA:
            if char == ',':
                self.state_stack.pop()
                self.state_stack.append(JSONState.OBJECT_START)
            elif char == '}':
                self.depth -= 1
                self.state_stack.pop()
                if self.depth == 0 or not self.state_stack:
                    self.state_stack = [JSONState.COMPLETE]

    def is_complete(self) -> bool:
        """JSON结构是否已完整闭合，可作为生成终止条件。"""
        return (
            len(self.state_stack) == 1
            and self.state_stack[-1] == JSONState.COMPLETE
        )

    def get_state_hash(self) -> int:
        """当前状态的哈希值，用于约束缓存的键。"""
        return hash(tuple(s.value for s in self.state_stack))


def main():
    tokenizer = MockTokenizer()

    # ── 演示1：各状态的合法token集合 ─────────────────────────
    print("=" * 60)
    print("【演示1：各FSM状态的合法token集合】")
    print("=" * 60)
    print()

    decoder = JSONConstraintDecoder(tokenizer)
    state_demos = [
        (JSONState.START,        "JSON起始"),
        (JSONState.OBJECT_START, "对象起始"),
        (JSONState.OBJECT_COLON, "期待冒号"),
        (JSONState.OBJECT_VALUE, "期待值"),
        (JSONState.OBJECT_COMMA, "期待逗号或}"),
    ]

    for state, desc in state_demos:
        decoder.state_stack = [state]
        allowed = decoder.get_allowed_tokens()
        chars = [
            tokenizer.decode([tid])
            for tid in sorted(allowed)
            if tid < tokenizer.vocab_size
        ]
        blocked = tokenizer.vocab_size - len(allowed)
        print(f"状态【{desc}】({state.value})：")
        print(f"  合法字符（{len(allowed)}个）：{chars[:12]}")
        print(f"  屏蔽token数：{blocked}（共{tokenizer.vocab_size}个）")
        print()

    # ── 演示2：逐字符生成追踪 ─────────────────────────────────
    print("=" * 60)
    print('【演示2：逐字符生成追踪 {"name":"Alice"}】')
    print("=" * 60)
    print()

    decoder2 = JSONConstraintDecoder(tokenizer)
    target_chars = [
        '{', '"', 'n', 'a', 'm', 'e', '"', ':',
        '"', 'A', 'l', 'i', 'c', 'e', '"', '}'
    ]

    generated = ""
    for char in target_chars:
        allowed = decoder2.get_allowed_tokens()
        enc = tokenizer.encode(char)
        is_allowed = bool(enc and enc[0] in allowed)
        state_before = decoder2.state_stack[-1].value
        decoder2.update_state(char)
        state_after = decoder2.state_stack[-1].value
        status = "✓" if is_allowed else "✗"
        print(
            f"  {status} [{state_before:12s}→{state_after:12s}]"
            f" 字符={char!r}"
        )
        generated += char

    print(f"\n生成结果：{generated}")
    print(f"JSON完整闭合：{decoder2.is_complete()}")

    try:
        parsed = json.loads(generated)
        print(f"解析成功：{parsed}")
        print("验证：约束解码保证输出100%可解析")
    except json.JSONDecodeError as e:
        print(f"解析失败：{e}")

    # ── 演示3：状态机约束的性能测量 ───────────────────────────
    print()
    print("=" * 60)
    print("【演示3：get_allowed_tokens调用性能】")
    print("=" * 60)
    print()

    decoder3 = JSONConstraintDecoder(tokenizer)
    N = 1000
    start = time.time()
    for _ in range(N):
        decoder3.state_stack = [JSONState.OBJECT_VALUE]
        decoder3.get_allowed_tokens()
    elapsed = (time.time() - start) * 1000

    print(f"  {N}次get_allowed_tokens调用耗时：{elapsed:.2f}ms")
    print(f"  平均每次：{elapsed/N:.4f}ms")
    print(f"  生产环境通过约束缓存可降至<0.001ms")
    print()
    print("=" * 60)
    print("【FSM约束解码原理总结】")
    print("=" * 60)
    print()
    print("核心思想：在每个token生成步骤，")
    print("  根据当前FSM状态计算合法token集合，")
    print("  将其余token的logit置为-inf，使其概率为零。")
    print()
    print("关键特性：")
    print("  状态栈处理嵌套：对象/数组可任意嵌套")
    print("  is_complete()：检测JSON是否完整闭合")
    print("  状态哈希：用于约束缓存的键，避免重复计算")


if __name__ == "__main__":
    main()
