# vllm/model_executor/models/llama.py
import math


class LlamaAttention(nn.Module):
    """
    LLaMA的注意力层
    功能：支持张量并行的Multi-Head Attention实现
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # 获取张量并行配置
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = self.num_heads // tp_size

        # QKV投影：列并行，切分head维度
        self.qkv_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=3 * self.hidden_size,
            bias=False,
            gather_output=False,  # 不gather，保持分片
        )

        # 输出投影：行并行
        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,  # 输入已经是分片的
        )

        # Attention实现，PagedAttention
        self.attn = PagedAttention(
            self.num_heads_per_partition,
            self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim)
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """前向传播"""
        # QKV投影，列并行，每个GPU得到部分head
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention计算，每个GPU独立计算自己的head
        attn_output = self.attn(
            q, k, v,
            kv_cache,
            attn_metadata,
        )

        # 输出投影，行并行，自动All-Reduce
        output, _ = self.o_proj(attn_output)

        return output


class LlamaMLP(nn.Module):
    """
    LLaMA的FFN层
    功能：支持张量并行的Feed-Forward Network实现
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # 上升投影：列并行
        self.gate_up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=2 * config.intermediate_size,
            bias=False,
            gather_output=False,
        )

        # 下降投影：行并行
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.act_fn = SiLUAndMul()  # SwiGLU激活

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 上升投影，列并行
        gate_up, _ = self.gate_up_proj(hidden_states)

        # 激活，每个GPU独立
        intermediate = self.act_fn(gate_up)

        # 下降投影，行并行，自动All-Reduce
        output, _ = self.down_proj(intermediate)

        return output


# 使用示例
if __name__ == "__main__":
    from dataclasses import dataclass


    @dataclass
    class LlamaConfig:
        hidden_size: int = 4096
        num_attention_heads: int = 32
        intermediate_size: int = 11008


    config = LlamaConfig()

    # 创建Attention层
    attn = LlamaAttention(config)
    hidden_states = torch.randn(32, 128, 4096).cuda()
    output = attn(None, hidden_states, None, None)
    print(f"Attention输出形状：{output.shape}")

    # 创建MLP层
    mlp = LlamaMLP(config)
    output = mlp(hidden_states)
    print(f"MLP输出形状：{output.shape}")