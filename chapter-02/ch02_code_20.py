# vllm/attention/backends/flash_attn.py
import torch
import math


class FlashAttentionBackend:
    """
    Flash Attention后端实现
    功能：根据序列长度选择最优的Attention实现
    """

    @staticmethod
    def forward(
            query: torch.Tensor,  # [num_tokens, num_heads, head_dim]
            key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_dim]
            value_cache: torch.Tensor,
            block_tables: torch.Tensor,
            context_lens: torch.Tensor,
            block_size: int,
            max_context_len: int
    ) -> torch.Tensor:
        """Flash Attention前向传播"""

        # 动态选择实现
        if max_context_len <= 512:
            # 短序列：使用标准实现，更简单，启动快
            return flash_attn_varlen_func(
                query, key_cache, value_cache,
                cu_seqlens_q=...,
                cu_seqlens_k=...,
                max_seqlen_q=...,
                max_seqlen_k=...,
                softmax_scale=1.0 / math.sqrt(head_dim)
            )
        else:
            # 长序列：使用分块Flash Attention
            return flash_attn_with_kvcache(
                query, key_cache, value_cache,
                cache_seqlens=context_lens,
                block_table=block_tables,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=True
            )


# 使用示例
if __name__ == "__main__":
    backend = FlashAttentionBackend()
    # 构造测试输入
    query = torch.randn(128, 32, 64).cuda()
    key_cache = torch.randn(100, 16, 32, 64).cuda()
    value_cache = torch.randn(100, 16, 32, 64).cuda()
    block_tables = torch.randint(0, 100, (8, 10)).cuda()
    context_lens = torch.randint(50, 150, (8,)).cuda()

    # 执行Attention
    output = backend.forward(query, key_cache, value_cache,
                             block_tables, context_lens, 16, 1024)
    print(f"输出形状：{output.shape}")