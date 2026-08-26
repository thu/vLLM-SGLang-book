def select_fusion_strategy(phase, seq_len):
    """
    根据阶段和序列长度选择融合策略
    功能：动态选择最优的算子融合配置
    """
    if phase == "prefill":
        # Prefill计算密集，融合收益大
        use_flash_attention = True
        use_fused_layernorm = True
    else:  # decode
        # Decode内存密集，融合收益小
        use_flash_attention = (seq_len > 512)  # 短序列收益有限
        use_fused_layernorm = False

    return {
        'flash_attention': use_flash_attention,
        'fused_layernorm': use_fused_layernorm
    }


# 块大小调优
def select_block_size(model_size):
    """
    根据模型大小选择块大小
    功能：PagedAttention的块大小影响融合效率
    """
    if model_size < "13B":
        BLOCK_SIZE = 16  # 小模型，降低内存碎片
    else:
        BLOCK_SIZE = 32  # 大模型，减少块表大小

    return BLOCK_SIZE


# 使用示例
if __name__ == "__main__":
    # Prefill阶段
    config = select_fusion_strategy("prefill", seq_len=1024)
    print(f"Prefill配置：{config}")

    # Decode阶段
    config = select_fusion_strategy("decode", seq_len=256)
    print(f"Decode配置：{config}")

    # 块大小选择
    block_size = select_block_size("70B")
    print(f"块大小：{block_size}")