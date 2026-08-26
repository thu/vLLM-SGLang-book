import torch


def gptq_quantize(W, X, bits=4, group_size=128):
    """
    GPTQ量化算法
    功能：使用Hessian引导和误差补偿进行权重量化

    Args:
        W：权重矩阵 [d_out, d_in]
        X：校准数据 [N, d_in]
        bits：量化位宽
        group_size：分组大小

    Returns:
        W_q：量化后的权重
        scales：每组的缩放因子
        zeros：每组的零点
    """
    d_out, d_in = W.shape
    num_groups = d_in // group_size

    # 计算Hessian
    H = (X.T @ X) / X.shape[0]
    H_inv = torch.inverse(H + 1e-6 * torch.eye(d_in))

    W_q = torch.zeros_like(W)
    scales = torch.zeros(d_out, num_groups)
    zeros = torch.zeros(d_out, num_groups)

    # 按组量化
    for group_idx in range(num_groups):
        start_col = group_idx * group_size
        end_col = start_col + group_size

        W_group = W[:, start_col:end_col].clone()
        H_group = H[start_col:end_col, start_col:end_col]
        H_inv_group = H_inv[start_col:end_col, start_col:end_col]

        # 按Hessian对角线大小排序
        diag_order = torch.argsort(torch.diag(H_group), descending=True)

        # 逐列量化
        for col_idx in diag_order:
            # 计算量化参数
            w_col = W_group[:, col_idx]
            scale = (w_col.max() - w_col.min()) / (2 ** bits - 1)
            zero = w_col.min()

            # 量化
            w_q_col = torch.round((w_col - zero) / scale).clamp(0, 2 ** bits - 1)
            w_dequant = w_q_col * scale + zero

            # 计算误差
            error = w_col - w_dequant

            # 误差补偿到后续未量化的列
            remaining_cols = [c for c in diag_order if c > col_idx]
            for other_col in remaining_cols:
                compensation = H_inv_group[col_idx, other_col] / \
                               H_inv_group[other_col, other_col]
                W_group[:, other_col] -= compensation * error

            # 保存量化结果
            W_q[:, start_col + col_idx] = w_q_col
            if col_idx == 0:  # 只存储每组的第一个scale/zero
                scales[:, group_idx] = scale
                zeros[:, group_idx] = zero

    return W_q, scales, zeros


# 使用示例
if __name__ == "__main__":
    # 构造测试数据
    d_out, d_in = 4096, 4096
    W = torch.randn(d_out, d_in)
    X = torch.randn(128, d_in)  # 128个校准样本

    # 执行GPTQ量化
    W_q, scales, zeros = gptq_quantize(W, X, bits=4, group_size=128)

    print(f"原始权重形状：{W.shape}")
    print(f"量化权重形状：{W_q.shape}")
    print(f"缩放因子形状：{scales.shape}")
    print(f"零点形状：{zeros.shape}")

    # 计算量化误差
    W_dequant = W_q * scales[:, W_q.shape[1] // 128] + zeros[:, W_q.shape[1] // 128]
    error = (W - W_dequant).abs().mean()
    print(f"平均量化误差：{error.item():.6f}")