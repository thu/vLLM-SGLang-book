import torch


def identify_sensitive_weights(W, X, ratio=0.01):
    """
    识别敏感权重
    功能：使用二阶敏感度选择需要保护的权重

    Args:
        W：权重矩阵 [d_out, d_in]
        X：校准数据 [N, d_in]
        ratio：保留为FP16的权重比例

    Returns:
        mask:bool矩阵，True表示敏感权重
    """
    # 计算Hessian对角线，近似
    H_diag = torch.mean(X ** 2, dim=0)  # [d_in]

    # 计算敏感度
    sensitivity = torch.abs(W) * torch.sqrt(H_diag[None, :])

    # 选择top-k最敏感的权重
    k = int(W.numel() * ratio)
    threshold = torch.topk(sensitivity.flatten(), k).values[-1]

    mask = sensitivity >= threshold

    return mask


# 使用示例
if __name__ == "__main__":
    # 构造测试数据
    W = torch.randn(4096, 4096)
    X = torch.randn(128, 4096)

    # 识别敏感权重
    sensitive_mask = identify_sensitive_weights(W, X, ratio=0.01)

    print(f"权重总数：{W.numel()}")
    print(f"敏感权重数：{sensitive_mask.sum().item()}")
    print(f"敏感权重比例：{sensitive_mask.float().mean().item():.2%}")