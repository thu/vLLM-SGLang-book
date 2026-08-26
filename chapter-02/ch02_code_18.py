import numpy as np


def online_softmax_attention(Q, K, V, block_size):
    """
    Flash Attention的在线Softmax算法
    功能：通过分块计算和在线更新实现低内存开销的Attention
    """
    N, d = Q.shape
    num_blocks = (N + block_size - 1) // block_size

    # 初始化
    O = np.zeros_like(Q)
    m = -np.inf * np.ones(N)
    l = np.zeros(N)

    # 遍历Q的块
    for i in range(num_blocks):
        Qi = Q[i * block_size:(i + 1) * block_size]

        # 遍历K,V的块
        for j in range(num_blocks):
            Kj = K[j * block_size:(j + 1) * block_size]
            Vj = V[j * block_size:(j + 1) * block_size]

            # 计算注意力分数
            Sij = Qi @ Kj.T / np.sqrt(d)  # [block_size, block_size]

            # 更新最大值
            m_new = np.maximum(m[i * block_size:(i + 1) * block_size],
                               np.max(Sij, axis=1))

            # 更新指数和，需要修正之前的累积值
            correction = np.exp(m[i * block_size:(i + 1) * block_size] - m_new)
            l_new = correction * l[i * block_size:(i + 1) * block_size] + \
                    np.sum(np.exp(Sij - m_new[:, None]), axis=1)

            # 更新输出，也需要修正
            O[i * block_size:(i + 1) * block_size] = \
                np.diag(correction) @ O[i * block_size:(i + 1) * block_size] + \
                np.exp(Sij - m_new[:, None]) @ Vj

            # 更新统计量
            m[i * block_size:(i + 1) * block_size] = m_new
            l[i * block_size:(i + 1) * block_size] = l_new

    # 最终归一化
    O = np.diag(1.0 / l) @ O
    return O


# 使用示例
if __name__ == "__main__":
    # 构造测试数据
    N, d = 128, 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # 执行Flash Attention
    output = online_softmax_attention(Q, K, V, block_size=32)
    print(f"输出形状：{output.shape}")