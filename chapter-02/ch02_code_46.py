def kmeans_quantize(W, n_clusters=16, sensitive_mask=None):
    """
    k-means非均匀量化
    功能：使用聚类实现更灵活的量化

    Args:
        W：权重矩阵
        n_clusters：聚类中心数，对应bit数：16→4bit
        sensitive_mask：敏感权重掩码
    """
    # 步骤1：分离敏感权重
    if sensitive_mask is not None:
        W_dense = W.clone()
        W_dense[sensitive_mask] = 0  # 敏感权重置0
        W_sparse = W.clone()
        W_sparse[~sensitive_mask] = 0
    else:
        W_dense = W
        W_sparse = None

    # 步骤2：对稠密权重进行k-means聚类
    from sklearn.cluster import KMeans

    W_flat = W_dense.flatten().cpu().numpy()
    W_nonzero = W_flat[W_flat != 0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(W_nonzero.reshape(-1, 1))

    # 步骤3：量化，用最近的聚类中心替换
    labels = kmeans.predict(W_nonzero.reshape(-1, 1))
    W_quantized = kmeans.cluster_centers_[labels].flatten()

    # 步骤4：恢复形状
    W_q = torch.zeros_like(W_flat)
    W_q[W_flat != 0] = torch.from_numpy(W_quantized).float()
    W_q = W_q.reshape(W.shape)

    # 步骤5：添加回稀疏部分
    if W_sparse is not None:
        W_final = W_q + W_sparse
    else:
        W_final = W_q

    return W_final, kmeans.cluster_centers_, labels


# 使用示例
if __name__ == "__main__":
    # 构造测试数据
    W = torch.randn(1024, 1024)
    X = torch.randn(128, 1024)

    # 识别敏感权重
    sensitive_mask = identify_sensitive_weights(W, X, ratio=0.01)

    # k-means量化
    W_quantized, centers, labels = kmeans_quantize(
        W, n_clusters=16, sensitive_mask=sensitive_mask
    )

    print(f"聚类中心数：{len(centers)}")
    print(f"量化权重形状：{W_quantized.shape}")

    # 计算压缩率
    original_size = W.numel() * 2  # FP16
    quantized_size = len(centers) * 2 + len(labels) * 0.5  # 4-bit索引
    compression_ratio = original_size / quantized_size
    print(f"压缩率：{compression_ratio:.2f}x")