/* 融合LayerNorm和Linear投影的kernel */
__global__ void fused_layernorm_linear_kernel(
    const float* input,          /* [batch, seq_len, hidden_size] */
    const float* gamma,          /* LayerNorm参数 [hidden_size] */
    const float* beta,           /* LayerNorm参数 [hidden_size] */
    const float* weight,         /* Linear权重 [out_features, hidden_size] */
    float* output,               /* [batch, seq_len, out_features] */
    int batch_size,
    int seq_len,
    int hidden_size,
    int out_features
) {
    /* 每个block处理一个token */
    int token_idx = blockIdx.x;
    int batch_idx = token_idx / seq_len;
    int seq_idx = token_idx % seq_len;

    /* 共享内存：存储归一化后的中间结果 */
    extern __shared__ float s_normalized[];

    /* ============ 阶段1：LayerNorm，规约操作 ============ */

    /* 步骤1.1：计算均值，使用warp规约 */
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * seq_len * hidden_size +
                  seq_idx * hidden_size + i;
        thread_sum += input[idx];
    }

    /* Warp-level reduction */
    __shared__ float s_mean;
    thread_sum = warpReduceSum(thread_sum);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_mean, thread_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_mean /= hidden_size;
    }
    __syncthreads();

    /* 步骤1.2：计算方差 */
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * seq_len * hidden_size +
                  seq_idx * hidden_size + i;
        float diff = input[idx] - s_mean;
        thread_var += diff * diff;
    }

    __shared__ float s_var;
    thread_var = warpReduceSum(thread_var);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_var, thread_var);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_var /= hidden_size;
    }
    __syncthreads();

    /* 步骤1.3：归一化并存入共享内存 */
    float inv_std = rsqrtf(s_var + 1e-5f);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * seq_len * hidden_size +
                  seq_idx * hidden_size + i;
        float val = input[idx];
        s_normalized[i] = (val - s_mean) * inv_std * gamma[i] + beta[i];
    }
    __syncthreads();

    /* ============ 阶段2：Linear投影，矩阵乘法 ============ */

    /* 每个线程计算输出的一部分元素 */
    for (int out_idx = threadIdx.x; out_idx < out_features; out_idx += blockDim.x) {
        float sum = 0.0f;

        /* 点积：output[out_idx] = sum(normalized * weight[out_idx, :]) */
        for (int h = 0; h < hidden_size; h++) {
            sum += s_normalized[h] * weight[out_idx * hidden_size + h];
        }

        /* 写回全局内存 */
        int out_global_idx = batch_idx * seq_len * out_features +
                             seq_idx * out_features + out_idx;
        output[out_global_idx] = sum;
    }
}

/* Warp规约辅助函数 */
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}