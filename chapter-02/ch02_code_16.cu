/* 融合后的kernel */
__global__ void fused_gelu_bias_dropout_kernel(
    float* x,                    /* 输入/输出 */
    const float* bias,           /* 偏置 */
    float dropout_prob,          /* dropout概率 */
    uint64_t seed,               /* 随机种子 */
    int n                        /* 元素数量 */
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        /* 所有操作在寄存器中完成，无需往返显存 */
        float val = x[idx];

        /* 步骤1：GELU激活 */
        val = 0.5f * val * (1.0f + tanhf(0.797885f *
              (val + 0.044715f * val * val * val)));

        /* 步骤2：加偏置 */
        val += bias[idx];

        /* 步骤3：Dropout */
        curandState state;
        curand_init(seed, idx, 0, &state);
        if (curand_uniform(&state) < dropout_prob) {
            val = 0.0f;
        } else {
            val /= (1.0f - dropout_prob);
        }

        /* 一次性写回 */
        x[idx] = val;
    }
}

/* 融合版本调用 */
void fused_execution(float* x, const float* bias, float prob, uint64_t seed, int n) {
    dim3 grid((n + 255) / 256);
    dim3 block(256);

    fused_gelu_bias_dropout_kernel<<<grid, block>>>(x, bias, prob, seed, n);
}