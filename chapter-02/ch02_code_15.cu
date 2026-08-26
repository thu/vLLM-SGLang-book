/* 未融合版本：三个独立kernel */

/* GELU激活函数kernel */
__global__ void gelu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = 0.5f * val * (1.0f + tanhf(0.797885f *
                 (val + 0.044715f * val * val * val)));
    }
}

/* 加偏置kernel */
__global__ void add_bias_kernel(float* x, const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += bias[idx];
    }
}

/* Dropout kernel */
__global__ void dropout_kernel(float* x, float prob, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        if (curand_uniform(&state) < prob) {
            x[idx] = 0.0f;
        } else {
            x[idx] /= (1.0f - prob);
        }
    }
}

/* 调用三次，三次内存往返 */
void unfused_execution(float* x, const float* bias, float prob, uint64_t seed, int n) {
    dim3 grid((n + 255) / 256);
    dim3 block(256);

    gelu_kernel<<<grid, block>>>(x, n);
    add_bias_kernel<<<grid, block>>>(x, bias, n);
    dropout_kernel<<<grid, block>>>(x, prob, seed, n);
}