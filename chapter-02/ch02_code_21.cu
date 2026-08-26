/* vllm/csrc/attention/attention_kernels.cu */

template<typename scalar_t, int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v2_kernel(
    scalar_t* __restrict__ out,           /* 输出 */
    const scalar_t* __restrict__ q,       /* Query */
    const scalar_t* __restrict__ k_cache, /* KV Cache */
    const scalar_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale
) {
    /* 每个block处理一个query token的一个head */
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    /* 线程协作：计算该head的attention */
    __shared__ float s_qk[HEAD_DIM];  /* Query向量，共享 */
    __shared__ float s_logits[MAX_SEQ_LEN];  /* 注意力分数 */

    /* 步骤1：加载Query到共享内存 */
    if (threadIdx.x < HEAD_DIM) {
        s_qk[threadIdx.x] = float(q[token_idx * num_heads * HEAD_DIM +
                                      head_idx * HEAD_DIM + threadIdx.x]);
    }
    __syncthreads();

    /* 步骤2：计算QK^T，使用在线softmax */
    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;

    const int context_len = context_lens[get_seq_idx(token_idx)];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* 遍历所有KV块 */
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_tables[...];

        for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
            const int pos = block_idx * BLOCK_SIZE + i;
            if (pos >= context_len) break;

            /* 加载Key */
            const scalar_t* k_ptr = k_cache + physical_block * ... + i * HEAD_DIM;

            /* 计算点积 */
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                qk += s_qk[d] * float(k_ptr[d]);
            }
            qk *= scale;

            /* 在线更新softmax统计量 */
            const float old_max = max_logit;
            max_logit = fmaxf(max_logit, qk);

            /* 修正之前的指数和 */
            const float correction = expf(old_max - max_logit);
            sum_exp = sum_exp * correction + expf(qk - max_logit);

            s_logits[pos] = qk;
        }
    }

    /* Block-level reduction获取全局最大值和指数和 */
    max_logit = blockReduceMax(max_logit);
    sum_exp = blockReduceSum(sum_exp * expf(max_logit - ...));
    __syncthreads();

    /* 步骤3：计算attention权重并加权Value */
    float acc[HEAD_DIM] = {0.0f};

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_tables[...];

        for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
            const int pos = block_idx * BLOCK_SIZE + i;
            if (pos >= context_len) break;

            /* 计算归一化的attention权重 */
            const float attn_weight = expf(s_logits[pos] - max_logit) / sum_exp;

            /* 加载Value并累加 */
            const scalar_t* v_ptr = v_cache + physical_block * ... + i * HEAD_DIM;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] += attn_weight * float(v_ptr[d]);
            }
        }
    }

    /* 步骤4：写回输出 */
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        out[token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d] =
            scalar_t(acc[d]);
    }
}