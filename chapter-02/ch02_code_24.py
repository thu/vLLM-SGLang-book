class RowParallelLinear(nn.Module):
    """
    行并行线性层
    功能：权重形状为[output_size, input_size_per_partition]，每个GPU只存储input_size/tp_size行
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            input_is_parallel: bool = False,
            params_dtype: Optional[torch.dtype] = None,
            reduce_results: bool = True,
            linear_method: Optional[LinearMethodBase] = None
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # 计算partition大小
        assert input_size % self.tp_size == 0
        self.input_size_per_partition = input_size // self.tp_size

        # 创建权重
        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition,
                dtype=params_dtype or torch.float16
            ),
            requires_grad=False
        )

        # 注意：bias只在rank 0创建，避免重复
        if bias and self.tp_rank == 0:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=params_dtype or torch.float16),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_:
                如果input_is_parallel=True:[batch, ..., input_size_per_partition]
                否则：[batch, ..., input_size]
        """
        # 如果输入未切分，先进行切分
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        # 本地矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight)

        if self.reduce_results:
            # All-Reduce求和
            output = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        # 加偏置，只在rank 0有bias
        if self.bias is not None:
            output = output + self.bias

        return output


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """
    张量并行All-Reduce通信
    功能：对所有GPU的输入求和
    """
    world_size = get_tensor_model_parallel_world_size()

    if world_size == 1:
        return input_

    # 执行All-Reduce求和
    torch.distributed.all_reduce(
        input_,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )

    return input_


# 使用示例
if __name__ == "__main__":
    # 创建行并行层
    layer = RowParallelLinear(
        input_size=12288,
        output_size=4096,
        bias=True,
        input_is_parallel=True,
        reduce_results=True
    )

    # 前向传播，输入已经是分片的
    input_tensor = torch.randn(32, 128, 12288 // tp_size).cuda()
    output = layer(input_tensor)
    print(f"输出形状：{output.shape}")  # [32, 128, 4096]