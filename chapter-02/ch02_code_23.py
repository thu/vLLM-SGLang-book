# vllm/model_executor/layers/linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层
    功能：权重形状为[output_size_per_partition, input_size]，每个GPU只存储output_size/tp_size列
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            gather_output: bool = True,
            params_dtype: Optional[torch.dtype] = None,
            linear_method: Optional[LinearMethodBase] = None
    ):
        super().__init__()

        # 获取张量并行配置
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # 计算每个partition的大小
        assert output_size % self.tp_size == 0, \
            f"output_size({output_size}) must be divisible by tp_size({self.tp_size})"

        self.output_size_per_partition = output_size // self.tp_size
        self.gather_output = gather_output

        # 创建权重，只存储本GPU的部分
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
                dtype=params_dtype or torch.float16
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype or torch.float16
                ),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

        # 线性方法，用于量化等
        if linear_method is not None:
            self.linear_method = linear_method(self)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_: [batch_size, ..., input_size]
        Returns:
            output: [batch_size, ..., output_size] 如果gather_output=True
                   [batch_size, ..., output_size_per_partition] 否则
        """
        # 本地矩阵乘法
        if self.linear_method is not None:
            output_parallel = self.linear_method.apply(
                input_, self.weight, self.bias
            )
        else:
            output_parallel = F.linear(input_, self.weight, self.bias)

        if self.gather_output:
            # All-Gather：收集所有GPU的输出
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            # 保持分片状态，用于下一层的行并行输入
            output = output_parallel

        return output


def tensor_model_parallel_all_gather(input_: torch.Tensor) -> torch.Tensor:
    """
    张量并行All-Gather通信
    功能：将所有GPU的输入在最后一个维度拼接
    """
    world_size = get_tensor_model_parallel_world_size()

    # 单GPU，无需通信
    if world_size == 1:
        return input_

    # 获取并行组
    group = get_tensor_model_parallel_group()

    # 计算输出shape
    last_dim = input_.size(-1)
    output_shape = list(input_.size())
    output_shape[-1] = last_dim * world_size

    # 分配输出tensor
    output = torch.empty(
        output_shape,
        dtype=input_.dtype,
        device=input_.device
    )

    # 执行All-Gather
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=group
    )

    return output


# 使用示例
if __name__ == "__main__":
    # 初始化分布式环境
    torch.distributed.init_process_group(backend='nccl')

    # 创建列并行层
    layer = ColumnParallelLinear(
        input_size=4096,
        output_size=12288,
        bias=True,
        gather_output=True
    )

    # 前向传播
    input_tensor = torch.randn(32, 128, 4096).cuda()
    output = layer(input_tensor)
    print(f"输出形状：{output.shape}")  # [32, 128, 12288]