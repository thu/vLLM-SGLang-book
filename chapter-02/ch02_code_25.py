class AsyncRowParallelLinear(RowParallelLinear):
    """
    异步行并行Linear
    功能：重叠通信与计算，隐藏All-Reduce延迟
    """

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """前向传播，异步通信"""
        # 切分输入，如果需要
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        # 本地计算
        output_parallel = F.linear(input_parallel, self.weight)

        # 异步启动All-Reduce
        handle = torch.distributed.all_reduce(
            output_parallel,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
            async_op=True  # 关键：异步操作
        )

        # 在通信进行时，可以执行其他操作
        # 例如：准备下一层的输入，加载权重等
        # ... 其他非依赖计算

        # 等待通信完成
        handle.wait()

        # 加偏置
        if self.bias is not None:
            output_parallel = output_parallel + self.bias

        return output_parallel


# 使用示例
if __name__ == "__main__":
    # 创建异步行并行层
    layer = AsyncRowParallelLinear(
        input_size=12288,
        output_size=4096,
        bias=True,
        input_is_parallel=True
    )

    # 前向传播
    input_tensor = torch.randn(32, 128, 12288 // tp_size).cuda()
    output = layer(input_tensor)
    print(f"输出形状：{output.shape}")