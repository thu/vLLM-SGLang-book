# vllm/worker/worker.py
from typing import List
import torch
import torch.nn as nn


class PipelineParallelWorker:
    """
    流水线并行Worker
    功能：管理流水线阶段的模型分片和P2P通信
    """

    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
    ):
        self.model_config = model_config
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = parallel_config.pipeline_parallel_rank

        # 计算本stage负责的层
        total_layers = model_config.num_hidden_layers
        layers_per_stage = total_layers // self.pp_size

        self.start_layer = self.pp_rank * layers_per_stage
        self.end_layer = (self.pp_rank + 1) * layers_per_stage

        # 加载模型的部分层
        self.model = self._load_model_shard()

        # P2P通信
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

    def _load_model_shard(self) -> nn.Module:
        """
        只加载本stage的层
        功能：从完整模型中提取本stage负责的层
        """
        # 伪代码
        full_model = load_model(self.model_config)

        # 提取本stage的层
        shard = nn.ModuleList(
            full_model.layers[self.start_layer:self.end_layer]
        )

        return shard

    def execute_model(
            self,
            micro_batches: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        执行流水线并行推理
        功能：处理微批次序列，实现流水线执行
        """
        outputs = []

        for mb_idx, micro_batch in enumerate(micro_batches):
            # 步骤1：从前一阶段接收输入，如果不是第一个stage
            if self.pp_rank > 0:
                input_tensor = self._recv_from_prev_stage()
            else:
                input_tensor = micro_batch

            # 步骤2：本地前向计算
            output_tensor = self.model(input_tensor)

            # 步骤3：发送到下一阶段，如果不是最后一个stage
            if self.pp_rank < self.pp_size - 1:
                self._send_to_next_stage(output_tensor)
            else:
                outputs.append(output_tensor)

            # 步骤4：异步预取下一个micro_batch的输入，优化
            if mb_idx + 1 < len(micro_batches):
                self._prefetch_next_input(mb_idx + 1)

        return outputs

    def _send_to_next_stage(self, tensor: torch.Tensor):
        """
        P2P发送到下一阶段
        功能：使用独立stream异步发送数据
        """
        next_rank = self.pp_rank + 1

        with torch.cuda.stream(self.send_stream):
            torch.distributed.send(
                tensor.contiguous(),
                dst=next_rank,
                group=get_pipeline_model_parallel_group()
            )

    def _recv_from_prev_stage(self) -> torch.Tensor:
        """
        P2P接收从前一阶段
        功能：使用独立stream异步接收数据
        """
        prev_rank = self.pp_rank - 1

        # 预分配接收buffer
        tensor = torch.empty(
            self.expected_input_shape,
            dtype=torch.float16,
            device='cuda'
        )

        with torch.cuda.stream(self.recv_stream):
            torch.distributed.recv(
                tensor,
                src=prev_rank,
                group=get_pipeline_model_parallel_group()
            )

        # 等待接收完成
        torch.cuda.current_stream().wait_stream(self.recv_stream)

        return tensor


# 使用示例
if __name__ == "__main__":
    # 初始化配置
    from dataclasses import dataclass


    @dataclass
    class ModelConfig:
        num_hidden_layers: int = 80


    @dataclass
    class ParallelConfig:
        pipeline_parallel_size: int = 4
        pipeline_parallel_rank: int = 0


    @dataclass
    class SchedulerConfig:
        pass


    # 创建Worker
    worker = PipelineParallelWorker(
        ModelConfig(),
        ParallelConfig(),
        SchedulerConfig()
    )

    # 执行推理
    micro_batches = [torch.randn(8, 128, 4096).cuda() for _ in range(8)]
    outputs = worker.execute_model(micro_batches)
    print(f"输出数量：{len(outputs)}")