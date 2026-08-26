# vllm/model_executor/layers/fused_moe/fused_moe.py

class FusionManager:
    """
    算子融合决策引擎
    功能：根据算子模式和上下文自动选择最优融合策略
    """

    def __init__(self, model_config, device_capability):
        self.model_config = model_config
        self.device_capability = device_capability
        self.fusion_rules = self._build_fusion_rules()

    def _build_fusion_rules(self):
        """构建融合规则"""
        rules = []

        # 规则1：逐点操作总是融合
        rules.append({
            'pattern': ['gelu', 'add', 'dropout'],
            'fusion_type': 'elementwise',
            'condition': lambda: True
        })

        # 规则2:LayerNorm + Linear在Prefill阶段融合
        rules.append({
            'pattern': ['layer_norm', 'linear'],
            'fusion_type': 'vertical',
            'condition': lambda phase: phase == 'prefill'
        })

        # 规则3:Attention根据序列长度选择
        rules.append({
            'pattern': ['attention'],
            'fusion_type': 'flash_attention',
            'condition': lambda seq_len: seq_len >= 512
        })

        return rules

    def should_fuse(self, op_sequence, context):
        """判断是否应该融合"""
        for rule in self.fusion_rules:
            if self._match_pattern(op_sequence, rule['pattern']):
                if rule['condition'](context):
                    return rule['fusion_type']
        return None

    def _match_pattern(self, ops, pattern):
        """模式匹配"""
        if len(ops) < len(pattern):
            return False
        return all(op.type == p for op, p in zip(ops, pattern))


# 使用示例
if __name__ == "__main__":
    manager = FusionManager(model_config=None, device_capability=8.0)
    # 测试融合决策
    ops = [{'type': 'gelu'}, {'type': 'add'}, {'type': 'dropout'}]
    fusion_type = manager.should_fuse(ops, context={'phase': 'prefill'})
    print(f"融合类型：{fusion_type}")