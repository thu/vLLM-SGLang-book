# 第8章：多轮对话 Agent 系统（旅游规划助手）

## 项目结构

```
chapter8/
├── src/
│   ├── core/
│   │   ├── state.py                # 对话状态数据模型
│   │   ├── session_manager.py      # Redis 会话管理
│   │   ├── dialogue_functions.py   # 所有 @sgl.function（模块级）
│   │   ├── dialogue_manager.py     # 对话管理器（不含 @sgl.function）
│   │   ├── context_manager.py      # 三策略上下文窗口管理
│   │   ├── tool_decision.py        # 五步工具调用决策链
│   │   └── streaming_manager.py    # 三种流式输出模式
│   ├── tools/
│   │   ├── base.py                 # 工具基类（软失败策略）
│   │   ├── weather.py              # 天气工具
│   │   ├── map.py                  # 景点搜索工具
│   │   ├── hotel.py                # 酒店查询工具
│   │   └── __init__.py             # ToolManager
│   ├── api/
│   │   ├── app.py                  # FastAPI 应用入口
│   │   └── streaming_routes.py     # SSE 流式端点
│   └── assistants/
│       ├── coding_assistant.py     # 编程助手 SGLang 函数
│       ├── education_bot.py        # 教育机器人 SGLang 函数
│       └── knowledge_base_assistant.py  # 知识库问答 SGLang 函数
├── scripts/chapter8/
│   ├── start_sglang_server.sh      # 启动 SGLang Runtime
│   ├── verify_function_calling.py  # Function Calling 三层验证
│   ├── verify_prefix_cache.py      # 前缀缓存效率验证（严格对照）
│   ├── evaluate_task_completion.py # 10 场景任务完成率评估
│   └── benchmark_response_latency.py  # 三场景延迟基准测试
├── tests/
│   ├── test_state.py               # 状态模型单元测试
│   ├── test_tools.py               # 工具层单元测试
│   └── test_context_manager.py     # 上下文管理器单元测试
├── configs/
│   ├── sglang_config.yaml
│   └── app_config.yaml
├── requirements.txt
└── pytest.ini
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 SGLang Runtime

```bash
bash scripts/chapter8/start_sglang_server.sh /path/to/chatglm3-6b
```

### 3. 启动 Redis

```bash
redis-server --daemonize yes
```

### 4. 启动 API 服务

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 运行单元测试（无需 SGLang/Redis）

```bash
pytest tests/ -v
```

## 关键技术约束

### @sgl.function 只能用于模块级函数

```python
# ✓ 正确：模块顶层定义
@sgl.function
def recognize_intent(s, user_input: str, context: list):
    ...

# ✗ 错误：不能用于类方法
class DialogueManager:
    @sgl.function  # 这会报错！
    def recognize_intent(self, s, ...):
        ...
```

### 前缀缓存验证必须使用 /flush_cache

```python
# ✓ 正确：真正的无缓存条件
await http_client.post("http://127.0.0.1:30000/flush_cache")

# ✗ 错误：无法保证无缓存
prompt = rebuild_prompt(...)  # 仍可能命中后端缓存
```

### 工具调用在流式生成前完成

流式输出在工具调用完成后才开始，避免响应中途停顿的体验问题。
工具调用超时设置为 3 秒（BaseTool 默认 10 秒，在 streaming_routes.py 中覆盖）。

## 测试环境

- 硬件: 单张 NVIDIA A100 80GB
- 模型: ChatGLM-3-6B（BF16 精度）
- 推理引擎: SGLang v0.1.17
- Python: 3.10+
- CUDA: 12.1

## 性能指标（实测）

| 指标 | 目标 | 实测 |
|------|------|------|
| 任务完成率（10轮内） | ≥90% | 90% |
| 首Token P95（无工具） | <500ms | 380-480ms |
| 前缀缓存 token 节省 | ≥40% | 约39% |
| 意图识别准确率 | ≥90% | 93% |
| 工具调用成功率 | ≥95% | 97% |
