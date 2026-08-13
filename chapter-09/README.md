# video-inference

实时视频内容理解推理加速，对应《LLM推理框架实战》第9章。

## 目录结构

```
video-inference/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── video_inference/
│       ├── __init__.py
│       ├── latency_profiler.py      # 延迟测量框架（P50/P95/P99 滑动窗口）
│       ├── video_pipeline.py        # 帧提取 + 场景变化检测 + base64 消息构建
│       ├── optimized_engine.py      # 四层优化引擎工厂函数（Layer 1~4）
│       ├── adaptive_serving.py      # 自适应并发异步推理服务（Semaphore 安全扩缩容）
│       ├── subtitle_generator.py    # 流式字幕生成（SRT 格式 + 时间戳对齐）
│       └── long_video_manager.py    # 长视频滑动窗口上下文管理
│
├── scripts/
│   ├── setup_models.py              # Step 1: 下载并验证两个模型
│   ├── baseline_benchmark.py        # Step 2: 基线测试（enforce_eager=True）
│   ├── token_compression_sweep.py   # Step 3: Visual Token 压缩参数扫描
│   ├── speculative_k_sweep.py       # Step 4: 投机解码 k 值扫描
│   ├── ablation_study.py            # Step 5: 消融实验（C0→C4 逐层叠加）
│   └── energy_efficiency.py         # Step 6: GPU 能耗效率分析（Tokens/Watt）
│
├── models/                          # 模型权重（setup_models.py 下载后生成）
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── Qwen2.5-1.5B-Instruct/
│
├── data/
│   └── test_video.mp4               # 测试视频（需自行提供）
│
└── results/                         # 实验结果 JSON（脚本运行后自动生成）
    ├── baseline_results.json
    ├── compression_pareto.json
    ├── speculative_k_sweep.json
    ├── ablation_table.json
    └── energy_results.json
```

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 下载模型并验证多模态路径
python scripts/setup_models.py

# 3. 准备测试视频（1080p，≥5 分钟，混合场景）
# cp your_video.mp4 data/test_video.mp4

# 4. 基线测试
python scripts/baseline_benchmark.py --video data/test_video.mp4

# 5. 消融实验（逐层验证各优化项收益）
python scripts/ablation_study.py --video data/test_video.mp4

# 6. 扫描最优 max_pixels
python scripts/token_compression_sweep.py --video data/test_video.mp4

# 7. 扫描最优投机解码 k 值
python scripts/speculative_k_sweep.py --video data/test_video.mp4

# 8. 能耗效率分析（需 pip install pynvml）
python scripts/energy_efficiency.py --video data/test_video.mp4
```

## 版本基线

| 组件 | 版本 |
|------|------|
| vLLM | v0.8.x |
| 目标模型 | Qwen2.5-VL-7B-Instruct |
| 草稿模型 | Qwen2.5-1.5B-Instruct |
| transformers | 4.49.x |
| Python | 3.10+ |
| CUDA | 12.1 |
| GPU | NVIDIA A100-SXM4-40GB |

## 实测性能摘要（A100-40GB，100 帧 1080p）

| 配置 | P50 | P95 | 相对基线 |
|------|-----|-----|---------|
| C0 基线（enforce_eager） | 845ms | 1178ms | — |
| C1 + FlashAttention-2 | 625ms | 872ms | -26% |
| C2 + Token 压缩（256t） | 385ms | 537ms | -54% |
| C3 + Prefix Caching | 320ms | 446ms | -62% |
| C4 + 投机解码（k=5） | 243ms | 408ms | -71% |

> 数据为作者在特定环境下的实测值，请以本地实测为准。

## max_pixels 与 visual token 数量换算

Qwen2.5-VL 使用 `patch_size=14`（每 patch 196 像素），
`token 数 ≈ max_pixels / 196`，受 `max_grid` 参数约束后取整，实际存在 ±10 token 波动。

| max_pixels | 约 token 数 | 适用场景 |
|------------|-------------|---------|
| 1_003_520 | ~1024 | 默认，OCR/细粒度检测 |
| 602_112 | ~614 | OCR 可接受精度损失时 |
| 200_704 | ~256 | 场景描述/动作识别（推荐） |
| 100_352 | ~128 | 极限延迟，OCR 不可用 |
| 50_176 | ~64 | 极限压缩，仅供实验 |

## 常见问题

**Q: 图像请求输出明显是幻觉？**  
A: 多模态路径可能未激活。运行 `scripts/setup_models.py` 验证。

**Q: 投机解码启动报 KV head 不兼容？**  
A: 运行 `setup_models.py` 中的 `verify_kv_head_compatibility()` 确认。

**Q: 能耗测量全为 0 Watt？**  
A: `pip install pynvml` 并确认用户有 GPU 功耗读取权限。

**Q: P95 偶发毛刺 >1 秒？**  
A: 可能是 KV Cache OOM 触发 vLLM 碎片整理，或 GPU 热降频。  
   降低 `gpu_memory_utilization`（建议 ≤0.90），并用 `nvidia-smi -q -d PERFORMANCE` 确认无降频。
