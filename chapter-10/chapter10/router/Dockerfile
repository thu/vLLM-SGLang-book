# 文件: chapter10/router/Dockerfile
#
# 构建路由分类服务镜像
# 构建方式（在 chapter10/ 目录下执行）:
#   docker build -f router/Dockerfile -t your-registry/ai-router:latest .

FROM python:3.11-slim

WORKDIR /app

# 先复制依赖文件，利用Docker层缓存加速重复构建
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.0.0 \
    prometheus-client>=0.20.0 \
    httpx>=0.27.0

# 路由服务不需要 tiktoken（默认使用字符数估算），
# 若启用 ROUTER_EXACT_TOKEN_COUNT=true 则需额外安装
# RUN pip install tiktoken>=0.7.0

COPY router/ ./router/

EXPOSE 8080

# 4 workers 适合路由服务的 IO 密集型负载（CPU密集场景请调整为 cpu_count * 2 + 1）
CMD ["uvicorn", "router.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
