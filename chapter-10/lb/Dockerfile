# 文件: chapter10/lb/Dockerfile
#
# 构建负载均衡控制器镜像
# 构建方式（在 chapter10/ 目录下执行）:
#   docker build -f lb/Dockerfile -t your-registry/lb-controller:latest .

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    kubernetes>=29.0.0 \
    httpx>=0.27.0 \
    prometheus-client>=0.20.0

COPY lb/ ./lb/

# 单进程运行：控制器不是HTTP服务，直接执行主循环
# 建议以 Deployment replicas=1 部署，防止多实例并发写入VirtualService
CMD ["python", "-m", "lb.controller"]
