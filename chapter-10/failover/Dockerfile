# 文件: chapter10/failover/Dockerfile
#
# 构建降级控制服务镜像
# 构建方式（在 chapter10/ 目录下执行）:
#   docker build -f failover/Dockerfile -t your-registry/degradation-ctrl:latest .

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    redis>=5.0.0 \
    prometheus-client>=0.20.0

COPY failover/ ./failover/

EXPOSE 8081

CMD ["uvicorn", "failover.degradation:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "2"]
