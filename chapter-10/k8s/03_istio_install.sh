#!/usr/bin/env bash
# 文件: chapter10/k8s/03_istio_install.sh
#
# 用途: 安装Istio minimal profile，配置推理命名空间的网格管理策略
#
# 前置条件:
#   - kubectl 已配置并指向目标集群
#   - istioctl 已安装（版本 >= 1.20）
#     安装方式: curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.20.0 sh -
#
# 使用方式:
#   chmod +x 03_istio_install.sh
#   ./03_istio_install.sh

set -euo pipefail

ISTIO_VERSION="${ISTIO_VERSION:-1.20.0}"
NAMESPACE="${NAMESPACE:-inference-prod}"

echo "==> [1/4] 验证istioctl版本"
istioctl version --remote=false

echo "==> [2/4] 安装Istio minimal profile"
# minimal profile 仅安装 istiod 控制面和 ingress gateway
# 关闭 sidecar 自动注入：推理Pod不注入sidecar，避免额外的网络跳转开销
# 推理服务的流量通过 Ingress Gateway 统一进入，无需服务网格内部的mTLS链路
istioctl install \
  --set profile=minimal \
  --set components.ingressGateways[0].enabled=true \
  --set components.ingressGateways[0].name=istio-ingressgateway \
  --set values.global.proxy.autoInject=disabled \
  --set values.pilot.env.PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true \
  --set values.pilot.traceSampling=1.0 \
  -y

echo "==> [3/4] 等待istiod就绪"
kubectl rollout status deployment/istiod -n istio-system --timeout=120s

echo "==> [4/4] 为推理命名空间配置Istio管理标签"
# istio-injection=disabled：不向该命名空间的Pod注入sidecar代理
# istio-managed=true：标记此命名空间受Istio控制面管理（供运维工具识别）
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace "${NAMESPACE}" \
  istio-injection=disabled \
  istio-managed=true \
  --overwrite

echo ""
echo "Istio ${ISTIO_VERSION} 安装完成。"
echo "Ingress Gateway 地址："
kubectl get svc istio-ingressgateway -n istio-system \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}{"\n"}'
