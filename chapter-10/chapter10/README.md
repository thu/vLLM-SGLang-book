# 第10章 实战项目5：混合架构生产系统（双框架协同）

## 项目代码目录结构

```
chapter10/
├── README.md                          # 本文件
├── requirements.txt                   # Python依赖
│
├── k8s/                               # Kubernetes资源配置
│   ├── 01_node_labels.yaml            # 节点标签与Taint配置
│   ├── 02_resource_quota.yaml         # ResourceQuota / LimitRange / PriorityClass
│   ├── 03_istio_install.sh            # Istio安装脚本
│   ├── 04_gateway_config.yaml         # Istio Gateway + VirtualService基础配置
│   ├── 05_vllm_deployment.yaml        # vLLM Deployment + HPA + Service
│   ├── 06_sglang_deployment.yaml      # SGLang Deployment + HPA + Service
│   ├── 07_circuit_breaker.yaml        # DestinationRule熔断配置
│   ├── 08_spot_job.yaml               # Spot节点批处理Job配置
│   └── 09_prometheus_adapter.yaml     # Prometheus Adapter Custom Metrics配置
│
├── router/                            # 智能路由服务
│   ├── classifier.py                  # 请求特征分类器（核心路由逻辑）
│   ├── server.py                      # FastAPI路由服务入口
│   ├── Dockerfile                     # 路由服务容器镜像
│   └── tests/
│       └── test_classifier.py         # 分类器单元测试
│
├── lb/                                # 负载均衡控制器
│   ├── controller.py                  # 跨框架负载均衡控制器
│   └── Dockerfile
│
├── failover/                          # 故障转移与降级
│   ├── degradation.py                 # 降级控制中间件
│   └── Dockerfile
│
├── spot/                              # Spot实例中断处理
│   └── termination_handler.py         # IMDS终止通知处理器
│
├── observability/                     # 可观测性
│   ├── prober.py                      # 端到端黑盒探针
│   ├── grafana_dashboard.json         # Grafana Dashboard定义
│   └── alert_rules.yaml               # Prometheus告警规则
│
├── cost_model/                        # 成本分析工具
│   └── analyzer.py                    # 单框架vs混合架构成本对比计算
│
└── tests/                             # 集成测试
    ├── test_routing_e2e.py            # 路由端到端测试
    └── test_failover.py               # 故障转移测试
```

## 版本基线

| 组件 | 版本 |
|------|------|
| vLLM | v0.8.x |
| SGLang | v0.5.x |
| Kubernetes | 1.29+ |
| Istio | 1.20+ |
| Python | 3.11+ |

## 快速验证

```bash
# 安装Python依赖
pip install -r requirements.txt

# 运行路由分类器单元测试
pytest router/tests/test_classifier.py -v

# 运行集成测试（需要本地k8s集群）
pytest tests/ -v --timeout=60
```
