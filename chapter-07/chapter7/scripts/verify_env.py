#!/usr/bin/env python3
# scripts/verify_env.py
# 验证环境安装是否正确
# 使用方法：python scripts/verify_env.py

import sys


def check_package(name: str, import_name: str = None, version_attr: str = "__version__"):
    """检查单个包是否可以导入，并打印版本"""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, version_attr, "（版本未知）")
        print(f"  ✓ {name:<20} {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name:<20} 未安装：{e}")
        return False


def check_cuda():
    """检查 CUDA 和 GPU 状态"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  ✓ CUDA 可用")
            print(f"  ✓ GPU: {gpu_name}（{total_gb:.1f} GB）")
            return True
        else:
            print("  ✗ CUDA 不可用，请检查 PyTorch 安装和驱动版本")
            return False
    except ImportError:
        print("  ✗ PyTorch 未安装")
        return False


def check_sglang_service():
    """检查 SGLang 服务是否运行"""
    try:
        import requests
        resp = requests.get("http://127.0.0.1:30000/health", timeout=3)
        if resp.status_code == 200:
            print("  ✓ SGLang 服务运行中（http://127.0.0.1:30000）")
            return True
        else:
            print(f"  ✗ SGLang 服务返回 HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ○ SGLang 服务未运行（{e}）")
        print("    提示：运行 scripts/start_sglang.sh 启动服务")
        return False


def main():
    print("=" * 50)
    print("Chapter 7 环境验证")
    print("=" * 50)

    print("\n[1/4] Python 版本")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        print(f"  ✓ Python {sys.version.split()[0]}")
    else:
        print(f"  ✗ Python {sys.version.split()[0]}（需要 3.10+）")

    print("\n[2/4] 核心依赖")
    all_ok = True
    packages = [
        ("sglang",       "sglang"),
        ("torch",        "torch"),
        ("pydantic",     "pydantic"),
        ("aiohttp",      "aiohttp"),
        ("aiofiles",     "aiofiles"),
        ("pymupdf",      "fitz",   "version"),
        ("python-docx",  "docx",   ""),
        ("loguru",       "loguru"),
        ("tqdm",         "tqdm"),
    ]
    for pkg_info in packages:
        name        = pkg_info[0]
        import_name = pkg_info[1] if len(pkg_info) > 1 else name
        ver_attr    = pkg_info[2] if len(pkg_info) > 2 else "__version__"
        if not check_package(name, import_name, ver_attr):
            all_ok = False

    print("\n[3/4] CUDA / GPU")
    check_cuda()

    print("\n[4/4] SGLang 服务")
    check_sglang_service()

    print("\n" + "=" * 50)
    if all_ok:
        print("核心依赖全部就绪，可以开始实验。")
    else:
        print("部分依赖缺失，请执行 pip install -r requirements.txt 后重新检查。")
    print("=" * 50)


if __name__ == "__main__":
    main()
