import time
from typing import Dict


def monitor_memory_continuously(block_manager, interval: int = 5):
    """
    持续监控内存状态
    功能：定期输出内存使用情况

    Args:
        block_manager：块管理器实例
        interval：监控间隔，秒
    """
    try:
        while True:
            # 收集指标
            metrics = block_manager.get_metrics()

            # 显示状态
            print(f"\n{'=' * 50}")
            print(f"时间：{time.strftime('%H:%M:%S')}")
            print(f"{'=' * 50}")
            print(f"总块数：{metrics['total_blocks']}")
            print(f"空闲块：{metrics['free_blocks']}")
            print(f"使用块：{metrics['used_blocks']}")
            print(f"共享块：{metrics['shared_blocks']}")
            print(f"利用率：{metrics['utilization']:.1%}")
            print(f"共享率：{metrics['sharing_ratio']:.1%}")

            # 警告检查
            if metrics['utilization'] > 0.9:
                print("⚠️  警告：内存利用率过高!")

            if metrics['free_blocks'] < 10:
                print("⚠️  警告：空闲块不足!")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n监控已停止")


# 使用示例
if __name__ == "__main__":
    # 创建模拟的块管理器
    class MockBlockManager:
        def get_metrics(self) -> Dict:
            import random
            total = 1000
            free = random.randint(100, 500)
            used = total - free
            shared = random.randint(0, used // 2)

            return {
                'total_blocks': total,
                'free_blocks': free,
                'used_blocks': used,
                'shared_blocks': shared,
                'utilization': used / total,
                'sharing_ratio': shared / used if used > 0 else 0.0
            }


    # 启动监控
    print("开始监控内存状态，按Ctrl+C停止...")
    monitor_memory_continuously(MockBlockManager(), interval=2)