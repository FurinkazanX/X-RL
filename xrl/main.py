"""主入口文件"""

import argparse
import importlib
from xrl.utils.config import load_config


def main():
    print("=" * 80)
    print("X-RL 训练脚本启动")
    print("=" * 80)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="X-RL 训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    print(f"加载配置: {args.config}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化 Controller
    controller_config = config.get("controller", {})
    controller_cls = getattr(
        importlib.import_module("xrl.controllers"),
        controller_config.get("type", "DefaultController")
    )
    controller = controller_cls(config)
    
    print("Controller 创建完成")
    
    # 初始化组件
    controller.initialize()
    
    print("组件初始化完成")
    
    # 启动训练
    controller.start()
    
    # 监控训练过程
    try:
        controller.monitor()
    except KeyboardInterrupt:
        print("训练被中断")
    finally:
        # 停止训练
        controller.stop()


if __name__ == "__main__":
    main()
