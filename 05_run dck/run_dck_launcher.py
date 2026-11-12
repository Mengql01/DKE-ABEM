# -*- coding: utf-8 -*-
"""
启动器：被 运行dck.bat 调用。
负责接收命令行参数，动态加载同目录下的「运行dck.py」并调用 run_trnsys()。
不修改原脚本，仅作为桥接层。
"""

import argparse
import importlib.util
import os
import sys


def load_runner():
    """从同目录加载「运行dck.py」模块。"""
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "运行dck.py")
    if not os.path.exists(target):
        print(f"[错误] 找不到运行脚本：{target}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("run_dck_module", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="TRNSYS .dck 运行启动器")
    parser.add_argument("--dck", required=True, help="要运行的 .dck 文件路径")
    parser.add_argument(
        "--mode",
        default="interactive",
        choices=["interactive", "batch", "hidden"],
        help="运行模式",
    )
    parser.add_argument(
        "--trnsys-exe",
        default=None,
        help="TRNSYS 主程序路径（可选，留空则用环境变量或脚本默认值）",
    )
    args = parser.parse_args()

    mode = args.mode.lower()

    # 各模式下是否等待仿真结束：
    #   interactive -> 不等待（GUI 交互，由用户手动关闭窗口）
    #   batch/hidden -> 等待完成，便于看到最终结果
    wait_map = {"interactive": False, "batch": True, "hidden": True}
    wait = wait_map.get(mode, False)

    module = load_runner()
    result = module.run_trnsys(
        args.dck,
        mode=mode,
        wait=wait,
        trnsys_exe=args.trnsys_exe,
    )

    print("-" * 60)
    print(result.get("message", result))
    print("-" * 60)
    sys.exit(0 if result.get("success") else 2)


if __name__ == "__main__":
    main()
