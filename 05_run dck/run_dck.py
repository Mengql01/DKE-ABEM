#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简 TRNSYS 仿真运行器
- 底层：run_simulation() 按 show_window/auto_close 控制
- 推荐：run_trnsys() 使用 mode="interactive" / "batch" / "hidden"
"""

import subprocess
import os


def run_simulation(
    dck_file,
    show_window=True,
    auto_close=False,
    wait=False,
    trnsys_exe=None,
):
    """
    运行 TRNSYS 仿真（直接调用 TRNExe.exe）

    底层接口：使用 show_window / auto_close 控制行为
    推荐使用上层封装 run_trnsys(...)

    Args:
        dck_file (str): DCK 文件路径
        show_window (bool): 是否显示 TRNSYS 主窗口
        auto_close (bool): 是否在仿真结束后自动关闭窗口
                           - 仅在 show_window=True 时生效
                           - True -> 使用 /n（批处理模式）
        wait (bool): 是否等待仿真结束（阻塞调用）
        trnsys_exe (str | None): TRNSYS 主程序路径（TrnExe.exe 或 TrnEXE64.exe）
                                 若为 None，则尝试从环境变量 TRNSYS_EXE 读取，
                                 否则退回到一个默认路径示例。

    Returns:
        dict: {'success': bool, 'message': str}
    """

    # 1. 路径检查 ------------------------------------------------------------
    if not os.path.exists(dck_file):
        return {'success': False, 'message': f'文件不存在: {dck_file}'}

    # 优先使用函数参数，其次环境变量，最后默认硬编码
    if trnsys_exe is None:
        trnsys_exe = (
            os.environ.get("TRNSYS_EXE")
            or r"D:\TRNSYS18_04\Exe\TrnEXE64.exe"
        )

    if not os.path.exists(trnsys_exe):
        return {'success': False, 'message': f'找不到 TRNSYS 可执行文件: {trnsys_exe}'}

    # 2. 组装命令行（根据官方 /n、/h 参数）-----------------------------------
    cmd = [trnsys_exe, dck_file]

    # 官方说明：
    #   /n : batch mode，前台批处理，自动结束且跳过错误对话框
    #   /h : hidden batch mode，隐含 /n，完全隐藏
    if show_window:
        if auto_close:
            # 显示窗口 & 自动关闭  -> 批处理模式
            cmd.append("/n")
        else:
            # 显示窗口 & 手动关闭 -> 交互模式（不加任何开关）
            pass
    else:
        # 不显示窗口 -> 使用官方隐藏批处理模式 /h（包含 /n）
        cmd.append("/h")

    # 3. Windows 下隐藏控制台的设置 -----------------------------------------
    creationflags = 0
    startupinfo = None

    if os.name == "nt":
        if not show_window:
            # 对控制台程序：不创建额外控制台窗口
            try:
                creationflags |= subprocess.CREATE_NO_WINDOW
            except AttributeError:
                # 某些极老版本 Python 可能没有该常量，忽略即可
                pass

    # 4. 设置工作目录为 dck 所在目录（保证输出/相对路径与 TRNSYS 一致） ------
    workdir = os.path.dirname(os.path.abspath(dck_file)) or None

    # 用于日志或调试输出的命令字符串
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    # 5. 启动进程 ------------------------------------------------------------
    try:
        if wait:
            # 阻塞，等待仿真结束
            subprocess.run(
                cmd,
                check=True,
                creationflags=creationflags,
                startupinfo=startupinfo,
                cwd=workdir,
            )
            return {
                'success': True,
                'message': f'仿真已完成（同步执行）：{cmd_str}',
            }
        else:
            # 非阻塞，只负责启动
            subprocess.Popen(
                cmd,
                creationflags=creationflags,
                startupinfo=startupinfo,
                cwd=workdir,
            )
            return {
                'success': True,
                'message': f'仿真已启动（异步执行）：{cmd_str}',
            }

    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'message': f'仿真进程返回非零退出码: {e.returncode}\n命令: {cmd_str}',
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'执行失败: {e}\n命令: {cmd_str}',
        }


# --------------------------------------------------------------------------
# 推荐使用的高层接口：按“模式”调用 TRNSYS
# --------------------------------------------------------------------------
def run_trnsys(
    dck_file,
    mode="interactive",
    wait=False,
    trnsys_exe=None,
):
    """
    语义化接口：按模式运行 TRNSYS

    Args:
        dck_file (str): DCK 文件路径
        mode (str): 运行模式：
            - "interactive" : 正常交互模式（无 /n /h）
            - "batch"       : 批处理模式（前台，使用 /n）
            - "hidden"      : 隐藏批处理模式（后台，使用 /h -> 隐含 /n）
        wait (bool): 是否等待仿真结束（阻塞调用）
        trnsys_exe (str | None): TRNSYS 主程序路径（可选）

    Returns:
        dict: {'success': bool, 'message': str}
    """

    mode = (mode or "interactive").lower()

    if mode == "interactive":
        # 正常模式：有窗口，手动关闭
        show_window = True
        auto_close = False
    elif mode == "batch":
        # 批处理：有窗口，自动关（/n）
        show_window = True
        auto_close = True
    elif mode == "hidden":
        # 隐藏批处理：无窗口（/h -> 隐含 /n）
        show_window = False
        auto_close = True  # 实际无所谓，但保持语义一致
    else:
        return {
            'success': False,
            'message': f'未知模式: {mode}，请使用 "interactive" / "batch" / "hidden"',
        }

    return run_simulation(
        dck_file=dck_file,
        show_window=show_window,
        auto_close=auto_close,
        wait=wait,
        trnsys_exe=trnsys_exe,
    )


# --------------------------------------------------------------------------
# 示例用法
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # dck_file = r"D:\TRNSYS18\MyProjects\AAA\SDHW1.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\ceshi15\V1_20260224_192456.dck"
    # dck_file = r"D:\TRNSYS18_04\LLM-based Multi-Agent Memory-V5.0\trnsys_dck_generator\validation_runs\unknown\V3_20260225_154720\V3_20260225_154720.dck"
    # dck_file = r"D:\TRNSYS18_04\LLM-based Multi-Agent Memory-V5.0\trnsys_dck_generator\validation_runs\unknown\V6_20260225_201905\V6_20260225_201905.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V15_20260302_080641\V15_20260302_080641.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V19_20260302_122558\V19_20260302_122558.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V24_20260305_070701\V24_20260305_070701.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V28_20260305_090115\V28_20260305_090115.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V53_20260306_003147\V53_20260306_003147.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V56_20260306_144446\V56_20260306_144446.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V107_20260312_123932\V107_20260312_123932.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V110_20260316_153828\V110_20260316_153828.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V122_20260318_102720\V122_20260318_102720.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V129_20260318_202458\V129_20260318_202458.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V157_20260322_172137\V157_20260322_172137 - fuben.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V176_20260327_001109\V176_20260327_001109 - fuben.dck"
    # dck_file = r"D:\Desktop\ceshixiugailujing\V199_20260401_133723\V199_20260401_133723.dck"   
    # dck_file = r"D:\Desktop\ceshixiugailujing\V205_20260401_153335\V205_20260401_153335.dck" 
    # dck_file = r"D:\Desktop\ceshixiugailujing\V209_20260401_162101\V209_20260401_162101.dck"  
    # dck_file = r"D:\Desktop\ceshixiugailujing\V3_20260402_034932\V3_20260402_034932_034932.dck"
    dck_file = r"E:\Desktop\ceshixiugailujing\V296_20260510_214636\V296_20260510_214636.dck"  
    
    # 1) 交互模式：正常前台运行，需要你手动关窗口
    result = run_trnsys(dck_file, mode="interactive", wait=False)

    # 2) 批处理模式：前台运行，自动关闭窗口（官方 /n）
    # result = run_trnsys(dck_file, mode="batch", wait=True)

    # # 3) 隐藏批处理模式：完全后台（官方 /h）
    # print("\n测试：隐藏批处理模式（hidden -> /h）")
    # result = run_trnsys(dck_file, mode="hidden", wait=False)
    # print(result)
