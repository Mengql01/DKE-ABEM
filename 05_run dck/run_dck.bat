@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
title TRNSYS .dck 运行器

cd /d "%~dp0"

set "LAUNCHER=run_dck_launcher.py"
set "RUNNER=运行dck.py"

:: ====================================================================
::  第一步：检测 Python 环境
::  （--version 等子进程加 <nul，避免在重定向场景下吞掉后续输入）
:: ====================================================================
set "PYEXE="

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 --version >nul 2>nul <nul
    if !errorlevel!==0 set "PYEXE=py -3"
)

if not defined PYEXE (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python --version >nul 2>nul <nul
        if !errorlevel!==0 set "PYEXE=python"
    )
)

if not defined PYEXE (
    echo.
    echo ============================================================
    echo  [错误] 未检测到 Python 环境
    echo ============================================================
    echo.
    echo  本程序需要 Python 3 才能运行。
    echo  请先安装 Python，安装时务必勾选：
    echo      [v] Add Python to PATH
    echo.
    echo  下载地址：https://www.python.org/downloads/
    echo.
    echo  安装完成后，重新双击本文件即可。
    echo.
    choice /c YN /n /m "是否现在打开 Python 下载页面？(Y=是 / N=否): "
    if !errorlevel!==1 start "" "https://www.python.org/downloads/"
    echo.
    pause
    exit /b 1
)
echo.
echo ============================================================
echo   TRNSYS .dck 运行器
echo ============================================================
echo  已检测到 Python：
for /f "delims=" %%v in ('%PYEXE% --version 2^>^&1 ^<nul') do echo      %%v
echo ============================================================

:: ====================================================================
::  第二步：检测必要文件
:: ====================================================================
if not exist "%RUNNER%" (
    echo.
    echo  [错误] 当前目录缺少核心脚本：%RUNNER%
    echo  请确认本 .bat 与「运行dck.py」放在同一文件夹。
    echo.
    pause
    exit /b 1
)
if not exist "%LAUNCHER%" (
    echo.
    echo  [错误] 当前目录缺少启动器：%LAUNCHER%
    echo.
    pause
    exit /b 1
)

:: ====================================================================
::  第三步：检测依赖（运行dck.py 仅用标准库，做一次健康检查）
:: ====================================================================
%PYEXE% -c "import subprocess, os, importlib.util" >nul 2>nul <nul
if not %errorlevel%==0 (
    echo.
    echo  [警告] Python 标准库导入异常，环境可能损坏，建议重装 Python。
    echo.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo.
    echo  检测到 requirements.txt，正在检查依赖...
    %PYEXE% -m pip install -r requirements.txt --quiet --disable-pip-version-check <nul
    if not !errorlevel!==0 (
        echo  [警告] 部分依赖安装失败，将尝试继续运行。
    ) else (
        echo  依赖检查完成。
    )
)
:: ====================================================================
::  第四步：选择运行模式（默认模式一：交互模式）
::  说明：用哨兵值区分「用户按回车（空串）」与「stdin 已结束/EOF」。
::        正常双击运行时 set /p 会阻塞等待键盘，不会触发 EOF 分支；
::        EOF 仅在被管道/重定向且输入耗尽时出现，连续多次即退出，防止空转。
:: ====================================================================
set "EOFHITS=0"
:SELECT_MODE
echo.
echo ------------------------------------------------------------
echo   请选择运行模式：
echo.
echo     [1] 交互模式   （前台运行，显示窗口，手动关闭）  [默认]
echo     [2] 批处理模式 （前台运行，仿真结束后自动关闭）
echo     [3] 隐藏模式   （完全后台运行，不显示窗口）
echo ------------------------------------------------------------
set "MODE=interactive"
set "CHOICE=#EOF#"
set /p "CHOICE=请输入 1/2/3 后回车（直接回车=默认交互模式）: "

if "!CHOICE!"=="#EOF#" goto ON_EOF
set "EOFHITS=0"

if not defined CHOICE goto INPUT_PATH
if "!CHOICE!"=="" goto INPUT_PATH
if "!CHOICE!"=="1" goto INPUT_PATH
if "!CHOICE!"=="2" (set "MODE=batch" & goto INPUT_PATH)
if "!CHOICE!"=="3" (set "MODE=hidden" & goto INPUT_PATH)

echo  [提示] 输入无效，请输入 1、2 或 3（或直接回车用默认）。
goto SELECT_MODE

:: ====================================================================
::  第五步：输入 .dck 文件路径并执行
:: ====================================================================
:INPUT_PATH
echo.
echo ------------------------------------------------------------
echo   当前模式：!MODE!
echo ------------------------------------------------------------
echo   请输入要运行的 .dck 文件路径：
echo     - 可直接把 .dck 文件拖到本窗口后回车
echo     - 路径含空格也没关系，无需手动加引号
echo     - 输入 q 回车 可退出程序
echo     - 输入 m 回车 可返回上一步重新选择模式
echo ------------------------------------------------------------
set "DCKPATH=#EOF#"
set /p "DCKPATH=.dck 路径: "

if "!DCKPATH!"=="#EOF#" goto ON_EOF
set "EOFHITS=0"

if not defined DCKPATH (
    echo  [提示] 路径不能为空。
    goto INPUT_PATH
)
if "!DCKPATH!"=="" (
    echo  [提示] 路径不能为空。
    goto INPUT_PATH
)
if /i "!DCKPATH!"=="q" goto END
if /i "!DCKPATH!"=="m" goto SELECT_MODE

set "DCKPATH=!DCKPATH:"=!"

if not defined DCKPATH (
    echo  [提示] 路径不能为空。
    goto INPUT_PATH
)

if not exist "!DCKPATH!" (
    echo.
    echo  [错误] 文件不存在：!DCKPATH!
    echo  请检查路径是否正确。
    echo.
    goto INPUT_PATH
)

echo.
echo ============================================================
echo   即将运行：
echo     文件：!DCKPATH!
echo     模式：!MODE!
echo ============================================================
echo.

%PYEXE% "%LAUNCHER%" --dck "!DCKPATH!" --mode !MODE!
set "RUNRC=!errorlevel!"

echo.
if "!RUNRC!"=="0" (
    echo  [完成] 已成功执行。
) else (
    echo  [注意] 执行返回码：!RUNRC!（非 0 表示可能出错，请查看上方信息）
)

echo.
choice /c YN /n /m "是否继续运行另一个 .dck 文件？(Y=继续 / N=退出): "
if !errorlevel!==1 (
    echo.
    goto SELECT_MODE
)
goto END

:: ====================================================================
::  EOF 兜底：stdin 已结束（被管道/重定向且输入耗尽）。
::  正常双击运行不会到这里；连续命中即退出，避免空转刷屏。
:: ====================================================================
:ON_EOF
set /a EOFHITS+=1
if !EOFHITS! geq 3 (
    echo.
    echo  [提示] 未检测到输入。请直接双击本文件运行，
    echo         不要通过管道或重定向方式调用。
    goto END
)
goto SELECT_MODE

:END
echo.
echo  已退出。感谢使用。
echo.
endlocal
pause
exit /b 0
