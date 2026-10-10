# -*- coding: utf-8 -*-
r"""链路体检：一次看清「QQ ↔ NapCat ↔ NoneBot」这条链现在通不通。

    python scripts/链路体检.py

为什么需要：用户反馈"一段时间没人理就断线"，但要分清是**哪一种**断：

  A. 反向 WS 断了
       -> 8095 没有 Established 连接。NapCat 会自动重连（重连间隔 30 秒）。
  B. WS 好着，但 QQ 登录被**静默**踢下线
       -> 8095 连接正常，进程也活着，但发消息永远没回应。
          这是 NapCat 的已知问题（PR NapNeko/NapCatQQ#2080
          "静默离线（无 KickedOffLine 通知）后自动恢复登录"），
          只能重新登录，重连 WS 没用。

这两者从"机器人不回话"这个表象上完全一样，但对策相反，
所以要先把它们分开。本脚本只做**只读检查**，不改任何配置。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PORT = 8095


def hr(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def check_listener() -> bool:
    """8095 是否有进程在监听（NoneBot 是否还活着）。"""
    hr("1) NoneBot 进程与端口")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-NetTCPConnection -LocalPort {PORT} -State Listen | "
             "Select-Object -First 1 -ExpandProperty OwningProcess"],
            capture_output=True, text=True, timeout=30)
        pid = (out.stdout or "").strip()
    except Exception as exc:
        print(f"  查询失败：{type(exc).__name__}")
        return False
    if not pid.isdigit():
        print(f"  ✗ 没有进程监听 {PORT} —— NoneBot **没有在跑**")
        print("    对策：cd integrations\\nonebot && python run.py")
        return False
    info = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"$p=Get-Process -Id {pid}; "
         "'{0}|{1}|{2}' -f $p.ProcessName, [math]::Round($p.WS/1MB), "
         "[math]::Round(((Get-Date)-$p.StartTime).TotalHours,1)"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    parts = info.split("|")
    name = parts[0] if parts else "?"
    mem = parts[1] if len(parts) > 1 else "?"
    hours = parts[2] if len(parts) > 2 else "?"
    print(f"  ✓ PID {pid}（{name}）正在监听 {PORT}")
    print(f"    内存 {mem} MB，已运行 {hours} 小时")
    if hours not in ("?",) and float(hours) < 0.05:
        print("    注意：刚启动不久，说明之前可能断过并被重启")
    return True


def check_connection() -> bool:
    """有没有 NapCat 连上来的 Established 连接。"""
    hr("2) 反向 WebSocket 连接")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {PORT} -State Established "
             "-ErrorAction SilentlyContinue | Measure-Object).Count"],
            capture_output=True, text=True, timeout=30)
        count = int((out.stdout or "0").strip() or 0)
    except Exception:
        count = 0
    if count <= 0:
        print(f"  ✗ {PORT} 上没有已建立的连接 —— **NapCat 没连上**")
        print("    这是情况 A：反向 WS 断开。")
        print("    NapCat 会按『重连间隔』自动重连（当前配置 30 秒）。")
        print("    若长期连不上，检查 NapCat 里那条 WebSocket 客户端是否『启用』，")
        print("    以及 URL 是否为 ws://127.0.0.1:8095/onebot/v11/ws")
        return False
    print(f"  ✓ 有 {count} 条已建立的连接 —— WS 层是通的")
    print("    （注意：**WS 通不等于 QQ 在线**，见第 3 步）")
    return True


def check_napcat() -> None:
    """NapCat / QQ 进程是否在跑。"""
    hr("3) NapCat / QQ 进程")
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='QQ.exe'\" | "
         "Where-Object { $_.CommandLine -like '*NapCat*' } | "
         "Measure-Object | Select-Object -ExpandProperty Count"],
        capture_output=True, text=True, timeout=40)
    try:
        n = int((out.stdout or "0").strip() or 0)
    except ValueError:
        n = 0
    if n <= 0:
        print("  ✗ 没找到 NapCat 的 QQ 进程 —— NapCat **没有在跑**")
        return
    print(f"  ✓ NapCat 的 QQ 进程有 {n} 个（正常，QQNT 是多进程架构）")
    out2 = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='NapCatWinBootMain.exe'\" | "
         "Measure-Object | Select-Object -ExpandProperty Count"],
        capture_output=True, text=True, timeout=40)
    print(f"    NapCatWinBootMain: {(out2.stdout or '0').strip()} 个")


def check_config() -> None:
    """看 NapCat 那条反 WS 配置的关键字段。"""
    hr("4) NapCat 反 WS 配置")
    import glob
    import json
    files = glob.glob(r"D:\NapCat\**\onebot11_*.json", recursive=True)
    if not files:
        print("  没找到 onebot11_*.json（路径可能不同）")
        return
    try:
        data = json.load(open(files[0], encoding="utf-8"))
    except Exception as exc:
        print(f"  读取失败：{type(exc).__name__}")
        return
    clients = (data.get("network") or {}).get("websocketClients") or []
    for c in clients:
        print(f"  名称        : {c.get('name')}")
        print(f"  启用        : {c.get('enable')}")
        print(f"  URL         : {c.get('url')}")
        print(f"  心跳间隔    : {c.get('heartInterval')} ms")
        print(f"  重连间隔    : {c.get('reconnectInterval')} ms")
        print(f"  Token       : {'（空）' if not c.get('token') else '（已设置）'}")
    print(f"  enableLocalFile2Url: {data.get('enableLocalFile2Url')}")
    if not data.get("enableLocalFile2Url"):
        print("    提示：这一项为 false 时，本地 file:// 媒体可能发不出去。")


def main() -> int:
    print("爱丽丝链路体检（只读，不改配置）")
    print(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    alive = check_listener()
    connected = check_connection() if alive else False
    check_napcat()
    check_config()

    hr("结论")
    if not alive:
        print("  NoneBot 没在跑 —— 先把 run.py 起起来。")
    elif not connected:
        print("  情况 A：反向 WS 断开（NapCat 应会在 30 秒内自动重连）。")
        print("  若反复发生，把 NapCat 的『重连间隔』调小，或看 NapCat 控制台报错。")
    else:
        print("  链路层看起来是通的。")
        print()
        print("  如果你仍然觉得『断线了』（发消息没反应），那多半是：")
        print("    情况 B：**QQ 登录被静默踢下线**。")
        print("    NapCat 已知问题（PR NapNeko/NapCatQQ#2080：静默离线、")
        print("    无 KickedOffLine 通知、不会自动恢复）。")
        print("    WS 看着正常，但消息永远不来 —— 只能重新登录。")
        print()
        print("  怎么确认是 B：在 QQ 里给机器人自己发一条消息，")
        print("  同时盯 run.py 的控制台。**一条日志都没有** = 情况 B。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
