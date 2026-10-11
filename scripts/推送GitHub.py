# -*- coding: utf-8 -*-
r"""把本地仓库推到 GitHub。

    python scripts\推送GitHub.py              # 检查 + 推送
    python scripts\推送GitHub.py --check      # 只检查，不推

**为什么需要这个脚本**：推送失败的原因有好几种，而且报错信息经常把人带偏。
实测踩过的例子：

    fatal: unable to access ...: Recv failure: Connection was reset

看起来像认证或仓库不存在，**其实是网络**——git 默认**不读系统代理**，
而国内直连 github.com:443 常常被重置。所以脚本第一步就先验证代理。

脚本按顺序检查，每一步失败都给出**具体该做什么**：
  1. 是不是 git 仓库、有没有提交、有没有配远程
  2. **代理能不能连到 GitHub**（最容易出问题的一步）
  3. 有没有认证过（凭据管理器里有没有 git 的凭据）
  4. 然后才真的推送

认证那一步**无法自动化**：GCM 会弹浏览器/图形窗口，必须在你的
交互式终端里完成。脚本会在需要时明确告诉你去做这一步。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER = "Zhangjiale2024"
REPO = "alice-qq-agent"
URL = f"https://github.com/{OWNER}/{REPO}.git"
PROXY = "http://127.0.0.1:11808"


def run(cmd: list, **kw) -> tuple:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore",
                           cwd=ROOT, timeout=kw.pop("timeout", 120), **kw)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def step(title: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {title}")
    if detail:
        for line in detail.splitlines():
            print(f"      {line}")
    if not ok and fix:
        print()
        print("      该怎么做：")
        for line in fix.splitlines():
            print(f"        {line}")
    print()
    return ok


def proxy_port_alive(server: str) -> bool:
    """系统代理里写的那个端口，现在到底有没有程序在监听。

    **这是实测踩到的坑，也是最容易被误判的一类故障**：
    用户的代理客户端（FlClash/xray）挂掉之后，Windows 的**系统代理设置
    仍然指向那个端口**。于是所有走系统代理的程序（浏览器、部分命令行工具）
    都会收到 connection reset / ERR_CONNECTION_RESET，
    而这些程序自己完全看不出是代理没了。

    用户当时的原始报错是：
        浏览器    Failed to load resource: net::ERR_CONNECTION_RESET
        git       Recv failure: Connection was reset
    看起来像 GitHub 被墙或认证失败，实际是**代理进程已经不在了**。
    """
    if not server:
        return True
    host = server.split(":")[0].strip()
    port = server.split(":")[-1].strip()
    if not port.isdigit():
        return True
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        return s.connect_ex((host or "127.0.0.1", int(port))) == 0
    except Exception:
        return False
    finally:
        s.close()


def system_proxy() -> str:
    """读 Windows 的系统代理设置（不依赖第三方库）。"""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
        server, _ = winreg.QueryValueEx(key, "ProxyServer")
        winreg.CloseKey(key)
        return server if enable else ""
    except Exception:
        return ""


def main() -> int:
    check_only = "--check" in sys.argv

    print("=" * 62)
    print("推送到 GitHub")
    print("=" * 62)
    print()

    # ---- 0) 先把"死代理"这个最坑的情况排除掉 ----
    server = system_proxy()
    if server and not proxy_port_alive(server):
        print("【0】系统代理")
        step("系统代理可用", False,
             f"系统代理指向 {server}，**但那个端口没有程序在监听**",
             "所有走系统代理的程序（浏览器、命令行）都会报\n"
             "    ERR_CONNECTION_RESET / Connection was reset\n"
             "看起来像被墙或认证失败，其实是**代理客户端挂了**。\n"
             "\n"
             "两个办法，任选一个：\n"
             "  A) 重新启动你的代理客户端（FlClash 等），恢复代理\n"
             "  B) 关掉系统代理，改走直连：\n"
             "       设置 -> 网络和 Internet -> 代理 -> 关闭「使用代理服务器」\n"
             "     或命令行（需要管理员）：\n"
             "       reg add \"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\"\n"
             "           /v ProxyEnable /t REG_DWORD /d 0 /f")
        print("  说明：本脚本检测到这个问题就先停下，因为继续下去只会报一堆误导性的错。")
        print()
        return 1
    if server:
        step("系统代理", True, f"{server}（端口在监听）")
    else:
        step("系统代理", True, "未启用（直连）")
    print()

    # ---- 1) 本地仓库状态 ----
    print("【1】本地仓库")
    code, out = run(["git", "rev-parse", "--is-inside-work-tree"])
    if code != 0:
        return 0 if step("是 git 仓库", False, "",
                         "先执行： git init") else 1

    code, out = run(["git", "log", "--oneline"])
    n_commits = len([l for l in out.splitlines() if l.strip()])
    if n_commits == 0:
        return 0 if step("有提交", False, "还没有任何提交",
                         "先执行： git add -A && git commit -m \"初始提交\"") else 1
    step("有提交", True, f"{n_commits} 个提交，分支 {run(['git','branch','--show-current'])[1]}")

    code, out = run(["git", "status", "--porcelain"])
    step("工作区", not out.strip(), "干净" if not out.strip() else f"有未提交改动：\n{out[:200]}")

    # ---- 2) 代理（最容易出问题的一步）----
    print("【2】网络与代理")
    sys_porxy = run(["git", "config", "--local", "--get", "http.proxy"])[1].strip()
    code, out = run(["git", "ls-remote", "https://github.com/git/git.git", "HEAD"],
                    timeout=90)
    if code != 0:
        step("git 能连到 GitHub", False, out.splitlines()[-1] if out else "",
             f"git 默认不读系统代理。给本仓库配上：\n"
             f"    git config --local http.proxy {PROXY}\n"
             f"    git config --local https.proxy {PROXY}\n"
             f"（如果代理端口不是 {PROXY}，换成你自己的）")
        return 1
    step("git 能连到 GitHub", True, f"代理 = {sys_porxy or '（直连）'}")

    # ---- 3) 远程地址 ----
    print("【3】远程仓库")
    code, out = run(["git", "remote", "get-url", "origin"])
    cur = out.strip()
    if cur != URL:
        step("远程地址", False, f"当前是 {cur or '（未设置）'}",
             f"git remote remove origin\n    git remote add origin {URL}")
        return 1
    step("远程地址", True, cur)

    # ---- 3.5) 仓库是否存在 ----
    exists = False
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{REPO}",
            headers={"User-Agent": "alice-push-helper"})
        proxy = urllib.request.ProxyHandler(
            {"https": PROXY, "http": PROXY} if sys_porxy else {})
        opener = urllib.request.build_opener(proxy)
        with opener.open(req, timeout=30) as r:
            exists = r.status == 200
    except Exception as exc:
        # 404 会抛 HTTPError，那是"不存在"，不是错误
        exists = "404" in str(exc)
    if exists:
        step("GitHub 仓库", True, f"已存在：github.com/{OWNER}/{REPO}")
    else:
        step("GitHub 仓库", True,
             f"还不存在 —— 推送时 GitHub 会**自动创建**（前提是认证通过）")

    # ---- 4) 认证状态 ----
    print("【4】认证")
    code, out = run(["cmdkey", "/list"])
    has_cred = "git:https://github" in out
    if has_cred:
        step("已有 GitHub 凭据", True, "可以直接推送")
    else:
        step("已有 GitHub 凭据", False,
             "凭据管理器里没有 git 的 GitHub 凭据",
             "这一步**只能你手动做一次**（脚本没法操作浏览器弹窗）：\n"
             "    在**交互式终端**里执行：  git push -u origin main\n"
             "    会弹出登录窗口 -> 选 \"Sign in with your browser\" -> 授权\n"
             "    成功之后凭据会被记住，以后推送都不再需要登录")
        print("  另外两个办法：")
        print("    A) 先在网页建仓库：https://github.com/new")
        print(f"       名字填 {REPO}，**不要勾** Add a README，然后重跑本脚本")
        print("    B) 用 Token：把一个有 repo 权限的 Personal Access Token 提供出来，")
        print("       可以直接做 URL 推送：")
        print(f"       git push https://<TOKEN>@github.com/{OWNER}/{REPO}.git main")
        print()
        return 1

    if check_only:
        print("（--check：只检查，不推送）")
        return 0

    # ---- 5) 推送 ----
    print("【5】推送")
    code, out = run(["git", "push", "-u", "origin", "main"], timeout=600)
    if code == 0:
        step("推送成功", True, out.splitlines()[-1] if out else "")
        print("=" * 62)
        print(f"仓库地址：https://github.com/{OWNER}/{REPO}")
        print("=" * 62)
        print()
        print("接下来建议做两件事（在网页上，各一个输入框）：")
        print(f"  1) 填显示名： https://github.com/settings/profile -> Name")
        print(f"  2) 填简介  ： 同页 -> Bio")
        print(f"  3) 给仓库加描述：https://github.com/{OWNER}/{REPO} 右侧 About 齿轮")
        return 0
    step("推送成功", False, out[-600:],
         "把上面的完整报错发出来，我帮你定位")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
