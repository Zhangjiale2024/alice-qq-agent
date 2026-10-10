"""一键配置：让"拿到项目就能跑"变成一条命令。

    python -m aliceqq --setup

设计原则（都是踩过坑总结的）：

1. **每一步都要能看清发生了什么**。不静默装东西、不静默改文件，
   每一步先打印"要做什么、为什么"，再做。
2. **可重复执行**。已经做过的步骤会跳过并说明，不会重复装、重复写。
   用户中途 Ctrl+C 之后再跑一次是常态。
3. **不假装成功**。缺少外部依赖（QQ 客户端、API Key）时**如实报告**
   并给出具体动作，而不是打一堆 ✓ 让人以为已经能用了。
4. **零第三方依赖**。这个模块只用标准库——它是"还没装依赖时"
   就要能跑起来的东西。
5. **不改用户的系统**。不写注册表、不改 PATH、不动全局 Python。
   装依赖只往当前解释器装。

失败了也要给出**下一步该做什么**，因为"一键"的价值在于
把"我该怎么办"变成"照这个做就行"。
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MIN_PYTHON = (3, 10)


class Step:
    """一步的结果。ok=True 表示这步不需要再管了。"""

    def __init__(self, title: str, ok: bool, detail: str = "",
                 action: str = "", skipped: bool = False):
        self.title = title
        self.ok = ok
        self.detail = detail
        self.action = action          # 需要用户做的事（非空表示没完全就绪）
        self.skipped = skipped


def _run(cmd: List[str], timeout: int = 900) -> Tuple[int, str]:
    """跑一个子进程并把输出按 UTF-8 收回来。

    **必须显式给子进程设 UTF-8 环境**，这是实测踩出来的：
    在中文 Windows 上默认控制台编码是 cp936(GBK)，子进程（pip、
    自检脚本）往管道里写非 GBK 能表示的内容时就会报

        decoding with 'cp936' codec failed

    用户第一次在干净目录跑 `--setup` 时就撞上了这个，
    报错还发生在"装依赖"这一步，看起来像是网络或包的问题，
    完全指不到真因。

    注意 `cwd=ROOT` 也是必要的：自检脚本要靠相对路径找 `.gitignore`
    和 `tests/`，换个工作目录就会得出错误结论。
    """
    try:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore",
                           cwd=ROOT, timeout=timeout, env=env)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 各步骤

def step_python() -> Step:
    v = sys.version_info
    cur = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        return Step(
            "Python 版本", False, f"当前 {cur}",
            action=f"需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 或更高。"
                   f"去 python.org 装一个，或让 py 启动器指定更高版本")
    return Step("Python 版本", True, f"{cur}（{'、'.join(platform.uname()[:2])}）")


def _venv_python() -> str:
    """项目自带虚拟环境的解释器路径（不存在就返回空串）。

    为什么要管这个：实测踩过——用系统 Python 跑 `--setup` 时，
    依赖检查看的是**当前解释器**，而 nonebot 装在项目自己的 venv 里，
    于是它报告"缺依赖"并尝试重装（当然也装不上，因为网络受限）。
    检查得看不区分环境的**结论**：项目 venv 里有没有。
    """
    for rel in (os.path.join("venv", "Scripts", "python.exe"),
                os.path.join("venv", "bin", "python")):
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            return p
    return ""


def step_deps() -> Step:
    """检查并（可选）安装接 QQ 需要的依赖。

    核心逻辑是纯标准库，所以**不装任何东西也能跑通对话与测试**。
    NoneBot 只有"接真实 QQ"时才需要。
    """
    target = _venv_python() or sys.executable
    code, out = _run([target, "-c",
                      "import importlib.util as u;"
                      "print(int(u.find_spec('nonebot') is not None)"
                      " + int(u.find_spec('nonebot.adapters.onebot.v11') is not None))"],
                     timeout=120)
    have = out.strip().splitlines()[-1] if code == 0 and out.strip() else "0"
    where = "项目 venv" if _venv_python() else "当前解释器"
    if have == "2":
        return Step("接 QQ 的依赖", True, f"NoneBot2 + OneBot v11 已装（{where}）")

    # 没装：尝试装进**项目 venv**（若没有 venv 就装进当前解释器）
    print(f"    正在安装到{where}...")
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.isfile(req):
        cmd = [target, "-m", "pip", "install", "-q", "-r", req]
    else:
        cmd = [target, "-m", "pip", "install", "-q",
               "nonebot2", "nonebot-adapter-onebot"]
    code, out = _run(cmd)
    if code == 0:
        return Step("接 QQ 的依赖", True, f"已安装 NoneBot2 + OneBot v11（{where}）")

    last = out.strip().splitlines()[-1][:120] if out.strip() else ""
    same = "（就是当前这个解释器）" if target == sys.executable else ""
    return Step(
        "接 QQ 的依赖", False, f"自动安装失败{same}：{last}",
        action="手动执行（先激活你要用的环境）：\n"
               "      pip install nonebot2 nonebot-adapter-onebot\n"
               "    装不上也不影响离线使用：对话、测试、演示都能跑")


def step_env_file() -> Step:
    """准备 .env。

    **绝不覆盖已有 .env**——里面可能有用户填好的密钥。
    """
    env = os.path.join(ROOT, ".env")
    example = os.path.join(ROOT, ".env.example")
    if os.path.isfile(env):
        return Step(".env 配置文件", True, "已存在，保持不动", skipped=True)
    if not os.path.isfile(example):
        return Step(".env 配置文件", False, "找不到 .env.example",
                    action="仓库不完整，重新 clone 一次")
    try:
        shutil.copyfile(example, env)
    except OSError as exc:
        return Step(".env 配置文件", False, f"复制失败：{exc}",
                    action=f"手动复制： copy .env.example .env")
    return Step(".env 配置文件", True, "已从 .env.example 生成",
                action="填写 LLM_API_KEY 才能用真实模型（不填也能用离线 mock）")


def step_api_key() -> Step:
    env = os.path.join(ROOT, ".env")
    if not os.path.isfile(env):
        return Step("模型 API Key", False, "还没有 .env",
                    action="先完成上一步")
    key = ""
    try:
        with open(env, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("LLM_API_KEY=") and not s.startswith("#"):
                    key = s.split("=", 1)[1].split("#")[0].strip()
    except OSError:
        pass
    if key:
        return Step("模型 API Key", True, f"已配置（长度 {len(key)}）")
    return Step("模型 API Key", False, "未配置",
                action="在 .env 里填 LLM_API_KEY=你的key\n"
                       "    不填也可以：离线 mock 模式能跑通全部测试与演示")


def step_selftest() -> Step:
    """跑一遍测试。失败要给具体是哪几条。"""
    print("    正在跑测试（纯标准库，不需要网络）...")
    code, out = _run([sys.executable, "-m", "unittest", "discover",
                      "-s", "tests", "-t", "."])
    tail = [l for l in out.splitlines() if l.strip()][-3:]
    if code == 0:
        return Step("离线自测", True, " | ".join(tail))
    fails = [l.strip() for l in out.splitlines()
             if l.startswith(("FAIL:", "ERROR:"))][:5]
    return Step("离线自测", False, "\n".join(fails) or tail[-1] if tail else "",
                action="先看上面失败的用例名；也可以跑 python -m aliceqq --test-plan 看详情")


def step_repo_health() -> Step:
    """跑项目自检（只看必须修的那类）。

    **解析要稳**：用户实测反馈过一次"项目自检：✗（后面空白）"，
    什么都看不到，比直接报错还糟——因为完全不知道该做什么。
    所以这里当解析不出内容时，**把原始输出的末尾带出去**，
    哪怕乱码也比空白强（至少能看出是脚本崩了、还是符号没匹配上）。
    """
    script = os.path.join(ROOT, "scripts", "项目自检.py")
    if not os.path.isfile(script):
        return Step("项目自检", True, "跳过（缺脚本）", skipped=True)
    print("    正在自检（隐私、体积、文档一致性等）...")
    target = _venv_python() or sys.executable
    code, out = _run([target, script], timeout=1200)

    summary = ""
    for line in out.splitlines():
        if line.startswith("通过"):
            summary = line.strip()
    fails = [l.strip() for l in out.splitlines() if l.strip().startswith("✗")]

    if code == 0:
        return Step("项目自检", True, summary or "全部通过")

    detail = summary or "（没能从输出里解析出结论）"
    if not fails:
        # 解析失败时兜底：带出原始输出末尾，便于定位到底哪一步出的问题
        tail = [l for l in out.strip().splitlines() if l.strip()][-6:]
        detail = (detail + "\n    原始输出末尾：\n"
                  + "\n".join("      " + l for l in tail))
    return Step("项目自检", False, detail,
                action="需要修的项：\n    " + "\n    ".join(fails[:6])
                if fails else "手动跑一次看详情：python scripts\\项目自检.py")


def step_hook() -> Step:
    """装保密钩子（有 .git 才有意义）。"""
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        return Step("保密钩子", True, "跳过（还不是 git 仓库）", skipped=True)
    dst = os.path.join(ROOT, ".git", "hooks", "pre-commit")
    if os.path.isfile(dst):
        return Step("保密钩子", True, "已安装", skipped=True)
    script = os.path.join(ROOT, "scripts", "安装钩子.py")
    if not os.path.isfile(script):
        return Step("保密钩子", True, "跳过（缺脚本）", skipped=True)
    code, out = _run([sys.executable, script])
    if code == 0:
        return Step("保密钩子", True, "已安装（提交时会自动检查密钥泄露）")
    return Step("保密钩子", False, out.splitlines()[-1] if out else "",
                action=f"手动执行： python {os.path.relpath(script, ROOT)}")


# ---------------------------------------------------------------- 外部依赖

EXTERNAL = [
    ("QQ 客户端 + NapCat", "接真实 QQ 才需要",
     "1) 装 QQNT 并登录机器人账号\n"
     "    2) 装 NapCat（https://github.com/NapNeko/NapCatQQ）\n"
     "    3) NapCat 里加一个**反向 WebSocket 客户端**，地址填：\n"
     "         ws://127.0.0.1:8095/onebot/v11/ws\n"
     "    4) 然后运行： python -m aliceqq --serve"),
    ("本地 TTS（可选）", "要爱丽丝音色才需要",
     "GPT-SoVITS 部署在别处时，在 .env 里指到它的地址：\n"
     "      TTS_PROVIDER=gptsovits\n"
     "      TTS_CLONE_URL=http://127.0.0.1:9880"),
]


def main() -> int:
    print("=" * 64)
    print("爱丽丝 QQ Agent · 一键配置")
    print("=" * 64)
    print()
    print("接下来会依次检查：环境 → 依赖 → 配置 → 自测 → 保密钩子")
    print("每一步都会说明在做什么；已完成的步骤会跳过。")
    print("**不会**改动你的系统设置（不写注册表、不改 PATH）。")
    print()

    steps = [
        ("检查运行环境", step_python),
        ("检查依赖", step_deps),
        ("准备配置", step_env_file),
        ("检查密钥", step_api_key),
        ("跑一遍测试", step_selftest),
        ("项目自检", step_repo_health),
        ("安装保密钩子", step_hook),
    ]

    results: List[Step] = []
    for i, (label, fn) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {label}")
        try:
            st = fn()
        except Exception as exc:
            st = Step(label, False, f"{type(exc).__name__}: {exc}",
                      action="这一步自身出错了，请把这个报错发出来")
        results.append(st)
        mark = "✓" if st.ok else ("·" if st.skipped else "✗")
        print(f"    {mark} {st.title}：{st.detail}")
        print()

    # ---- 汇总 ----
    print("=" * 64)
    blocked = [s for s in results if not s.ok]
    if not blocked:
        print("✓ 配置完成，可以用了")
    else:
        print(f"完成，但有 {len(blocked)} 项需要你处理")
    print("=" * 64)

    print()
    print("现在就能做的事（**不需要** QQ、不需要 API Key）：")
    print("    python -m aliceqq --demo            # 完整自检演示")
    print("    python -m aliceqq \"你好\" --repl     # 交互式聊天")
    print("    python -m aliceqq --test-plan       # 看项目全景")

    if blocked:
        print()
        print("-" * 64)
        print("还需要你处理：")
        for st in blocked:
            print()
            print(f"  ✗ {st.title}")
            if st.detail:
                print(f"      {st.detail}")
            if st.action:
                for line in st.action.splitlines():
                    print(f"    {line}")

    print()
    print("-" * 64)
    print("想接真实 QQ 时，需要额外准备（这些必须手动做，无法自动化）：")
    for name, why, how in EXTERNAL:
        print()
        print(f"  · {name}（{why}）")
        for line in how.splitlines():
            print(f"    {line}")
    print()
    print("准备好之后： python -m aliceqq --serve")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
