# -*- coding: utf-8 -*-
r"""安装 git pre-commit 保密钩子（一条命令，可重复执行）。

    python scripts\安装钩子.py

为什么用 Python 而不是让人手抄 shell 命令：手抄容易漏、容易忘。
装完之后**每次 commit 都会自动**跑隐私扫描和暂存区检查，
密钥/个人信息就进不了版本库。

钩子本身是 `scripts/pre-commit`（跟着仓库走，别人 clone 下来也能装）。
"""
from __future__ import annotations

import os
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "pre-commit")
DST = os.path.join(ROOT, ".git", "hooks", "pre-commit")


def main() -> int:
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        print("这里不是 git 仓库。先执行： git init")
        return 1
    if not os.path.isfile(SRC):
        print(f"找不到钩子源文件：{SRC}")
        return 1

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    shutil.copyfile(SRC, DST)
    # 加可执行位（Windows 上无实际作用，但换到 Linux/macOS 就必需）
    try:
        mode = os.stat(DST).st_mode
        os.chmod(DST, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    print(f"已安装 pre-commit 钩子 -> {DST}")
    print()
    print("从此每次 git commit 都会自动检查：")
    print("  * 真实 .env 的值有没有残留到代码/文档里")
    print("  * 暂存区里有没有 .env / 数据库 / 媒体文件")
    print()
    print("验证是否生效：随便改一行再 git add，然后 git commit；")
    print("如果钩子正常运行，会先打印 [pre-commit] 保密的检查结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
