# -*- coding: utf-8 -*-
r"""重置本地配置：删掉 .env，然后重新从模板生成一份干净的。

    python scripts\重置配置.py            # 删掉 .env 并从模板重建
    python scripts\重置配置.py --keep     # 只备份，不删除

**为什么需要这个脚本**：清理 `.env` 在三种 shell 里写法都不同，
用户很容易踩坑：

    cmd          del .env
    PowerShell   Remove-Item .env
    bash/WSL     rm .env

而且 `.env` 里可能已经填了 API Key，直接删会丢。所以这里做了三件事：
  1. 先备份成 `.env.bak-<时间戳>`（哪怕只是空模板也留个记录）；
  2. 再删除；
  3. 从 `.env.example` 重新生成，并**校验没有重复键**。

重复键这件事值得单独说：`.env.example` 曾经有三段重复的 TTS 配置，
后一段的空值静默覆盖了前面的值，导致"混合音色怎么调都不生效"。
所以生成之后**一定要校验**，这个脚本会把结果直接打出来。
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV = os.path.join(ROOT, ".env")
EXAMPLE = os.path.join(ROOT, ".env.example")


def duplicate_keys(path: str) -> list:
    keys = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in line:
                    continue
                keys.append(line.split("=", 1)[0].strip())
    except OSError:
        return []
    return [k for k, n in Counter(keys).items() if n > 1]


def main() -> int:
    keep_only = "--keep" in sys.argv

    if not os.path.isfile(EXAMPLE):
        print(f"找不到模板：{EXAMPLE}")
        print("仓库不完整，重新 clone 一次。")
        return 1

    # 1) 备份
    if os.path.isfile(ENV):
        backup = ENV + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
        try:
            shutil.copyfile(ENV, backup)
            print(f"已备份旧配置 -> {os.path.basename(backup)}")
            if keep_only:
                print("（--keep：不删除旧文件，到此为止）")
                return 0
        except OSError as exc:
            print(f"备份失败：{exc}")
            print("为避免丢失已填的密钥，脚本停止。请手动处理。")
            return 1
    else:
        print("当前没有 .env，直接从模板生成。")

    # 2) 删除
    if os.path.isfile(ENV):
        try:
            os.remove(ENV)
            print("已删除旧的 .env")
        except OSError as exc:
            print(f"删除失败：{exc}")
            return 1

    # 3) 从模板重建
    try:
        shutil.copyfile(EXAMPLE, ENV)
    except OSError as exc:
        print(f"从模板生成失败：{exc}")
        return 1
    print("已从 .env.example 生成新的 .env")

    # 4) 校验重复键（这是这个脚本存在的主要理由之一）
    dups = duplicate_keys(ENV)
    print()
    if dups:
        print(f"⚠ 新生成的 .env 里有 {len(dups)} 个重复键：")
        for k in sorted(dups):
            print(f"    {k}")
        print()
        print("这才是问题所在——**模板本身**有重复。请跑一次：")
        print("    python scripts\\项目自检.py")
        return 1
    print("✓ 校验通过：没有重复键")

    print()
    print("接下来要做的事：")
    print("  1) 用编辑器打开 .env，填 LLM_API_KEY（不填也能用离线 mock）")
    print("  2) 需要接 QQ 再配 OWNER_IDS 与 NapCat")
    print("  3) 验证：python -m aliceqq --setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
