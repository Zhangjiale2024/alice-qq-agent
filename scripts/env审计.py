# -*- coding: utf-8 -*-
r"""审计真实 `.env` 的结构问题（**不打印任何值**）。

为什么需要：`.env` 是最容易出问题又最难发现的地方——它不进版本库，
所以没人 review，写错了也没人提醒。实测踩过两次：

  1. **重复键**：文件末尾有个空的 `TTS_VC_BLEND=`，静默覆盖了上面
     真正设置的值，导致"混合音色怎么调都不生效"；
  2. **示例号当成真号**：`--user 10001` 试出来的档案进了库，
     主动关心就一直给这个不存在的号发消息。

本脚本只报**结构**问题：重复键、空值、格式错误、可疑项。
**绝不打印任何值**，只打印键名与长度。
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

# 这些键即使是空的也正常（可选功能）
MAY_BE_EMPTY_OK = {
    "TTS_VOICE", "TTS_VC_BLEND", "TTS_VC_URL", "TTS_CLONE_PROMPT_TEXT",
    "TTS_EMOTION_REF", "ASR_MODEL", "IMAGE_GEN_MODEL", "ONEBOT_ACCESS_TOKEN",
    "LLM_API_KEY", "BOCHA_API_KEY", "BAIDU_API_KEY", "IMAGE_GEN_API_KEY",
    "TTS_API_KEY", "ASR_API_KEY", "SEMANTIC_SCHOLAR_API_KEY",
    "BILIBILI_SESSDATA", "SINGING_SCRIPT", "SINGING_PYTHON",
    "SINGING_CACHE_DIR", "TTS_CLONE_REF", "TTS_CLONE_REF2",
}

SENSITIVE = re.compile(r"KEY|TOKEN|SECRET|SESSDATA|PASSWORD|COOKIE", re.I)


def audit(path: str) -> int:
    if not os.path.isfile(path):
        print(f"找不到 {path}")
        return 2

    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    keys: list[str] = []
    empty: list[tuple[int, str]] = []
    malformed: list[tuple[int, str]] = []
    sensitive_info: list[tuple[str, int]] = []

    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            malformed.append((lineno, stripped[:40]))
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # 行内注释：`KEY=value  # 说明`
        if "#" in value and not value.startswith("#"):
            value = value.split("#", 1)[0].strip()
        keys.append(key)
        if not value:
            empty.append((lineno, key))
        if SENSITIVE.search(key) and value:
            sensitive_info.append((key, len(value)))

    print(f"文件: {path}")
    print(f"总行数 {len(lines)}，有效配置项 {len(keys)}，"
          f"其中含敏感值 {len(sensitive_info)} 项")
    print()

    problems = 0

    # 1) 重复键 —— 最隐蔽的一类：后面的会静默覆盖前面的
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    print("1) 重复键")
    if dupes:
        problems += len(dupes)
        for key in sorted(dupes):
            locs = [str(i + 1) for i, k in enumerate(keys) if k == key]
            print(f"   ✗ {key} 出现 {locs} 次 —— **后面的会静默覆盖前面的**")
            print("     这是实测踩过的坑：末尾一个空值把上面的配置覆盖了")
    else:
        print("   ✓ 无重复")

    # 2) 空值
    print()
    print("2) 空值")
    suspicious_empty = [(n, k) for n, k in empty if k not in MAY_BE_EMPTY_OK]
    if suspicious_empty:
        problems += len(suspicious_empty)
        for lineno, key in suspicious_empty:
            print(f"   ? 第 {lineno} 行 {key} 为空（不在已知可空清单里，确认是否有意为之）")
    else:
        print(f"   ✓ {len(empty)} 项为空，都在已知可空清单里")

    # 3) 格式问题
    print()
    print("3) 格式")
    if malformed:
        problems += len(malformed)
        for lineno, text in malformed:
            print(f"   ✗ 第 {lineno} 行没有 = 号: {text}")
    else:
        print("   ✓ 无")

    # 4) 敏感项清单（只报键名与长度）
    print()
    print("4) 含敏感值的键（只显示长度，不显示内容）")
    if sensitive_info:
        for key, length in sorted(sensitive_info):
            print(f"   {key:32} 长度 {length}")
        print()
        print("   ⚠ 这些值**绝不能**进版本库；确认 .env 已在 .gitignore 中")
    else:
        print("   （无）")

    print()
    print("=" * 60)
    if problems:
        print(f"发现 {problems} 个结构问题，建议逐一确认")
    else:
        print("结构检查通过")
    return 1 if problems else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ".env"
    raise SystemExit(audit(target))
