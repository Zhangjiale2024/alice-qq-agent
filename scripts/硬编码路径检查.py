# -*- coding: utf-8 -*-
r"""查"换台机器就跑不起来"的硬编码本机路径。

为什么单列一项：这类问题**在本机永远测不出来**——
因为路径恰好是对的。别人 clone 下来才会发现，
而且报错往往很含糊（文件不存在 / 模块导入失败）。

本项目实际踩到过：唱歌功能里写死了
`D:\DSHWORK\envs\gptsovits\Scripts\python.exe`，
别人 clone 下来唱歌一定失败，而错误信息完全看不出是这个原因。

检查项：
  * 盘符绝对路径（例如 `X:\某个目录`）
  * Unix 绝对路径（家目录下的那种）
  * `os.environ` 之外直接拼死的用户目录

（这段说明**故意不写完整示例路径**：本脚本自己也会被扫，
写了真的路径出来反而会把扫描器自己命中。已经踩过一次。）

用法：
    python scripts\硬编码路径检查.py
只读。
"""
from __future__ import annotations

import os
import re
import sys

SKIP_DIRS = {"venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
SUFFIXES = (".py", ".md", ".json", ".toml", ".yml", ".yaml", ".example",
            ".prod", ".cfg", ".ini", ".ps1", ".bat", ".sh")

PATTERNS = [
    ("Windows 盘符路径", re.compile(r"(?<![\w/])([A-Za-z]:\\[^\s\"'，。）)]+)")),
    ("Unix 家目录路径", re.compile(r"(/(?:home|Users)/[^\s\"'，。）)]+)")),
]

# 逐条登记的正常命中（附理由）。宽泛地放宽正则会让真问题一起漏掉。
ALLOWED = {
    # 文档里讲"坑"时举的**反例**，本身不是代码
    "docs/项目状态交接.md": "文档里举的反例 / 说明文字",
    "docs/测试手册.md": "文档里的操作示例",
    "README.md": "README 里的操作示例",
}

# 代码里**允许**出现本机路径的地方：都是"默认值 + 可用环境变量覆盖"的写法
CODEFILE_NOTE = {
    "aliceqq/adapters/nonebot_plugin_alice.py":
        "唱歌用解释器：默认值 + SINGING_PYTHON 可覆盖",
    "aliceqq/ffmpeg.py": "ffmpeg 探测：多路径候选依次尝试",
}


def main() -> int:
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    hits: list[tuple[str, int, str, str]] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if rel in ALLOWED:
                continue
            n += 1
            try:
                lines = open(full, encoding="utf-8", errors="ignore").readlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                s = line.strip()
                # 注释里的例子不算（但代码行要算）
                is_comment = s.startswith("#") or s.startswith("//")
                for label, pat in PATTERNS:
                    for m in pat.finditer(line):
                        token = m.group(1)
                        # 跳过明显是 URL 或文档占位
                        if token.startswith("//") or "<" in token:
                            continue
                        kind = "注释示例" if is_comment else "**代码中的硬编码**"
                        hits.append((rel, lineno, f"{label}｜{kind}", token))

    print(f"扫描了 {n} 个文件")
    print()
    if not hits:
        print("✓ 未发现硬编码本机路径")
        return 0

    code_hits = [h for h in hits if "代码中的硬编码" in h[2]]
    doc_hits = [h for h in hits if "代码中的硬编码" not in h[2]]
    print(f"发现 {len(hits)} 处：代码 {len(code_hits)} 处，文档/注释 {len(doc_hits)} 处")
    print()
    if code_hits:
        print("【代码中的硬编码 —— 换台机器就会失败】")
        for rel, lineno, label, token in code_hits:
            note = CODEFILE_NOTE.get(rel)
            print(f"  {rel}:{lineno}")
            print(f"      {label}")
            print(f"      {token[:90]}")
            if note:
                print(f"      注：{note}")
        print()
    if doc_hits:
        print("【文档/注释里的路径 —— 通常是示例，确认无误即可】")
        for rel, lineno, label, token in doc_hits[:15]:
            print(f"  {rel}:{lineno}  {token[:70]}")
        if len(doc_hits) > 15:
            print(f"  …还有 {len(doc_hits) - 15} 处")
    print()
    print("修法：优先给默认值 + 环境变量覆盖，例如")
    print("    python = os.environ.get('XXX_PYTHON') or shutil.which('python') or sys.executable")
    return 1 if code_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
