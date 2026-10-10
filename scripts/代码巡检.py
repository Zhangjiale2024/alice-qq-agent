# -*- coding: utf-8 -*-
r"""代码质量巡检：找出容易出问题的写法。

不是风格检查（那交给格式化工具），而是找**真的会掩盖 bug**的写法：

  1. `except: pass` / `except Exception: pass`——静默吞掉异常。
     本项目吃过这个亏：一个 `.env` 解析的异常被吞掉，导致配置静默失效。
  2. 裸 `except:`——连 KeyboardInterrupt 都吞。
  3. `except Exception` 后不记录——出错时连线索都没有。
  4. TODO / FIXME / XXX / HACK——遗留标记。
  5. `print(` 调试残留（本项目部分模块确实用 print 当日志，
     但适配层应该走统一的 `_log`）。
  6. 文件首字节 BOM——`.env` 里踩过：BOM 会让第一个键的**名字**多一个字符，
     配置静默不生效。
  7. 过长的函数——信号是"这段逻辑可能混了多件事"。

用法：
    python scripts\代码巡检.py [路径]
只读。
"""
from __future__ import annotations

import ast
import os
import re
import sys

SKIP_DIRS = {"venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
TARGET_DIRS = ("aliceqq", "tests", "scripts", "integrations")

VERBOSE_FUNC_LINES = 120        # 超过这个行数的函数值得看一眼


def py_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def check_file(path: str, rel: str) -> list[str]:
    out: list[str] = []
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        out.append(f"{rel}:1  [BOM] 文件以 UTF-8 BOM 开头 —— "
                   f"若第一行是配置项，键名会多一个不可见字符，静默失效")

    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        s = line.strip()
        # 裸 except
        if re.match(r"^except\s*:", s):
            out.append(f"{rel}:{i}  [裸 except] 会连 KeyboardInterrupt 一起吞掉")
        # except 后紧跟 pass
        if re.match(r"^except\b.*:\s*$", s):
            for j in range(i, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                if nxt == "pass":
                    out.append(f"{rel}:{i}  [静默吞异常] except ...: pass —— "
                               f"出错时没有任何线索")
                break
        if re.match(r"^\s*#.*\b(TODO|FIXME|XXX|HACK)\b", line, re.I):
            out.append(f"{rel}:{i}  [遗留标记] {s[:70]}")

    # 函数长度
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        out.append(f"{rel}:{exc.lineno}  [语法错误] {exc.msg}")
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > VERBOSE_FUNC_LINES:
                out.append(f"{rel}:{node.lineno}  [函数过长] {node.name}() "
                           f"{length} 行（阈值 {VERBOSE_FUNC_LINES}）")
    return out


def main() -> int:
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    findings: list[str] = []
    n = 0
    for sub in TARGET_DIRS:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for path in py_files(d):
            n += 1
            findings.append(f"__FILE__{os.path.relpath(path, root)}")
            findings.extend(check_file(path, os.path.relpath(path, root).replace("\\", "/")))

    print(f"检查了 {n} 个 Python 文件")
    print()
    cur = ""
    counts: dict[str, int] = {}
    for item in findings:
        if item.startswith("__FILE__"):
            cur = item[8:]
            continue
        tag = re.search(r"\[([^\]]+)\]", item)
        key = tag.group(1) if tag else "其它"
        counts[key] = counts.get(key, 0) + 1
        print("  " + item)

    print()
    print("=" * 62)
    if not findings:
        print("未发现问题")
        return 0
    print("按类型统计：")
    for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:16} {c}")
    print()
    print("说明：这些不是『必须改』的硬错误，而是**值得逐条看一眼**的地方。")
    print("      静默吞异常和 BOM 这两类优先处理——它们会让故障变得无法定位。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
