# -*- coding: utf-8 -*-
r"""上传前隐私扫描：找出会**随仓库公开**的个人信息。

为什么需要单独扫这个：常规密钥扫描（sk-xxx、SESSDATA=…）抓不到
**个人标识**——QQ 号、B 站昵称、本机绝对路径、内网地址。
这些不算"凭据"，但同样是隐私，而且很容易残留在示例配置和测试用例里。

实测踩到的例子：`.env.example` 里 `OWNER_IDS=<真实QQ号>`，
`tests/test_proactive.py` 里也把它当测试夹具用了——两处都会随仓库公开，
而常规密钥扫描完全看不见。

**第一版写得很糟，记一下教训**：
  * 它把 GBK 汉字编码 `1073741819`（其实是"烫烫烫"）当成 QQ 号，
    还把 README 里写的 Windows 错误码 `0xC0000005` 十进制形式也报出来——
    凡是 5~12 位数字都当 QQ 号，误报率极高，等于没法用；
  * 它扫描了 `data/eval.json`，而那个文件**已经被 .gitignore 排除**，
    根本不会公开。

所以这一版只做两件事：
  1. **只扫会被提交的文件**（读 .gitignore 自己判断，不依赖 git 命令）；
  2. **只报真实个人标识**——从真实 `.env` 里取出实际值再反查，
     而不是靠"像不像 QQ 号"猜。

用法：
    python scripts\隐私扫描.py
只读，不改任何文件。
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys

SKIP_DIRS = {"venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
TEXT_SUFFIXES = (".py", ".md", ".json", ".txt", ".toml", ".yml", ".yaml",
                 ".example", ".prod", ".cfg", ".ini", ".ps1", ".bat", ".sh", "")

# 会随仓库公开、且**故意**包含示例号的路径（不该报警）
EXPECTED_EXAMPLE_OK = (".env.example",)

# 本机路径 / 内网地址：与 .env 内容无关，独立检查
STATIC_PATTERNS = [
    ("Windows 绝对路径（含用户名）", re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")),
    ("内网地址", re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b")),
    ("疑似 API Key", re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b")),
]

# 已知的**正常**命中，逐条白名单（附理由）。
# 为什么要白名单而不是放宽正则：放宽会让真正的泄露也漏过。
# 这里逐条登记，既压掉噪音，又保证新增命中一定会被看见。
ALLOWED = {
    # SSRF 拦截测试：**故意**用内网地址验证会被拒绝，是安全特性的一部分
    ("tests/test_search.py", "内网地址"): "SSRF 拦截测试，故意使用内网地址",
    # 文档里举的例子用的是 <用户> 占位符，不是真实用户名
    ("docs/项目状态交接.md", "Windows 绝对路径（含用户名）"): "示例写法，含 <用户> 占位符",
}


def load_ignore_patterns(root: str) -> list[str]:
    """读 .gitignore，返回可用的通配模式（够用即可，不追求与 git 完全一致）。"""
    path = os.path.join(root, ".gitignore")
    pats: list[str] = []
    if not os.path.isfile(path):
        return pats
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pats.append(line.rstrip("/"))
    return pats


def is_ignored(rel: str, patterns: list[str]) -> bool:
    """rel 是相对仓库根的 posix 路径。"""
    parts = rel.split("/")
    for pat in patterns:
        p = pat.lstrip("/")
        if "/" in p:
            # 带目录的模式：按前缀或 fnmatch 匹配
            if fnmatch.fnmatch(rel, p) or rel.startswith(p + "/"):
                return True
        else:
            # 不带目录：匹配任意一层
            if any(fnmatch.fnmatch(seg, p) for seg in parts):
                return True
    return False


# 明显是占位符/示例值的，不作为"真实值"去反查。
#
# **为什么必须有这个**：`.env.example` 里的 `OWNER_IDS` 用的是示例号 `10001`。
# 用户从模板生成 `.env` 之后，扫描器会把它当成"真实值"，
# 于是在全项目里找出 163 处 `10001` 的匹配（cli.py 的默认参数、
# 各种测试夹具……），报一大堆根本不该报的东西。
#
# 第一版设计得不严谨：**完全依赖 .env 的内容判断什么是"真实值"**，
# 于是用户配置里只要出现示例号，扫描就废了。
# 现在改成：先用这份名单排掉明显是占位的值，只对剩下的做反查。
PLACEHOLDER_VALUES = {
    "10001", "10002", "10003", "10004", "10005", "10006", "10007",
    "10008", "10020", "10030", "10040", "123456", "12345678",
    "88888888", "99999999", "123456789", "1234567890", "123456789012",
    "20001", "20002", "1", "0", "your_key", "your-key", "test",
}


def _looks_like_placeholder(value: str) -> bool:
    """这个值看起来是不是示例/占位，而不是用户真实的凭据或标识。"""
    v = value.strip()
    if not v:
        return True
    if v in PLACEHOLDER_VALUES:
        return True
    low = v.lower()
    if any(w in low for w in ("your", "example", "placeholder", "xxx",
                              "todo", "changeme", "test", "demo", "<", ">")):
        return True
    # 连续重复字符（如 000000）也当占位
    if len(set(v)) <= 2 and len(v) >= 6:
        return True
    return False


def real_secrets(root: str) -> dict[str, str]:
    """从真实 .env 里取出实际值，用它去反查代码里有没有残留。

    这比"猜哪个数字像 QQ 号"精确得多——只报**确实是你本人的**信息。

    但**不能完全信任 .env 的内容**：如果用户从模板生成配置后没改，
    里面就是示例号。所以先过一遍 `_looks_like_placeholder()`，
    否则会把示例号当成真实值、报出上百条误报（实测踩过）。
    """
    found: dict[str, str] = {}
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return found
    wanted = ("OWNER_IDS", "LLM_API_KEY", "BOCHA_API_KEY", "BILIBILI_SESSDATA",
              "BAIDU_API_KEY", "IMAGE_GEN_API_KEY", "TTS_API_KEY",
              "SEMANTIC_SCHOLAR_API_KEY", "ASR_API_KEY")
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in wanted or not value:
                continue
            for item in re.split(r"[,;\s]+", value):
                item = item.strip()
                if len(item) >= 5 and not _looks_like_placeholder(item):
                    found[item] = key
    return found


def scan(root: str) -> int:
    patterns = load_ignore_patterns(root)
    secrets = real_secrets(root)
    print(f"从 .env 提取到 {len(secrets)} 个真实值，用于反查残留")
    print(f".gitignore 模式 {len(patterns)} 条")
    print()

    hits: list[tuple[str, int, str, str]] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if is_ignored(rel, patterns):
                continue
            if not (name.endswith(TEXT_SUFFIXES) or "." not in name):
                continue
            n += 1
            try:
                with open(full, encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                for value, key in secrets.items():
                    if value in line:
                        hits.append((rel, lineno, f"真实 {key} 的值", value))
                for label, pat in STATIC_PATTERNS:
                    if (rel, label) in ALLOWED:
                        continue              # 已登记的正常命中
                    for m in pat.finditer(line):
                        hits.append((rel, lineno, label, m.group(0)))

    print(f"扫描了 {n} 个**会被提交**的文本文件")
    print()
    if not hits:
        print("✓ 未发现会公开的个人信息或凭据")
        return 0

    # 示例配置里出现真实 OWNER_IDS 是**必须修**的：
    # 否则别人 clone 下来会拿你的 QQ 号当 owner。
    print(f"发现 {len(hits)} 处需要处理：")
    print()
    for rel, lineno, label, token in hits:
        shown = token if len(token) <= 12 else f"{token[:6]}…{token[-4:]}"
        note = ""
        if rel in EXPECTED_EXAMPLE_OK and "OWNER" in label:
            note = "   <- 示例配置必须换成假号"
        print(f"  {rel}:{lineno}  [{label}] {shown}{note}")
    print()
    print("处理建议：")
    print("  * `.env.example` 里的 OWNER_IDS 必须用假号（如 10001）")
    print("  * 测试夹具用明显假号，避免和真实账号重合")
    print("  * 本机绝对路径改成相对路径或环境变量")
    return 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    raise SystemExit(scan(os.path.abspath(root)))
