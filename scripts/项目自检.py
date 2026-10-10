# -*- coding: utf-8 -*-
r"""项目自检：上传/交付前跑一次，确认没有明显问题。

    python scripts\项目自检.py

它不是"再跑一遍单元测试"——测试只管代码逻辑对不对，
这个脚本管**测试管不到的那些事**：

  1. 文档与实现是否一致（README 里写的数字过期没有）
  2. 是否混入了不该提交的文件（凭据/数据库/媒体）
  3. 关键文件是否齐全（LICENSE、.gitignore、CI）
  4. 真实 .env 有没有结构问题
  5. 代码里有没有静默吞异常、BOM 这类难查的写法
  6. 敏感信息有没有残留

每条检查都给出**可执行的修法**，而不是只报"有问题"。
只读，不改任何文件。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, level: str, title: str, detail: str = "") -> None:
        self.items.append((level, title, detail))
        icon = {PASS: "✓", WARN: "?", FAIL: "✗"}[level]
        print(f"  {icon} {title}")
        if detail:
            for line in detail.splitlines():
                print(f"      {line}")

    def summary(self) -> int:
        n_fail = sum(1 for lv, _, _ in self.items if lv == FAIL)
        n_warn = sum(1 for lv, _, _ in self.items if lv == WARN)
        n_pass = sum(1 for lv, _, _ in self.items if lv == PASS)
        print()
        print("=" * 64)
        print(f"通过 {n_pass}｜需注意 {n_warn}｜必须修 {n_fail}")
        print("=" * 64)
        return 1 if n_fail else 0


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", cwd=ROOT, timeout=900)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 检查项

def check_files(rep: Report) -> None:
    print("\n【1】关键文件")
    must = {
        "README.md": "项目说明（简历第一眼看到的东西）",
        "LICENSE": "许可证；没有它别人法律上不能用你的代码",
        ".gitignore": "防止凭据/数据库被提交",
        ".env.example": "配置模板（不能含真实密钥）",
        "requirements.txt": "依赖声明",
        ".github/workflows/tests.yml": "CI；706 项测试的自动化证据",
        "aliceqq/testplan.py": "一条命令看项目状态",
    }
    missing = [f"{p}  <- {why}" for p, why in must.items()
               if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        rep.add(FAIL, f"缺少 {len(missing)} 个关键文件", "\n".join(missing))
    else:
        rep.add(PASS, f"关键文件齐全（{len(must)} 个）")


def check_gitignore(rep: Report) -> None:
    print("\n【2】.gitignore 覆盖")
    path = os.path.join(ROOT, ".gitignore")
    if not os.path.isfile(path):
        rep.add(FAIL, "没有 .gitignore", "凭据会被提交，必须加")
        return
    content = open(path, encoding="utf-8").read()
    need = {
        ".env": "凭据文件",
        "venv": "虚拟环境（几百 MB）",
        "__pycache__": "字节码缓存",
        "data/test-run": "测试残留（曾经 1248 个目录 / 84 MB）",
        "media": "QQ 收发的图片/语音缓存",
    }
    absent = [f"{k}  <- {v}" for k, v in need.items()
              if not re.search(rf"^\s*/?{re.escape(k)}", content, re.M)]
    if absent:
        rep.add(WARN, f".gitignore 可能漏了 {len(absent)} 项", "\n".join(absent))
    else:
        rep.add(PASS, ".gitignore 覆盖了关键项")


def check_secrets(rep: Report) -> None:
    print("\n【3】敏感信息")
    # 真实 .env 是否被忽略
    if os.path.isdir(os.path.join(ROOT, ".git")):
        code, out = run(["git", "check-ignore", "-q", ".env"])
        if code != 0:
            rep.add(FAIL, ".env **没有被 git 忽略**",
                    "立刻修：把 .env 加进 .gitignore，并确认 git status 里看不到它")
        else:
            rep.add(PASS, ".env 已被 git 忽略")
        # 有没有误提交历史
        code, out = run(["git", "log", "--oneline", "--all", "--", ".env"])
        if code == 0 and out.strip():
            rep.add(FAIL, "git 历史里**出现过 .env**",
                    "密钥已经泄漏。必须轮换所有密钥，并重写历史\n"
                    "    参考：git filter-repo 或直接重建仓库")
        else:
            rep.add(PASS, "git 历史里没有 .env")
    else:
        rep.add(WARN, "还不是 git 仓库", "git init 之后再跑一次本检查")

    # 隐私扫描
    script = os.path.join(ROOT, "scripts", "隐私扫描.py")
    if os.path.isfile(script):
        code, out = run([PY, script, "."])
        if code == 0:
            rep.add(PASS, "隐私扫描通过（真实凭据未残留在代码/文档里）")
        else:
            tail = "\n".join(out.strip().splitlines()[-14:])
            rep.add(FAIL, "隐私扫描未通过", tail)


def check_env_structure(rep: Report) -> None:
    print("\n【4】真实 .env 结构")
    script = os.path.join(ROOT, "scripts", "env审计.py")
    env = os.path.join(ROOT, ".env")
    if not os.path.isfile(script) or not os.path.isfile(env):
        rep.add(WARN, "跳过（缺 .env 或审计脚本）")
        return
    code, out = run([PY, script, env])
    # 只看"重复键"和"BOM"这两个真问题
    if "✗" in out:
        bad = [l for l in out.splitlines() if "✗" in l]
        rep.add(FAIL, ".env 有结构问题（重复键会让配置静默失效）", "\n".join(bad[:8]))
    else:
        rep.add(PASS, ".env 无重复键等结构问题")


def check_docs_consistency(rep: Report) -> None:
    print("\n【5】文档与实现一致")
    # README 里写的测试数 vs 实际
    readme = os.path.join(ROOT, "README.md")
    if not os.path.isfile(readme):
        rep.add(FAIL, "没有 README.md")
        return
    text = open(readme, encoding="utf-8", errors="ignore").read()
    actual = 0
    tests_dir = os.path.join(ROOT, "tests")
    if os.path.isdir(tests_dir):
        for name in os.listdir(tests_dir):
            if name.startswith("test_") and name.endswith(".py"):
                src = open(os.path.join(tests_dir, name), encoding="utf-8",
                           errors="ignore").read()
                actual += len(re.findall(r"^\s+def test_", src, re.M))
    mentioned = [int(n) for n in re.findall(r"\*\*(\d{2,4})\s*项", text)]
    if actual and mentioned:
        if actual in mentioned:
            rep.add(PASS, f"README 的测试数 {actual} 与实际一致")
        else:
            rep.add(FAIL, f"README 写的测试数与实际不符",
                    f"README 提到 {sorted(set(mentioned))}，实际 {actual}\n"
                    f"    修法：把 README 里的数字改成 {actual}")
    else:
        rep.add(WARN, f"README 里没写测试数（实际 {actual} 项）")

    # 过期的版本名/功能声明
    stale = []
    if "未实现 AI 语音" in text and os.path.isfile(os.path.join(ROOT, "aliceqq", "generation.py")):
        stale.append("README 说『未实现 AI 语音』，但 generation.py 已存在")
    if stale:
        rep.add(FAIL, "README 有过期声明", "\n".join(stale))
    else:
        rep.add(PASS, "未发现过期功能声明")


def check_tests(rep: Report) -> None:
    print("\n【6】测试")
    code, out = run([PY, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    if code == 0:
        rep.add(PASS, "全部测试通过", " | ".join(tail))
    else:
        rep.add(FAIL, "测试未通过", "\n".join(tail))
    # 代码指纹
    code, out = run([PY, "-c",
                     "import sys; sys.path.insert(0,'.'); "
                     "from aliceqq.testplan import code_fingerprint; "
                     "print(code_fingerprint())"])
    if code == 0 and out.strip():
        rep.add(PASS, f"代码指纹 {out.strip().splitlines()[-1]}")


def check_code_smells(rep: Report) -> None:
    print("\n【7】代码写法")
    script = os.path.join(ROOT, "scripts", "代码巡检.py")
    if not os.path.isfile(script):
        rep.add(WARN, "跳过（缺代码巡检脚本）")
        return
    code, out = run([PY, script, "."])
    counts = dict(re.findall(r"^\s{2}(\S+)\s+(\d+)$", out, re.M))
    # 注意：要匹配 **[BOM]** 这个标记，不能只找 "BOM" 三个字母——
    # 巡检脚本的说明文字里本身就写了 "BOM" 这个词，
    # 用 `"BOM" in out` 判断会永远误报。（第一版就是这么错的。）
    bom_lines = [l for l in out.splitlines() if "[BOM]" in l]
    if bom_lines:
        rep.add(FAIL, "有文件以 UTF-8 BOM 开头",
                "若第一行是配置项，键名会多一个不可见字符而静默失效\n"
                + "\n".join("    " + l.strip() for l in bom_lines[:5]))
    else:
        rep.add(PASS, "无 BOM 问题")
    silent = int(counts.get("静默吞异常", 0))
    long_fn = int(counts.get("函数过长", 0))
    if silent:
        rep.add(WARN, f"静默吞异常 {silent} 处",
                "多数是合理的安全写法；但请确认没有『吞掉后返回默认值』\n"
                "    导致功能静默失效（如 _was_mentioned 吞异常后 return False）")
    else:
        rep.add(PASS, "无静默吞异常")
    if long_fn:
        rep.add(WARN, f"函数过长 {long_fn} 个",
                "信号是这段逻辑可能混了多件事；不影响正确性，但影响可读性")


def _committed_files() -> list[str]:
    """取**会被提交**的文件列表。

    优先用 git（最准，直接问它）；没有 git 或不是仓库时退回手工遍历 +
    .gitignore 判断。第一版这里直接 `os.walk` 全盘统计，
    结果把 84 MB 的测试残留也算进去了，报"151 MB 偏大"——
    那个数字没有意义，因为那些文件根本不会被提交。
    """
    if os.path.isdir(os.path.join(ROOT, ".git")):
        code, out = run(["git", "ls-files", "--cached", "--others",
                         "--exclude-standard"])
        if code == 0:
            return [l.strip() for l in out.splitlines() if l.strip()]
    # 退路：手工遍历 + 读 .gitignore
    patterns: list[str] = []
    gi = os.path.join(ROOT, ".gitignore")
    if os.path.isfile(gi):
        patterns = [l.strip().rstrip("/")
                    for l in open(gi, encoding="utf-8", errors="ignore")
                    if l.strip() and not l.startswith("#")]
    import fnmatch
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {"venv", "__pycache__", ".git", "node_modules"}]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT).replace("\\", "/")
            parts = rel.split("/")
            if any(fnmatch.fnmatch(seg, p) or fnmatch.fnmatch(rel, p)
                   for p in patterns for seg in [parts[0]]):
                continue
            files.append(rel)
    return files


def check_repo_size(rep: Report) -> None:
    print("\n【8】仓库体积（只算会被提交的文件）")
    files = _committed_files()
    total = 0
    biggest: list[tuple[int, str]] = []
    for rel in files:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        total += size
        biggest.append((size, rel))
    mb = total / 1048576
    biggest.sort(reverse=True)
    detail = "最大的 5 个：\n" + "\n".join(
        f"    {s / 1024:.0f} KB  {r}" for s, r in biggest[:5])
    if mb > 20:
        rep.add(WARN, f"待提交 {mb:.1f} MB / {len(files)} 个文件（偏大）", detail)
    else:
        rep.add(PASS, f"待提交 {mb:.1f} MB / {len(files)} 个文件（适合上传）", detail)


def main() -> int:
    print("=" * 64)
    print("项目自检（只读）")
    print("=" * 64)
    rep = Report()
    for fn in (check_files, check_gitignore, check_secrets, check_env_structure,
               check_docs_consistency, check_tests, check_code_smells,
               check_repo_size):
        try:
            fn(rep)
        except Exception as exc:
            rep.add(FAIL, f"{fn.__name__} 自身出错",
                    f"{type(exc).__name__}: {exc}")
    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
