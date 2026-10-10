"""一键冒烟：跑测试 + 端到端演示 + 模板变量检查。

    python scripts/smoke.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, args: list) -> bool:
    print(f"\n=== {label} ===")
    result = subprocess.run([sys.executable, *args], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[失败] {label} 退出码 {result.returncode}")
        return False
    return True


def main() -> int:
    checks = [
        ("单元测试", ["-m", "unittest", "discover", "-s", "tests", "-t", "."]),
        ("提示词模板变量检查", ["-m", "aliceqq", "--check"]),
        ("端到端演示", ["-m", "aliceqq", "--demo"]),
    ]
    failed = [label for label, args in checks if not run(label, args)]
    print("\n" + ("全部通过" if not failed else f"失败项：{', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
