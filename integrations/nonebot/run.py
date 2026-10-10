"""NoneBot2 启动入口。

    cd integrations/nonebot
    python run.py

它做三件事：
1. 把项目根目录加进 sys.path，让 aliceqq 可以被导入；
2. 加载项目根目录的 .env（角色与模型配置）；
3. 调用 nonebot.init() 并 nonebot.run()。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # integrations/nonebot -> 项目根

sys.path.insert(0, str(PROJECT_ROOT))

# 让 aliceqq.config 的 PROJECT_ROOT 指向真实项目根（它按模块位置推断，本来就正确）
os.environ.setdefault("ALICE_PROJECT_ROOT", str(PROJECT_ROOT))


def _bootstrap() -> None:
    try:
        import nonebot  # noqa: F401
    except ImportError:
        print(
            "缺少 NoneBot2。请先安装：\n"
            "    pip install nonebot2 nonebot-adapter-onebot\n"
        )
        raise SystemExit(2)

    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

    nonebot.init()
    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    # 加载插件（如果 pyproject.toml 已经声明，也可以交给 load_plugins）
    nonebot.load_plugin("aliceqq.adapters.nonebot_plugin_alice")

    nonebot.run()


if __name__ == "__main__":
    _bootstrap()
