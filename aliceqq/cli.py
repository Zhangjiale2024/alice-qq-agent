"""本地模拟器：不接 QQ 也能完整跑通 V1 流程。

**第一次用请从这里开始：**

    python -m aliceqq --setup     # 一键配置（环境/依赖/.env/自测），可重复执行
    python -m aliceqq --demo      # 完整自检演示（不需要 API Key）
    python -m aliceqq --repl      # 交互式聊天
    python -m aliceqq --serve     # 配好 QQ 之后启动机器人

其它用法示例：
    python -m aliceqq --user 10001 --name 小星
    python -m aliceqq --group 88888 --user 10001 --name 小星 --mention
    python -m aliceqq --show-prompt "你好"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import AppConfig, PROJECT_ROOT
from .db import Storage
from .engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT


def _reset_database(db_path: Path) -> None:
    """删除数据库文件。

    Windows 上文件可能被残留进程锁住（WinError 32），这里不能直接抛异常——
    否则一个"清空再演示"的动作会把整个命令搞崩。删不掉就退化为清空表内容。
    """
    locked = False
    for suffix in ("-wal", "-shm", ""):
        candidate = Path(str(db_path) + suffix)
        if not candidate.exists():
            continue
        try:
            candidate.unlink()
        except OSError:
            locked = True
            break
    if not locked:
        return
    print("[提示] 数据库文件被占用，无法删除，改为清空表内容。")
    try:
        from .db import Storage
        storage = Storage(db_path)
        with storage._cursor() as cur:
            for table in ("conversation_log", "user_profile", "memory_fact",
                          "group_config", "usage_counter", "kv_store"):
                cur.execute(f"DELETE FROM {table}")
        storage.close()
    except Exception as exc:
        print(f"[提示] 清空数据库失败（继续执行）：{exc}")


def build_engine(args) -> Engine:
    config = AppConfig.load(args.config)
    db_path = config.db_file
    if args.reset and db_path.exists():
        _reset_database(db_path)
    # 关键：演示/模拟/模板检查必须离线可用。
    # 即使 .env 配了真实服务商，这里也强制用 mock，避免"跑个演示结果在联网"。
    # 想用真实模型请加 --live（或用 --evaluate）。
    if not getattr(args, "live", False):
        config.llm.provider = "mock"
        config.llm.api_key = ""
    return Engine(config)


def print_trace(outcome, config=None) -> None:
    r"""打印一次交互的**决策链**。

    为什么要单独做这个：Agent 类项目的价值有一半在"它怎么决定的"——
    用了哪个模型、是否触发工具、调了什么参数、工具真的成功了吗、
    有没有被护栏拦下。只给最终回复的话，这些全都看不见，
    读者只能听你口述。

    输出形如：

        ── 决策链 ──
        模型      deepseek / deepseek-chat
        工具目录  remember ✓  web_search ✓  academic_search ✓
        ① 模型决定调用  web_search({"query": "蔚蓝档案 新活动"})
        ② 执行结果      成功：检索到 5 条，请给出标题与链接后作答
        ③ 二次生成      带工具结果再问一次模型 -> 最终回复
        护栏      memory_rejected: 1

    **只打印真实发生过的事**：没有工具调用就不打这一段，
    不存在的阶段绝不编造。
    """
    debug = (outcome.debug or {}) if hasattr(outcome, "debug") else {}
    print()
    print("── 决策链 ──")
    print(f"  决策      {outcome.decision}"
          + (f"（{outcome.reason}）" if outcome.reason else ""))
    provider = debug.get("provider") or "?"
    model = debug.get("model") or "?"
    print(f"  模型      {provider} / {model}")

    if config is not None:
        try:
            from .tools import registry
            reg = registry(config,
                           search_available=config.search.enabled,
                           academic_available=getattr(config.academic, "enabled", False))
            items = [f"{s.name}{'✓' if s.available else '✗'}" for s in reg.values()]
            print(f"  工具目录  {'  '.join(items)}")
        except Exception:
            pass

    results = getattr(outcome, "tool_results", None) or []
    # 是否发生过检索（两条路径都可能：模型主动调工具，或引擎判定为先检索）
    searched = bool(debug.get("search") or debug.get("academic"))
    if results:
        for i, item in enumerate(results, 1):
            mark = "成功" if item.get("ok") else "失败"
            print(f"  ① 模型决定调用  {item.get('name')}")
            print(f"  ② 执行结果      {mark}：{str(item.get('message'))[:90]}")
        print(f"  ③ 二次生成      把工具结果作为 role=tool 回灌，再问一次模型")
    elif searched:
        # 这条路径容易被误解，必须说清楚：模型**没有**主动调工具，
        # 是引擎判定"这是时效性问题"之后自己先查了，再把结果注入上下文。
        # 原因是角色扮演模式下很多模型不会去调工具（见 _maybe_search 的注释）。
        print("  检索路径      引擎判定为先检索（模型未主动调工具）")
        print("                结果作为上下文注入提示词，不走 role=tool 回灌")
    else:
        print("  工具      本轮未触发工具调用，也未判定为需要检索")

    if debug.get("search"):
        s = debug["search"]
        print(f"  检索      provider={s.get('provider')} ok={s.get('ok')} "
              f"条数={s.get('count')} 耗时={s.get('elapsed')}")
    if debug.get("search_skipped"):
        print(f"  检索被拦  {debug.get('search_skipped')}（群级开关或额度门禁）")
    if debug.get("academic"):
        a = debug["academic"]
        print(f"  学术      provider={a.get('provider')} ok={a.get('ok')} "
              f"条数={a.get('count')}")
    if getattr(outcome, "memory_saved", None):
        print(f"  记忆落库  {len(outcome.memory_saved)} 条")
    if getattr(outcome, "memory_rejected", None):
        print(f"  护栏拦截  memory_rejected: {len(outcome.memory_rejected)} 条"
              f"（敏感信息按规则不保存）")


def _print_outcome(outcome, *, show_prompt: bool = False, trace: bool = False,
                   config=None) -> None:
    if outcome.reply:
        print(f"爱丽丝 > {outcome.reply}")
    else:
        print(f"[不回复] decision={outcome.decision} reason={outcome.reason}")
    notes = []
    if outcome.memory_saved:
        notes.append("新记忆: " + "、".join(outcome.memory_saved))
    if outcome.memory_rejected:
        notes.append("拒绝保存: " + "、".join(outcome.memory_rejected))
    if outcome.tool_results:
        notes.append("工具: " + ", ".join(f"{t['name']}={'ok' if t['ok'] else 'fail'}" for t in outcome.tool_results))
    if notes:
        print("[*] " + " | ".join(notes))
    if trace:
        print_trace(outcome, config)
    if show_prompt and outcome.prompt_system:
        print("\n----- SYSTEM PROMPT -----")
        print(outcome.prompt_system)
        print("----- END PROMPT -----\n")


def run_once(args) -> int:
    engine = build_engine(args)
    message = IncomingMessage(
        user_id=str(args.user), text=args.text, is_private=args.group in (None, "", 0),
        group_id="" if args.group is None else str(args.group), user_name=args.name,
        mentioned=args.mention, force_reply=args.force,
    )
    outcome = engine.handle(message)
    _print_outcome(outcome, show_prompt=args.show_prompt, trace=args.trace,
                    config=engine.config)
    if not outcome.reply:
        return 1
    return 0


def run_repl(args) -> int:
    engine = build_engine(args)
    user_id = str(args.user)
    group_id = "" if args.group is None else str(args.group)
    is_private = args.group is None
    scene = "私聊" if is_private else f"群 {group_id}"
    mode_note = "真实模型" if args.live else "离线 mock"
    print(f"爱丽丝模拟器（场景：{scene}，用户：{user_id} / {args.name}，模型：{mode_note}）")
    print("输入消息开始聊天；:mode 查看模式，:prompt 显示提示词，:quit 退出。\n")
    while True:
        try:
            raw = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw in (":quit", ":exit", "quit", "exit"):
            return 0
        if raw == ":mode":
            profile = engine.storage.ensure_profile(user_id, args.name, engine.config.bot.default_mode)
            print(f"[模式] {profile.mode}；记忆 {engine.storage.count_memory(user_id)} 条")
            continue
        show_prompt = raw == ":prompt"
        if show_prompt:
            raw = input("（这次要发给爱丽丝的内容）你 > ").strip()
            if not raw:
                continue
        mentioned = args.mention or (not is_private and engine.config.bot.character in raw)
        outcome = engine.handle(IncomingMessage(
            user_id=user_id, text=raw, is_private=is_private, group_id=group_id,
            user_name=args.name, mentioned=mentioned, force_reply=args.force,
        ))
        _print_outcome(outcome, show_prompt=show_prompt, trace=args.trace,
                        config=engine.config)


DEMO_SCRIPT = [
    ({"user_id": "10001", "user_name": "小星", "text": "你好，爱丽丝", "is_private": True}, None),
    ({"user_id": "10001", "user_name": "小星", "text": "记住我喜欢玩明日方舟，也喜欢猫", "is_private": True}, "记忆提取"),
    ({"user_id": "10001", "user_name": "小星", "text": "/记忆", "is_private": True}, "查看记忆"),
    ({"user_id": "10001", "user_name": "小星", "text": "我最近在打一个很难的关卡", "is_private": True}, "多轮对话"),
    ({"user_id": "10001", "user_name": "小星", "text": "/助手", "is_private": True}, "切换模式"),
    ({"user_id": "10001", "user_name": "小星", "text": "帮我解释一下什么是递归", "is_private": True}, "助手模式"),
    ({"user_id": "10002", "user_name": "路人甲", "text": "/记忆", "is_private": True}, "验证记忆隔离"),
    ({"user_id": "10003", "user_name": "群友", "text": "今天天气不错", "is_private": False,
      "group_id": "88888", "mentioned": False}, "群聊未触发"),
    ({"user_id": "10003", "user_name": "群友", "text": "爱丽丝在吗，推荐个游戏", "is_private": False,
      "group_id": "88888", "mentioned": True}, "群聊被@"),
]


def run_demo(args) -> int:
    args.reset = True
    engine = build_engine(args)
    engine.config.group.access = "open"
    print("=== 爱丽丝 V1 自检演示 ===\n")
    for index, (payload, label) in enumerate(DEMO_SCRIPT, 1):
        print(f"--- [{index}] {label} ---")
        print(f"用户 > {payload['text']}")
        engine.storage.upsert_group(payload.get("group_id", "88888") or "88888", enabled=True,
                                    extra={"approved": True})
        outcome = engine.handle(IncomingMessage(**payload))
        _print_outcome(outcome)
        print()
    print("=== 统计 ===")
    print(engine.storage.stats())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("aliceqq", description="爱丽丝 QQ 伙伴 Agent V1 本地模拟器")
    parser.add_argument("text", nargs="?", help="要发送的消息")
    parser.add_argument("--user", default="10001", help="用户 QQ 号")
    parser.add_argument("--name", default="测试用户", help="用户昵称")
    parser.add_argument("--group", default=None, help="群号；不填表示私聊")
    parser.add_argument("--mention", action="store_true", help="模拟被 @")
    parser.add_argument("--force", action="store_true", help="强制回复（跳过群聊触发判定）")
    parser.add_argument("--repl", action="store_true", help="进入交互模式")
    parser.add_argument("--demo", action="store_true", help="跑一遍自检演示")
    parser.add_argument("--reset", action="store_true", help="清空数据库后运行")
    parser.add_argument("--config", default=None, help="指定配置文件")
    parser.add_argument("--show-prompt", action="store_true", help="打印最终注入的系统提示词")
    parser.add_argument("--check", action="store_true", help="检查模板变量是否齐全")
    parser.add_argument("--check-voice", action="store_true",
                        help="自检语音链路：参考音频、克隆服务、情感与混合配置")
    parser.add_argument("--test-plan", action="store_true",
                        help="环境自检 + 离线自测 + 真实 QQ 人工测试清单")
    parser.add_argument("--simulate", action="store_true",
                        help="用真实模型跑 12 条端到端场景（不装 NoneBot 也能测）")
    parser.add_argument("--key", default=None, help="API Key；建议改用 .env 的 LLM_API_KEY")
    parser.add_argument("--live", action="store_true",
                        help="用 .env 里配置的真实模型（默认强制离线 mock）")
    parser.add_argument("--mock", action="store_true", help="配合 --simulate 使用离线 mock")
    parser.add_argument("--probe", action="store_true", help="探测哪个模型服务商可用（需要联网）")
    parser.add_argument("--evaluate", action="store_true", help="用真实模型跑一轮角色质量评估（需要联网）")
    parser.add_argument("--base", default=None, help="配合 --evaluate 指定 base_url")
    parser.add_argument("--model", default=None, help="配合 --evaluate 指定模型名")
    parser.add_argument("--json", default=None, help="把评估结果写入指定 json 文件")
    # 一键配置与启动：让"拿到项目就能跑"变成两条命令
    parser.add_argument("--trace", action="store_true",
                        help="打印决策链：模型、工具调用、检索来源、护栏拦截（Agent 调试用）")
    parser.add_argument("--setup", action="store_true",
                        help="一键配置：检查环境 → 装依赖 → 生成 .env → 自测（可重复执行）")
    parser.add_argument("--serve", action="store_true",
                        help="启动 QQ 机器人（需要 NoneBot2 + NapCat，见 --setup 的提示）")
    args = parser.parse_args(argv)

    if args.setup:
        from .bootstrap import main as bootstrap_main
        return bootstrap_main()
    if args.serve:
        return run_serve()
    if args.test_plan:
        from .testplan import print_plan
        return 0 if print_plan() else 1
    if args.simulate:
        return run_simulate(args)
    if args.check:
        return run_check(args)
    if args.check_voice:
        from .voicecheck import check_voice_setup
        config = AppConfig.load(args.config)
        print(check_voice_setup(config))
        return 0
    if args.probe:
        return run_probe(args)
    if args.evaluate:
        return run_evaluate(args)
    if args.demo:
        return run_demo(args)
    if args.repl:
        return run_repl(args)
    if not args.text:
        parser.print_help()
        return 2
    return run_once(args)


def run_simulate(args) -> int:
    import json as jsonlib
    from .simulate import run_simulation
    config = AppConfig.load(args.config)
    if args.base:
        config.llm.base_url = args.base.rstrip("/")
    if args.model:
        config.llm.model = args.model
    if args.key:
        config.llm.api_key = args.key
    if not args.mock:
        if not config.llm.api_key:
            print("没有 API Key：请在 .env 里设置 LLM_API_KEY。")
            print("想离线跑流程请加 --mock。")
            return 2
        config.llm.provider = "openai"
    else:
        config.llm.provider = "mock"
        config.llm.api_key = ""
    print(f"模拟模式：{config.llm.provider} / {config.llm.model}\n")
    summary = run_simulation(config)
    if args.json:
        Path(args.json).write_text(jsonlib.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")
    return 0 if summary["answered"] else 1


def run_probe(args) -> int:
    from .evaluate import probe_all
    config = AppConfig.load(args.config)
    api_key = config.llm.api_key or args.key
    if not api_key:
        print("没有 API Key：请在 .env 里设置 LLM_API_KEY，或用 --key 传入。")
        return 2
    print("逐个探测常见的 OpenAI 兼容服务商（每个一次极小请求）\n")
    usable = []
    for result in probe_all(api_key):
        mark = "可用" if result.ok else "不可用"
        print(f"[{mark}] {result.base_url}\n        模型 {result.model}：{result.detail}")
        if result.ok:
            usable.append(result)
    print()
    if not usable:
        print("没有探测到可用服务商。可能是：网络不通 / Key 无效 / 服务商不在候选列表里。")
        print("如果是自建或其他服务商，请直接指定：python -m aliceqq --evaluate --base <URL> --model <模型名>")
        return 1
    print("可用的服务商：")
    for result in usable:
        print(f"  --base {result.base_url} --model {result.model}")
    print("\n把选中的 base_url 与模型名写入 .env 的 LLM_BASE_URL / LLM_MODEL，然后运行 --evaluate。")
    return 0


def run_evaluate(args) -> int:
    import json as jsonlib
    from .evaluate import run_evaluation
    config = AppConfig.load(args.config)
    if args.base:
        config.llm.base_url = args.base.rstrip("/")
    if args.model:
        config.llm.model = args.model
    api_key = config.llm.api_key or args.key
    if not api_key:
        print("没有 API Key：请在 .env 里设置 LLM_API_KEY，或用 --key 传入。")
        return 2
    config.llm.api_key = api_key
    print(f"使用服务商 {config.llm.base_url} / 模型 {config.llm.model}\n")
    try:
        summary = run_evaluation(config)
    except Exception as exc:
        print(f"评估中断：{exc}")
        return 1
    if args.json:
        Path(args.json).write_text(jsonlib.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")
    return 0 if summary["answered"] else 1


def run_serve() -> int:
    r"""启动 QQ 机器人：把 integrations/nonebot/run.py 拉起来。

    为什么要包一层：`--setup` 已经把所有前置检查做完了，
    用户不该再需要记住"cd integrations/nonebot && python run.py"。
    两条命令搞定：`--setup` 然后 `--serve`。

    启动前把**缺什么**说清楚，而不是让 NoneBot 抛一堆 import 错误：
      * 没装 NoneBot2 -> 直接告诉他跑 --setup
      * 没配 OWNER_IDS -> 提醒一下（不影响启动，但没人能执行管理指令）
    """
    runner = Path(PROJECT_ROOT) / "integrations" / "nonebot" / "run.py"
    if not runner.is_file():
        print(f"找不到启动脚本：{runner}")
        print("仓库似乎不完整，重新 clone 一次。")
        return 1

    try:
        import nonebot  # noqa: F401
    except ImportError:
        print("还没装 NoneBot2 —— 它是接真实 QQ 才需要的依赖。")
        print()
        print("先跑一键配置：")
        print("    python -m aliceqq --setup")
        print()
        print("（只想本地聊两句的话不需要它：python -m aliceqq --repl）")
        return 2

    config = AppConfig.load()
    if not config.bot.owner_ids:
        print("提醒：.env 里没配 OWNER_IDS，没人能执行 /授权、/资料 这类管理指令。")
        print()

    print(f"启动中：{runner}")
    print("NapCat 的反向 WebSocket 请指向 ws://127.0.0.1:8095/onebot/v11/ws")
    print("（链路不通时另开一个窗口跑：python scripts\\链路体检.py）")
    print()
    # 用同一个解释器拉起，保证依赖一致
    proc = subprocess.run([sys.executable, str(runner)],
                          cwd=str(runner.parent),
                          env=dict(os.environ,
                                   ALICE_PROJECT_ROOT=str(PROJECT_ROOT),
                                   PYTHONUTF8="1",
                                   PYTHONIOENCODING="utf-8"))
    return proc.returncode


def run_check(args) -> int:
    from . import prompt as prompt_mod
    engine = build_engine(args)
    ok = True
    from .characters import CHARACTERS
    # 跟着角色注册表走：新增角色时不会漏检（漏检过 whale 一次）
    for character in CHARACTERS.values():
        name = character.template
        template = engine.template(name)
        variables = engine.build_context(IncomingMessage(user_id=str(args.user), text="检查",
                                                        user_name=args.name), character.id)
        report = prompt_mod.list_variables(template, variables)
        print(f"[{name}] 使用 {len(report['used'])} 个变量；缺失：{report['missing'] or '无'}")
        if report["missing"]:
            ok = False
    print("模板检查：" + ("通过" if ok else "存在缺失变量"))
    return 0 if ok else 1
