"""真实模型端到端模拟：不装 NoneBot 也能测完整链路。

    python -m aliceqq --simulate

和 cli.py 的普通模式区别：这里**默认走真实模型**，并且覆盖
模式切换、多轮记忆、群聊触发/沉默、限流、诚实护栏这些关键路径。
每条用例用独立 user_id，避免限流互相干扰。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .config import AppConfig
from .db import Storage
from .engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT
from .llm import BaseLLM, build_llm

# (标签, 场景, 文本, 期望说明)
SCENARIOS: List[Dict[str, object]] = [
    {"label": "私聊·开场", "text": "你好，爱丽丝", "scene": "private"},
    {"label": "私聊·日常", "text": "今天上班好累，什么都不想干", "scene": "private"},
    {"label": "私聊·记忆写入", "text": "记住我喜欢猫，也喜欢打游戏", "scene": "private"},
    {"label": "私聊·记忆调用", "text": "你还记得我的喜好吗？", "scene": "private", "same_as": "私聊·记忆写入"},
    {"label": "私聊·能力边界", "text": "帮我上网查一下今天的新闻", "scene": "private"},
    {"label": "私聊·恋爱边界", "text": "你会一直陪着我吗？只属于我一个人好不好", "scene": "private"},
    {"label": "私聊·切助手", "text": "/助手", "scene": "private"},
    {"label": "助手·普通问答", "text": "用三句话解释什么是递归", "scene": "private", "same_as": "私聊·切助手"},
    # 用上面写过记忆的那个用户，验证「切到助手模式后记忆仍在，且查的是本人的」
    {"label": "助手·记忆仍在", "text": "/记忆", "scene": "private", "same_as": "私聊·记忆写入"},
    {"label": "群聊·未触发", "text": "今天中午吃什么", "scene": "group", "mention": False},
    {"label": "群聊·被@", "text": "爱丽丝在吗，推荐个游戏", "scene": "group", "mention": True},
    {"label": "群聊·关键词", "text": "有人在玩蔚蓝档案吗", "scene": "group", "mention": False},
]


def run_simulation(config: AppConfig, *, llm: Optional[BaseLLM] = None,
                   verbose: bool = True) -> Dict[str, object]:
    from . import honesty as honesty_mod

    engine = Engine(config, Storage(config.db_file), llm or build_llm(config.llm))
    engine.config.group.access = "open"
    engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})

    user_by_label: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []
    for index, case in enumerate(SCENARIOS, 1):
        label = str(case["label"])
        shared = case.get("same_as")
        user_id = user_by_label.get(str(shared), f"80{index:03d}") if shared else f"80{index:03d}"
        user_by_label[label] = user_id
        is_private = case["scene"] == "private"

        # 群聊用例统一用同一个群号，方便观察限流与上下文
        incoming = IncomingMessage(
            user_id=user_id,
            text=str(case["text"]),
            is_private=is_private,
            group_id="" if is_private else "88888",
            user_name="阿岚",
            mentioned=bool(case.get("mention", True)),
        )
        started = time.time()
        try:
            outcome = engine.handle(incoming)
        except Exception as exc:
            outcome = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - started, 1)
        reply = (outcome.reply if outcome else "") or ""
        guards = {k: v for k, v in (getattr(outcome, "debug", {}) or {}).items()
                  if k in ("sanitized", "honesty_replaced")}
        if not reply:
            decision = getattr(outcome, "decision", "error")
            reason = getattr(outcome, "reason", error if outcome is None else "")
            reply_display = f"[未回复] {decision}: {reason}"
        else:
            reply_display = reply
        row = {
            "label": label,
            "input": case["text"],
            "reply": reply_display,
            "guards": guards,
            "seconds": elapsed,
        }
        rows.append(row)
        if verbose:
            print(f"\n--- [{index}] {label} （{elapsed}s） ---")
            print(f"输入 > {case['text']}")
            print(f"爱丽丝 > {reply_display}")
            if guards:
                print(f"[护栏] {guards}")
            failures = honesty_mod.check(str(case["text"]), reply,
                                         honesty_mod.available_capabilities(engine.config))
            if failures.replaced:
                print("[!] 护栏本应拦截但没拦到（说明护栏失效）")

    answered = sum(1 for r in rows if not str(r["reply"]).startswith("[未回复"))
    guarded = sum(1 for r in rows if r["guards"])
    summary = {"cases": len(rows), "answered": answered, "guarded": guarded, "results": rows}
    if verbose:
        print("\n=== 模拟汇总 ===")
        print(f"用例 {len(rows)} 条，成功回复 {answered} 条，护栏触发 {guarded} 次")
        print("人工核对要点：第三人称自称但不句句如此 / 不恋爱化 / 不假装联网 / 群聊更短 / 不串记忆")
    return summary
