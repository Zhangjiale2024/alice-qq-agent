"""群聊触发判定：什么时候可以在群里说话。

对应文档"五、QQ 场景"：被 @、命中关键词、收到明确指令、或正在参与的话题才回复。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import AppConfig
from .db import GroupConfig


@dataclass
class TriggerResult:
    should_reply: bool
    matched: List[str] = field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:
        return self.should_reply


@dataclass
class TriggerContext:
    """触发判定需要的最小上下文。"""
    text: str
    mentioned: bool = False
    is_command: bool = False
    replied_to_bot: bool = False
    seconds_since_bot_reply: Optional[float] = None
    seconds_since_own_message: Optional[float] = None
    # 当前说话者是否就是机器人上一次在群里回复的那个人（决定能否算"话题延续"）
    recent_topic_with_me: bool = False
    force: bool = False


def keyword_hit(text: str, keywords: Sequence[str]) -> List[str]:
    low = (text or "").lower()
    hits = []
    for keyword in keywords:
        if not keyword:
            continue
        if keyword.lower() in low:
            hits.append(keyword)
    return hits


def evaluate(config: AppConfig, ctx: TriggerContext, group: Optional[GroupConfig] = None,
             extra_keywords: Optional[Sequence[str]] = None) -> TriggerResult:
    """按配置的 GROUP_TRIGGER 顺序判定是否回复。

    extra_keywords：群级配置里的额外关键词（来自 /群设置 关键词）。
    **追加而不是替换**——替换语义很容易让管理员一不小心把"爱丽丝"本身关掉。
    """
    policy = config.group
    triggers = list(policy.triggers or [])
    keywords: List[str] = list(policy.keywords)
    if group is not None and group.keywords:
        keywords.extend(k for k in group.keywords if k not in keywords)
    for word in (extra_keywords or ()):
        text = str(word).strip()
        if text and text not in keywords:
            keywords.append(text)

    if ctx.force:
        return TriggerResult(True, ["force"], "被显式要求回复")

    matched: List[str] = []
    reasons: List[str] = []

    if "mention" in triggers and ctx.mentioned:
        matched.append("mention")
        reasons.append("被 @")
    if "command" in triggers and ctx.is_command:
        matched.append("command")
        reasons.append("明确指令")
    if "reply" in triggers and ctx.replied_to_bot:
        matched.append("reply")
        reasons.append("回复了机器人")
    hits = keyword_hit(ctx.text, keywords) if "keyword" in triggers else []
    if hits:
        matched.append("keyword")
        reasons.append("命中关键词 " + "/".join(hits[:3]))
    # 话题延续：只在"同一个人接着往下说"时成立。
    # 早期实现只看时间窗口，导致机器人在群里说过一句话之后的 90 秒内，
    # 任何人说话都会被当成延续话题并触发回复。
    if "relevant_topic" in triggers and ctx.recent_topic_with_me:
        if (ctx.seconds_since_bot_reply is not None
                and ctx.seconds_since_bot_reply <= policy.continuity_seconds):
            matched.append("relevant_topic")
            reasons.append("延续与机器人的对话")

    if not matched:
        return TriggerResult(False, [], "未命中任何触发条件")

    # 避免同一用户连续刷屏：他刚说完还没过冷却，就不插话。
    # 注意把 relevant_topic 也算进来——早期它绕过了这个判定，导致言无不尽地连回。
    if (ctx.seconds_since_own_message is not None and ctx.seconds_since_own_message < policy.cooldown_seconds
            and "mention" not in matched and "command" not in matched and not ctx.replied_to_bot):
        return TriggerResult(False, matched, "同一用户发言过于密集，暂不响应")

    return TriggerResult(True, matched, "；".join(reasons))


def strip_mention(text: str, bot_names: Sequence[str] = ()) -> str:
    """去掉 @机器人 / CQ 码 / 开头的称呼，留下真正的正文。"""
    cleaned = re.sub(r"\[CQ:at,[^\]]*\]", " ", text or "")
    for name in bot_names:
        if not name:
            continue
        cleaned = cleaned.replace(f"@{name}", " ")
    return re.sub(r"\s+", " ", cleaned).strip()
