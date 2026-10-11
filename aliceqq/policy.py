"""访问控制、频率限制与额度：决定"这条消息该不该回"。

把判断结果做成结构化 Reason，便于测试和排障，也方便日志里说明为什么没回。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .config import AppConfig
from .db import Storage


class Decision(str, Enum):
    ALLOW = "allow"
    SILENT = "silent"            # 正常忽略（如群里没被叫到）
    DENIED = "denied"            # 明确拒绝（如该群未授权）
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    DISABLED = "disabled"


@dataclass
class Verdict:
    decision: Decision
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def __bool__(self) -> bool:
        return self.allowed


def _period_minute(now: Optional[float] = None) -> str:
    return time.strftime("%Y%m%d%H%M", time.localtime(time.time() if now is None else now))


def _period_day(now: Optional[float] = None) -> str:
    return time.strftime("%Y%m%d", time.localtime(time.time() if now is None else now))


def check_group_access(storage: Storage, config: AppConfig, group_id: str) -> Verdict:
    """受控公开：access=open 时全部放行；approved_groups 时必须在白名单内；
    controlled 时必须在白名单内且该群 enabled。"""
    policy = config.group
    if not policy.enabled:
        return Verdict(Decision.DISABLED, "群聊功能未启用")
    record = storage.get_group(group_id)
    if policy.access == "open":
        if record is not None and not record.enabled:
            return Verdict(Decision.DENIED, "该群已被单独关闭")
        return Verdict(Decision.ALLOW, "open 模式放行")
    if record is None:
        return Verdict(Decision.DENIED, "该群未在授权列表中", {"group_id": group_id, "access": policy.access})
    if not record.enabled:
        return Verdict(Decision.DENIED, "该群已被关闭")
    if policy.access == "controlled" and not bool(record.extra.get("approved", False)):
        return Verdict(Decision.DENIED, "controlled 模式下该群未标记为已批准")
    return Verdict(Decision.ALLOW, "群已授权")


def check_rate_limit(storage: Storage, scope: str, scope_id: str, limit: int,
                     now: Optional[float] = None) -> Verdict:
    """按分钟计数；limit<=0 视为不限。"""
    if limit <= 0:
        return Verdict(Decision.ALLOW, "未启用频率限制")
    period = _period_minute(now)
    used = storage.get_counter(scope, scope_id, period)
    if used >= limit:
        return Verdict(Decision.RATE_LIMITED, f"{scope} 每分钟上限 {limit} 次已用尽", {"used": used, "limit": limit})
    return Verdict(Decision.ALLOW, "频率正常", {"used": used, "limit": limit})


def consume_rate_limit(storage: Storage, scope: str, scope_id: str, now: Optional[float] = None) -> int:
    return storage.incr_counter(scope, scope_id, _period_minute(now))


def check_quota(storage: Storage, config: AppConfig, user_id: str, is_private: bool,
                now: Optional[float] = None, group_cap: Optional[int] = None) -> Verdict:
    if not config.quota.enabled:
        return Verdict(Decision.ALLOW, "额度系统未启用")
    period = _period_day(now)
    scope = "quota_private" if is_private else "quota_group"
    # group_cap：群级额度覆盖（/群设置 群额度）。None 表示跟随全局配置。
    if is_private:
        cap = config.quota.daily_private
    else:
        cap = config.quota.daily_group if group_cap is None else int(group_cap)
    used = storage.get_counter(scope, user_id, period)
    if cap > 0 and used >= cap:
        return Verdict(Decision.QUOTA_EXCEEDED, f"今日{'私聊' if is_private else '群聊'}额度已用尽", {"used": used, "cap": cap})
    global_used = storage.get_counter("quota_global", "all", period)
    if config.quota.daily_global > 0 and global_used >= config.quota.daily_global:
        return Verdict(Decision.QUOTA_EXCEEDED, "今日全站额度已用尽", {"used": global_used, "cap": config.quota.daily_global})
    return Verdict(Decision.ALLOW, "额度充足", {"used": used, "cap": cap, "global_used": global_used})


def consume_quota(storage: Storage, config: AppConfig, user_id: str, is_private: bool,
                  now: Optional[float] = None) -> None:
    if not config.quota.enabled:
        return
    period = _period_day(now)
    storage.incr_counter("quota_private" if is_private else "quota_group", user_id, period)
    storage.incr_counter("quota_global", "all", period)


# 高成本能力的额度维度：特征 -> (私聊计数域, 群聊计数域, 全局计数域)
FEATURE_QUOTA_SCOPES = {
    "search": ("quota_search_user", "quota_search_group", "quota_search_global"),
    "image": ("quota_image_user", "quota_image_group", "quota_image_global"),
    "voice": ("quota_voice_user", "quota_voice_group", "quota_voice_global"),
    # 语音识别：按时长计费，单独计额度
    "asr": ("quota_asr_user", "quota_asr_group", "quota_asr_global"),
    # 图片理解：**按张计费**，和"生成图片"是两件事，所以单独一项。
    # 群聊里连发多图很常见，不独立计额度的话，一次刷图就会把
    # 其它能力的额度一起吃掉。
    "vision": ("quota_vision_user", "quota_vision_group", "quota_vision_global"),
    # 语音理解（用户发来的语音转文字）。
    # **和 "asr" 分开**：那个是"转写视频音轨"，一段几分钟、成本高；
    # 这个是"群友随口说一句"，通常十几秒。用量级也完全不同
    # （前者一天几次，后者可能几十次），共用额度会导致
    # 转了两段视频就把语音对话的额度吃光。
    "asr_voice": ("quota_asrvoice_user", "quota_asrvoice_group", "quota_asrvoice_global"),
}

FEATURE_LABELS = {"search": "联网检索", "image": "图片", "voice": "语音",
                  "asr": "语音识别", "vision": "图片理解", "asr_voice": "语音理解"}


def check_asr_quota(storage: Storage, config: AppConfig, now: Optional[float] = None) -> Verdict:
    """语音识别的每日总量检查（按时长计费的能力，用总数而不是按人计）。"""
    settings = getattr(config, "asr", None)
    if settings is None or not settings.enabled:
        return Verdict(Decision.DISABLED, "未启用语音识别")
    if not config.quota.enabled or settings.daily_calls <= 0:
        return Verdict(Decision.ALLOW, "未限制识别次数")
    used = storage.get_counter("quota_asr_daily", "all", _period_day(now))
    if used >= settings.daily_calls:
        return Verdict(Decision.QUOTA_EXCEEDED,
                       f"今日语音识别次数已用尽（{used}/{settings.daily_calls}）",
                       {"used": used, "cap": settings.daily_calls})
    return Verdict(Decision.ALLOW, "额度充足", {"used": used, "cap": settings.daily_calls})


def consume_asr_quota(storage: Storage, now: Optional[float] = None) -> None:
    storage.incr_counter("quota_asr_daily", "all", _period_day(now))


def feature_quota_cap(config: AppConfig, feature: str, is_private: bool) -> int:
    quota = config.quota
    if feature == "search":
        return quota.daily_search_private if is_private else quota.daily_search_group
    if feature == "image":
        return quota.daily_image
    if feature == "voice":
        return quota.daily_voice
    if feature == "vision":
        return quota.daily_vision
    if feature == "asr_voice":
        return quota.daily_asr_voice
    return 0


def check_feature_quota(storage: Storage, config: AppConfig, feature: str, user_id: str,
                        group_id: str = "", is_private: bool = True,
                        now: Optional[float] = None) -> Verdict:
    """检查某一项高成本能力的额度。

    V3 要求"语音、图片和联网搜索可以分别使用独立额度"，且
    "群额度和个人额度必须分别判断"——所以按 特征 × 场景 分别计数。
    """
    if not config.quota.enabled or feature not in FEATURE_QUOTA_SCOPES:
        return Verdict(Decision.ALLOW, "额度系统未启用")
    user_scope, group_scope, global_scope = FEATURE_QUOTA_SCOPES[feature]
    period = _period_day(now)
    cap = feature_quota_cap(config, feature, is_private)
    scope = group_scope if not is_private else user_scope
    scope_id = str(group_id) if not is_private else str(user_id)
    used = storage.get_counter(scope, scope_id, period)
    label = FEATURE_LABELS.get(feature, feature)
    if cap > 0 and used >= cap:
        who = "本群" if not is_private else "你"
        return Verdict(Decision.QUOTA_EXCEEDED, f"{who}今日的{label}额度已用尽",
                       {"used": used, "cap": cap, "feature": feature})
    global_cap = config.quota.daily_search_global if feature == "search" else 0
    if global_cap > 0:
        global_used = storage.get_counter(global_scope, "all", period)
        if global_used >= global_cap:
            return Verdict(Decision.QUOTA_EXCEEDED, f"今日全站的{label}额度已用尽",
                           {"used": global_used, "cap": global_cap, "feature": feature})
    return Verdict(Decision.ALLOW, "额度充足", {"used": used, "cap": cap, "feature": feature})


def consume_feature_quota(storage: Storage, config: AppConfig, feature: str, user_id: str,
                          group_id: str = "", is_private: bool = True,
                          now: Optional[float] = None) -> None:
    if not config.quota.enabled or feature not in FEATURE_QUOTA_SCOPES:
        return
    user_scope, group_scope, global_scope = FEATURE_QUOTA_SCOPES[feature]
    period = _period_day(now)
    if is_private:
        storage.incr_counter(user_scope, str(user_id), period)
    else:
        storage.incr_counter(group_scope, str(group_id), period)
    storage.incr_counter(global_scope, "all", period)


def feature_quota_text(storage: Storage, config: AppConfig, user_id: str, group_id: str = "",
                       is_private: bool = True) -> str:
    if not config.quota.enabled:
        return "功能额度：未启用"
    period = _period_day()
    user_scope, group_scope, _ = FEATURE_QUOTA_SCOPES["search"]
    scope, scope_id = (user_scope, str(user_id)) if is_private else (group_scope, str(group_id))
    cap = feature_quota_cap(config, "search", is_private)
    used = storage.get_counter(scope, scope_id, period)
    who = "你" if is_private else "本群"
    return f"联网检索：{who} {used}/{cap}"


def quota_status_text(storage: Storage, config: AppConfig, user_id: str, is_private: bool) -> str:
    if not config.quota.enabled:
        return "额度系统：未启用"
    period = _period_day()
    scope = "quota_private" if is_private else "quota_group"
    cap = config.quota.daily_private if is_private else config.quota.daily_group
    used = storage.get_counter(scope, user_id, period)
    return f"今日{'私聊' if is_private else '群聊'}额度：{used}/{cap}"


def permission_context_text(storage: Storage, config: AppConfig, user_id: str,
                            is_private: bool, group_id: str = "") -> str:
    if is_private:
        return "当前为私聊场景，只能使用该用户自己的记忆，禁止引用其他用户的任何信息。"
    parts = ["当前为群聊场景：不得暴露任何用户的私聊内容或他人隐私。"]
    record = storage.get_group(group_id) if group_id else None
    if record is not None:
        parts.append(f"本群授权状态：{'已启用' if record.enabled else '已停用'}；模式覆盖：{record.mode or '跟随用户'}。")
    if str(user_id) in config.bot.owner_ids:
        parts.append("当前用户是机器人所有者，可执行资料维护类操作。")
    return " ".join(parts)
