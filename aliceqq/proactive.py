"""主动关心：低频、可预测、可关闭。

对应配置 PROACTIVE_PRIVATE / PROACTIVE_GROUP / PROACTIVE_FREQUENCY。
设计上刻意保守：
- 默认只在私聊；群里默认不主动（避免打扰所有人）。
- 有安静时段、每日上限、最短闲置时长三重约束。
- 只挑模板，不调用模型，避免"突然说一句奇怪的话"。
- 同一个人连续多天没互动也只会每天最多一条。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .config import AppConfig
from .db import Storage


@dataclass
class ProactiveTask:
    user_id: str
    text: str
    reason: str
    # V4：群聊主动话题时填群号，user_id 留空
    group_id: str = ""

    @property
    def is_group(self) -> bool:
        return bool(self.group_id)


FREQUENCY_PROFILE = {
    # frequency -> (最短闲置分钟数倍率, 每日上限倍率)
    "off": (0, 0),
    "low": (1.0, 1.0),
    "normal": (0.5, 2.0),
}


def _period_day(now: Optional[float] = None) -> str:
    return time.strftime("%Y%m%d", time.localtime(time.time() if now is None else now))


def in_quiet_hours(config: AppConfig, now: Optional[float] = None) -> bool:
    hour = time.localtime(time.time() if now is None else now).tm_hour
    return hour in set(config.proactive.quiet_hours or [])


def _render(template: str, name: str, hours: float) -> str:
    return (template.replace("{name}", name)
                    .replace("{hours}", str(max(1, int(hours))))
                    .replace("{minutes}", str(int(hours * 60))))


def is_reachable_user(user_id: str) -> bool:
    """这个 user_id 看起来是不是**真实可发送**的 QQ 号。

    为什么必须查：实测（2026-10-10 日志）主动关心每隔一轮就给
    `10001` / `10002` 发送，QQ 侧一律回：

        ActionFailed retcode=1200 无法获取用户信息

    这两个号是 README / `.env.example` 里的**示例用户**——开发时用命令行
    `--user 10001` 试过，于是档案进了库、`message_count > 0`，
    主动关心就把它当成"真实用户"反复重试，日志被刷、消息永远发不出去。

    QQ 号是纯数字。**长度门槛设成 6 位以上**：实测踩到的是
    `10001` / `10002`——它们是 README 和 `.env.example` 里的示例号，
    开发时用 `--user 10001` 试过，于是进了档案库并被反复重试。
    真实 QQ 号现在最短也有 6 位以上，5 位号是很久以前的靓号段。
    """
    text = str(user_id or "").strip()
    if not text.isdigit():
        return False
    return 6 <= len(text) <= 12


def _in_failure_backoff(storage: Storage, user_id: str, now: float) -> bool:
    """连续发送失败过的用户，在一段时间内不再尝试。

    没有这个退避，发送失败每轮都会重来一次（默认 15 分钟一轮，一天 96 次），
    而失败原因往往是**永久的**（对方不是有效用户、已把我们拉黑）。
    """
    last = storage.kv_get(f"proactive_fail:{user_id}", 0) or 0
    try:
        last = float(last)
    except (TypeError, ValueError):
        return False
    return (now - last) < FAILURE_BACKOFF_SECONDS


def mark_failed(storage: Storage, user_id: str, now: Optional[float] = None) -> None:
    """记一次发送失败，进入退避。"""
    storage.kv_set(f"proactive_fail:{user_id}",
                   time.time() if now is None else now)


# 失败后退避 24 小时：既不让日志被刷，也不至于永久放弃一个只是临时异常的用户
FAILURE_BACKOFF_SECONDS = 24 * 3600


def plan(config: AppConfig, storage: Storage, now: Optional[float] = None,
         limit: int = 5) -> List[ProactiveTask]:
    """挑出本轮应该主动关心的用户。返回空列表表示什么都不做。"""
    stamp = time.time() if now is None else now
    policy = config.proactive
    if not policy.enabled or not config.features.proactive_private:
        return []
    if policy.frequency == "off":
        return []
    if in_quiet_hours(config, stamp):
        return []

    idle_factor, cap_factor = FREQUENCY_PROFILE.get(policy.frequency, (1.0, 1.0))
    min_idle = policy.min_idle_minutes * idle_factor
    max_per_day = max(0, int(policy.max_per_user_per_day * cap_factor))
    if max_per_day <= 0:
        return []

    tasks: List[ProactiveTask] = []
    period = _period_day(stamp)
    for profile in storage.iterate_profiles():
        if len(tasks) >= limit:
            break
        if profile.message_count <= 0 or not profile.last_seen:
            continue
        # 只给**看起来是真实 QQ 号**的用户发。开发时用命令行造出来的
        # 示例档案（10001/10002 之类）会让 QQ 侧一直回 1200 无法获取用户信息，
        # 而且每轮重试——既发不出去又刷日志。
        if not is_reachable_user(profile.user_id):
            continue
        # 发过失败的用户退避 24 小时，不要每轮都撞一次
        if _in_failure_backoff(storage, profile.user_id, stamp):
            continue
        idle_minutes = (stamp - profile.last_seen) / 60.0
        if idle_minutes < min_idle:
            continue
        # 用户正在其他场景活跃时不打扰
        if storage.kv_get(f"bot_last_reply:last:{profile.user_id}"):
            continue
        sent_key = f"proactive_sent:{profile.user_id}"
        sent_today = int(storage.kv_get(f"{sent_key}:{period}", 0) or 0)
        if sent_today >= max_per_day:
            continue
        templates = list(policy.templates or [])
        if not templates:
            continue
        template = templates[profile.message_count % len(templates)]
        tasks.append(ProactiveTask(
            user_id=profile.user_id,
            text=_render(template, profile.preferred_name, idle_minutes / 60.0),
            reason=f"闲置 {int(idle_minutes)} 分钟",
        ))
    return tasks


def plan_group_topics(config: AppConfig, storage: Storage, now: Optional[float] = None,
                      limit: int = 2) -> List[ProactiveTask]:
    """挑出本轮可以主动发起话题的群（V4）。

    门槛刻意设得很高——群里主动说话打扰的是所有人：
    1. 全局开关 features.proactive_group 打开；
    2. 该群的 proactive 设置不是 False（群主可以关掉）；
    3. 群已授权且启用；
    4. 不在安静时段；
    5. 群内安静时长 >= group_min_idle_minutes；
    6. 每群每日上限。
    """
    stamp = time.time() if now is None else now
    policy = config.proactive
    if not policy.enabled or not config.features.proactive_group:
        return []
    if policy.frequency == "off":
        return []
    if in_quiet_hours(config, stamp):
        return []
    if not policy.group_templates:
        return []

    from .config import GroupSettings

    tasks: List[ProactiveTask] = []
    period = _period_day(stamp)
    for record in storage.list_groups():
        if len(tasks) >= limit:
            break
        if not record.enabled:
            continue
        settings = GroupSettings.from_extra(record.extra)
        # 群主没明确开启就不发言（None 表示跟随全局，这里要求显式允许）
        if settings.proactive is not True:
            continue
        last_reply = storage.kv_get(f"bot_last_reply:{record.group_id}", 0) or 0
        try:
            last_reply = float(last_reply)
        except (TypeError, ValueError):
            last_reply = 0.0
        idle_minutes = (stamp - last_reply) / 60.0 if last_reply else 1e9
        if idle_minutes < policy.group_min_idle_minutes:
            continue
        sent_key = f"proactive_group_sent:{record.group_id}:{period}"
        if int(storage.kv_get(sent_key, 0) or 0) >= policy.group_max_per_day:
            continue
        # 轮流用模板，避免每次都是同一句
        index = int(storage.kv_get(f"proactive_group_idx:{record.group_id}", 0) or 0)
        template = policy.group_templates[index % len(policy.group_templates)]
        tasks.append(ProactiveTask(
            user_id="", group_id=str(record.group_id), text=template,
            reason=f"群内安静 {int(idle_minutes)} 分钟"))
    return tasks


def mark_group_sent(storage: Storage, group_id: str, now: Optional[float] = None) -> None:
    stamp = time.time() if now is None else now
    period = _period_day(stamp)
    key = f"proactive_group_sent:{group_id}:{period}"
    storage.kv_set(key, int(storage.kv_get(key, 0) or 0) + 1)
    index = int(storage.kv_get(f"proactive_group_idx:{group_id}", 0) or 0)
    storage.kv_set(f"proactive_group_idx:{group_id}", index + 1)


def mark_sent(storage: Storage, user_id: str, now: Optional[float] = None) -> None:
    period = _period_day(now)
    key = f"proactive_sent:{user_id}:{period}"
    current = int(storage.kv_get(key, 0) or 0)
    storage.kv_set(key, current + 1)
    storage.kv_set(f"proactive_last:{user_id}", time.time() if now is None else now)
