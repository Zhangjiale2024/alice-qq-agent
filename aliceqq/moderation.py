"""V5 群管理：规则匹配与反广告判定。

**这个模块只做"判断"，不做"动作"**——执行层在 engine，因为动作要经过
权限检查和（高风险操作的）管理员确认。分开的好处是这层可以完全离线测试。

设计原则（对应 V5 第六节"默认安全策略"）：

1. **没有明确规则就不判定**。默认只有一组很窄的广告特征，
   宁漏不误伤——误伤正常群友的代价远大于漏掉一条广告。
2. **不确定就不动作**。命中低置信度特征只记录、提示管理员，不自动处罚。
3. **绝不因为批评机器人而处罚**。对机器人的负面评价、吐槽、争论
   一律不构成处罚理由（有专门的回归测试，V5 明确要求）。
4. **可解释**。每次判定都要能说清"命中了哪条规则、原话是什么"，
   否则管理员无法复核，也没法调规则。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 规则与判定

# 判定等级（confidence）：越高的越危险，越需要动作
CONF_LOW = "low"        # 只记录 / 提示管理员
CONF_MEDIUM = "medium"  # 可以自动警告
CONF_HIGH = "high"      # 可以进入处罚流程（仍需按群配置/确认）

# 动作类型
ACTION_NONE = "none"
ACTION_WARN = "warn"
ACTION_RECALL = "recall"
ACTION_MUTE = "mute"
ACTION_KICK = "kick"

# 高风险动作：默认需要管理员确认（V5 明确点名 禁言/踢出/批量撤回）
HIGH_RISK_ACTIONS = (ACTION_MUTE, ACTION_KICK, ACTION_RECALL)

ACTION_LABELS = {
    ACTION_WARN: "警告",
    ACTION_RECALL: "撤回消息",
    ACTION_MUTE: "禁言",
    ACTION_KICK: "移出群聊",
}


@dataclass
class ModerationVerdict:
    """一次判定的结果。hit=False 表示没命中任何规则，不要动作。"""
    hit: bool = False
    rule: str = ""                  # 规则标识，用于解释和审计
    label: str = ""                 # 人类可读的规则名
    confidence: str = CONF_LOW
    reason: str = ""                # 说明命中了什么（会写进审计日志）
    evidence: str = ""              # 触发判定的原文片段
    suggested: str = ACTION_NONE    # 建议动作（是否执行由上层决定）

    def explain(self) -> str:
        """给管理员看的解释。"""
        if not self.hit:
            return "没有命中任何规则"
        parts = [f"命中规则「{self.label}」（{self.confidence}）"]
        if self.evidence:
            parts.append(f"依据：{self.evidence}")
        return "；".join(parts)


@dataclass
class ModerationRules:
    """一个群的审核规则集（由群配置构造）。"""
    enabled: bool = True
    # 只记录不动作：新群默认先观察，避免一开就把人误伤
    dry_run: bool = True
    # 自动警告所需的最高容忍度：medium 以上才自动警告
    auto_warn: bool = True
    # 这些动作默认需要管理员确认
    confirm_actions: tuple = HIGH_RISK_ACTIONS
    # 达到几次警告后升级处罚（0 = 不自动升级）
    warn_threshold: int = 3
    # 管理员自定义的敏感词（命中即 medium）
    ban_words: List[str] = field(default_factory=list)
    # 白名单：这些用户/词永不判定（管理员自己人、常见正常词）
    allow_words: List[str] = field(default_factory=list)
    allow_users: List[str] = field(default_factory=list)
    # 重复刷屏：同一用户同样内容出现次数阈值
    repeat_threshold: int = 3
    repeat_window: int = 120        # 秒


# ---------------------------------------------------------------- 特征识别

# 广告特征1：卖号/代练/外挂/私服 这类词本身就高度指向广告
_PROMO_WORDS = (
    "代练", "代肝", "代抽", "卖号", "收号", "出号", "租号", "借号",
    "外挂", "脚本挂", "私服", "破解版", "辅助软件", "低价出",
    "加微信", "加vx", "加VX", "加QQ", "加qq", "私聊我", "点击链接",
    "扫码进群", "进群免费", "免费领取", "限时优惠", "充值优惠",
    "低价充值", "批发出售", "一手货源", "刷单", "兼职日结",
)

# 广告特征2b：主动导流到私聊（这类词本身就是"我要跟你私下交易"的信号）
_CONTACT_WORDS = (
    "私聊我", "私我", "私信我", "加我好友", "加我微信", "加我qq", "加我QQ",
    "加微信", "加vx", "加VX", "加qq", "加QQ", "联系我", "详聊", "戳我",
)

# 广告特征2：QQ 群号 / 微信号 + 邀请语气（单独出现群号不算，必须是邀请）
_INVITE_PATTERNS = (
    re.compile(r"(加|进|扫)\s*(群|q群|Q群|qq群)\s*[:：]?\s*\d{5,}"),
    re.compile(r"群号\s*[:：]?\s*\d{5,}"),
    re.compile(r"(微信|vx|wx|VX|WX)\s*[:：]?\s*[A-Za-z0-9_\-]{5,}"),
    re.compile(r"(https?://)?(www\.)?(taobao|tmall|jd|pdd)\.(com|cn)/\S*"),
)

# 广告特征3：URL + 促销语气
_URL_RE = re.compile(r"https?://\S+|www\.\S+\.\S+")
_PROMO_TONE = ("优惠", "折扣", "秒杀", "特价", "包邮", "免费送", "下单", "购买", "订购")

# 明确**不构成**处罚的信号：对机器人的评价、观点分歧、正常求助。
# V5 明确"不得因为用户批评机器人而处罚用户"，所以这些必须显式豁免。
_NEVER_PUNISH = (
    re.compile(r"(我觉得|感觉|其实|但是|不过)[^。！？\n]{0,12}(不对|不好|错了|难用)"),
    re.compile(r"(为什么|怎么|求助|请问)[^。！？\n]{0,20}"),
)

# 提到机器人本身 + 任何负面词 = 吐槽。
# 为什么拆成两个条件：实测把负面词写进一条正则会漏掉"这个机器人不太好用"
# 这类中间带修饰词的写法，而穷举修饰词必然漏。所以改成
# "主语是机器人" 且 "句子里有负面词"两个条件同时成立。
_BOT_SUBJECT = re.compile(r"(爱丽丝|机器人|这个bot|这bot|这个AI|人工智能)")
_NEGATIVE_WORDS = (
    "蠢", "笨", "傻", "垃圾", "没用", "不好用", "不太好用", "太难用", "好难用",
    "难用", "坏了", "有问题", "弱智", "拉胯", "离谱", "不行", "答非所问",
    "听不懂", "气人", "讨厌", "鸡肋", "差劲", "无语",
)


def _mentions_bot_with_negative(text: str) -> bool:
    if not _BOT_SUBJECT.search(text):
        return False
    return any(word in text for word in _NEGATIVE_WORDS)

# 明显是正常交流的词：出现这些就不按广告处理（降低误伤）
_NORMAL_CONTEXT = (
    "官方", "公告", "更新", "维护", "活动", "攻略", "怎么", "请问", "求助",
    "打不过", "卡关", "推荐", "角色", "抽卡", "剧情", "联机", "开黑",
)


def _find(text: str, words) -> str:
    for word in words:
        if word in text:
            return word
    return ""


def _visible(text: str) -> str:
    """归一化以便匹配：去零宽字符、全角转半角空格、小写化。

    群里打广告的常见规避手段就是在词中间插零宽字符或空格，所以匹配前先清理。
    注意：证据展示用的是原文，不是这个清理后的串。
    """
    cleaned = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    cleaned = cleaned.replace("\ufeff", "")
    return cleaned.lower()


def is_never_punish(text: str) -> bool:
    """是否属于"无论怎样都不该处罚"的内容（批评机器人、求助、观点分歧）。"""
    plain = _visible(text)
    if _mentions_bot_with_negative(plain):
        return True
    return any(pattern.search(plain) for pattern in _NEVER_PUNISH)


def judge_ad(text: str, *, history: Optional[List[str]] = None,
             rules: Optional[ModerationRules] = None) -> ModerationVerdict:
    """判断一条群消息是不是广告/刷屏。

    history：该用户最近在该群说过的话（用于重复刷屏判定）。
    返回值 hit=False 表示"不判定为广告"，上层应当什么都不做。
    """
    rules = rules or ModerationRules()
    raw = str(text or "")
    plain = _visible(raw)
    if not plain.strip():
        return ModerationVerdict()

    # 0) 管理员白名单词
    for word in rules.allow_words:
        if word and word.lower() in plain:
            return ModerationVerdict()

    # 1) 绝不处罚：批评机器人 / 正常求助 / 观点分歧
    #    放在最前面，确保任何后续规则都不能推翻它。
    if is_never_punish(plain):
        return ModerationVerdict()

    # 2) 管理员自定义敏感词：命中即 medium（由管理员负责调准）
    hit = _find(plain, [w.lower() for w in rules.ban_words if w])
    if hit:
        return ModerationVerdict(True, rule="ban_word", label="管理员设定词",
                                 confidence=CONF_MEDIUM,
                                 reason=f"命中本群设定词「{hit}」",
                                 evidence=_snippet(raw, hit),
                                 suggested=ACTION_WARN)

    # 3) 正常上下文词：出现就整体降级为不处理
    #    例如"官方公告""怎么抽卡"这类，不该因为它们含"群"或"链接"就被判广告。
    normal = _find(plain, _NORMAL_CONTEXT)

    # 4) 邀请进群/加微信/电商链接（高置信度广告特征）
    for pattern in _INVITE_PATTERNS:
        found = pattern.search(plain)
        if found:
            # 有正常上下文词时降级为 low，只提示管理员
            confidence = CONF_LOW if normal else CONF_HIGH
            return ModerationVerdict(
                True, rule="invite", label="拉群/引流信息", confidence=confidence,
                reason="消息里包含拉群、加好友或电商链接的邀请",
                evidence=found.group(0)[:60],
                suggested=ACTION_WARN if confidence == CONF_LOW else ACTION_MUTE)

    # 5) 促销词 / 主动拉私聊
    #    分三档，避免"一句话就判广告"的误伤：
    #      促销词 + （链接 / 促销语气 / 联系方式） -> medium，可自动警告
    #      促销词 单独出现                        -> low，只记录
    #      "私聊我""加微信"这类**本身就是导流**     -> medium
    #    最后这条是实测补的：光看"卖号"会判成 low，但"卖号 私聊我"明显是在导流。
    promo = _find(plain, _PROMO_WORDS)
    contact = _find(plain, _CONTACT_WORDS)
    url = _URL_RE.search(plain)
    tone = _find(plain, _PROMO_TONE)
    if contact and (promo or tone):
        confidence = CONF_LOW if normal else CONF_MEDIUM
        return ModerationVerdict(
            True, rule="contact", label="引导私下交易", confidence=confidence,
            reason=f"同时出现推广词「{promo or tone}」和联系方式「{contact}」",
            evidence=_snippet(raw, contact), suggested=ACTION_WARN)
    if promo and (url or tone or "群" in plain):
        confidence = CONF_LOW if normal else CONF_MEDIUM
        return ModerationVerdict(
            True, rule="promo", label="推广/交易信息", confidence=confidence,
            reason=f"消息里出现推广词「{promo}」" + ("并附带链接" if url else ""),
            evidence=_snippet(raw, promo), suggested=ACTION_WARN)
    if promo:
        return ModerationVerdict(True, rule="promo_weak", label="疑似推广词",
                                 confidence=CONF_LOW,
                                 reason=f"出现推广词「{promo}」，但缺少链接或联系方式，可能是正常聊天",
                                 evidence=_snippet(raw, promo), suggested=ACTION_NONE)

    # 6) 重复刷屏：同一内容反复发
    if history:
        same = sum(1 for item in history[-rules.repeat_threshold * 2:]
                   if _visible(item).strip() == plain.strip())
        if same + 1 >= rules.repeat_threshold:
            return ModerationVerdict(
                True, rule="repeat", label="重复刷屏", confidence=CONF_MEDIUM,
                reason=f"同样的内容在短时间内出现了 {same + 1} 次",
                evidence=_snippet(raw, raw[:20]), suggested=ACTION_WARN)

    return ModerationVerdict()


def _snippet(raw: str, needle: str, width: int = 20) -> str:
    """截取包含关键词的一小段原文，供管理员复核。"""
    if not needle:
        return ""
    idx = raw.lower().find(needle.lower())
    if idx < 0:
        return raw[:width]
    start = max(0, idx - width // 2)
    return raw[start:start + width * 2]


def needs_confirmation(action: str, rules: Optional[ModerationRules] = None) -> bool:
    """该动作是否需要管理员确认。

    V5：禁言、踢出和批量撤回等高影响操作默认需要确认。
    """
    rules = rules or ModerationRules()
    return action in (rules.confirm_actions or HIGH_RISK_ACTIONS)


def escalate_action(warn_count: int, rules: Optional[ModerationRules] = None) -> str:
    """按累计警告次数决定升级到哪个动作。

    0 或未达阈值时只警告；达到阈值才升级为禁言。
    **不自动踢人**——踢出是不可逆的高影响操作，必须由管理员显式决定。
    """
    rules = rules or ModerationRules()
    if rules.warn_threshold and warn_count >= rules.warn_threshold:
        return ACTION_MUTE
    return ACTION_WARN


def build_rules(settings: Any) -> ModerationRules:
    """从群设置（GroupSettings）构造规则集。

    settings 可能是 None（群没配置过）——那时返回默认规则：
    **enabled 仍为 True 但 dry_run 也是 True**，即只观察不动作。

    **注意取的是 ban_word_list() 而不是 ban_words**：后者是管理员存进去的
    一整行文本（「词1,词2」），直接当列表用会退化成"每个字一条规则"，
    于是任何消息都会命中。这个坑踩过一次，见 test_moderation 的回归用例。

    **dry_run / auto_warn 的 None 语义**：这两个键的"默认"就是"开"，
    所以 None 必须解释成 True。否则管理员执行 /群设置 审核试运行 关 时，
    apply() 把它存成 None（=跟随默认），又变回"开"，根本关不掉——
    实测踩过，表现为"关了试运行还是没有真的警告"。
    """
    if settings is None:
        return ModerationRules()
    if hasattr(settings, "ban_word_list"):
        ban_words = list(settings.ban_word_list())
        allow_words = list(settings.allow_word_list())
    else:  # 兼容普通对象/字典
        raw_ban = getattr(settings, "ban_words", None) or []
        raw_allow = getattr(settings, "allow_words", None) or []
        ban_words = list(raw_ban) if isinstance(raw_ban, (list, tuple)) else [raw_ban]
        allow_words = list(raw_allow) if isinstance(raw_allow, (list, tuple)) else [raw_allow]
    raw_dry = getattr(settings, "moderation_dry_run", None)
    raw_warn = getattr(settings, "moderation_auto_warn", None)
    return ModerationRules(
        enabled=bool(getattr(settings, "moderation", True)),
        dry_run=True if raw_dry is None else bool(raw_dry),
        auto_warn=True if raw_warn is None else bool(raw_warn),
        warn_threshold=int(getattr(settings, "warn_threshold", 0) or 0),
        ban_words=[w for w in ban_words if w],
        allow_words=[w for w in allow_words if w],
    )
