"""长期记忆：从对话中提取值得长期保存的信息，并做敏感信息拦截。

设计原则（对应文档"四、长期记忆"）：
1. 只保存对长期聊天有帮助的稳定信息；
2. 密码/验证码/支付/证件/精确住址等一律不保存；
3. 用户要求"记住"时必须给出明确反馈；
4. 记忆永远按 user_id 隔离。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 命中这些词就整条丢弃，不做任何存储
SENSITIVE_PATTERNS: Tuple[str, ...] = (
    "密码", "口令", "passwd", "password", "验证码", "短信码", "动态码", "token", "api_key", "apikey",
    "密钥", "私钥", "身份证", "证件号", "护照号", "银行卡", "卡号", "信用卡", "支付密码",
    "手机号", "电话号码", "住址", "家庭住址", "身份证号", "cvv", "安全码", "密保",
)

SENSITIVE_RE = re.compile("|".join(re.escape(p) for p in SENSITIVE_PATTERNS), re.IGNORECASE)

# 明确的"请记住"意图
EXPLICIT_MARKERS: Tuple[str, ...] = (
    "记住", "记一下", "记下", "帮我记", "记得", "别忘了", "不要忘记", "请牢记", "存起来",
    "存入记忆", "添加到记忆", "remember",
)
_EXPLICIT_RE = re.compile("(" + "|".join(re.escape(m) for m in EXPLICIT_MARKERS) + ")")
# 从"记住……"里剥掉指令词，留下要记的内容
_STRIP_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|一定要|千万)?\s*(?:记住|记一下|记下|记得|别忘了|不要忘记|请牢记|存起来|"
    r"存入记忆|添加到记忆|remember)\s*[:：,，。]?\s*", re.IGNORECASE)

# "请调用 remember 工具，把「X」记下来，然后告诉我结果" 这类元指令：
# 不能把整句当记忆存进去，要抽出真正的 X。
_META_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|你|爱丽丝)?\s*(?:调用|使用|用|执行)\s*[\w\u4e00-\u9fa5]*\s*"
    r"(?:工具|函数|tool|function)?\s*[:：,，。]?\s*", re.IGNORECASE)
_META_VERB_RE = re.compile(r"^(?:把|将)s*")
_META_SUFFIX_RE = re.compile(
    r"(?:记下来|记一下|存起来|记录下来|存下来|记到记忆里|写进记忆|添加到记忆|保存(?:下来|到记忆)?|"
    r"然后)?\s*(?:然后|再)?\s*(?:告诉|通知|回复|反馈|答复)(?:我|一下|给我)?[^，,。！？!?\n]{0,12}$")

# 带引号/书名号的内容优先：X 才是要记的东西
_QUOTED_RE = re.compile(r"[「『“\"《]([^」』”\"》]{2,60})[」』”\"》]")
_MEMORY_CORE_RE = re.compile(
    r"(?:记住|记一下|记下|记得|别忘了|不要忘记|请牢记|存起来|存入记忆|添加到记忆)\s*[:：,，。]?\s*(.{2,60})",
    re.IGNORECASE)
_TRAILING_NOISE_RE = re.compile(r"(?:这件事|这事情|这个|这条|好吗|好嘛|可以吗|谢谢|谢谢啦)[。！!?？~～\s]*$")
# "记住A，也B" 这种并列表达：显式记忆只保留第一小句，B 由偏好提取单独处理
_TRAILING_CLAUSE_RE = re.compile(r"[，,]\s*(?:也|还|还有|并且|以及|顺便|而且)[^，,。！？!?\n]{0,30}$")

# 第一人称偏好表达
_INTEREST_RE = re.compile(
    r"我(?:很|超|最|特别|非常|挺|蛮|比较|有点)?(喜欢|爱玩|爱看|爱听|爱|讨厌|不喜欢|不擅长|擅长)([^。！？!?\n]{1,30})")
_NICKNAME_RE = re.compile(
    r"(?:就)?(?:叫我|叫我做|称呼我为|喊我|可以叫我|请叫我)\s*([\w\u4e00-\u9fa5A-Za-z0-9_\-]{1,16}?)"
    r"(?=就好|就行|好了|吧|啊|呀|哦|嘛|了|，|,|。|！|!|？|\?|\s|$)")
_PRONOUN_FIX_RE = re.compile(r"我更喜欢|我还是喜欢")
_SELF_INTRO_RE = re.compile(r"我(?:是|叫)([\u4e00-\u9fa5A-Za-z0-9_\-]{1,12})(?:[，,。！!]|$)")

MIN_FACT_LENGTH = 2
# 上限卡在 64：这样"用户明确要求记住："前缀 + 正则能捕获的 60 字一定会超限，
# 从而让过长的内容被 _acceptable 整条拒绝，而不是被正则悄悄截断后当成合法记忆。
MAX_FACT_LENGTH = 64


@dataclass
class ExtractedFact:
    content: str
    kind: str            # nickname | interest | profile | explicit | fact
    confidence: float
    source: str = "auto"


def looks_sensitive(text: str) -> bool:
    return bool(SENSITIVE_RE.search(text or ""))


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip(" \t\r\n，,。.！!？?；;：:~～\"'“”‘’")
    return text


def _acceptable(text: str) -> bool:
    if not text:
        return False
    if not (MIN_FACT_LENGTH <= len(text) <= MAX_FACT_LENGTH):
        return False
    if looks_sensitive(text):
        return False
    # 纯符号/纯数字不记
    if re.fullmatch(r"[\W_\d]+", text):
        return False
    return True


def has_explicit_request(text: str) -> bool:
    return bool(_EXPLICIT_RE.search(text or ""))


def _extract_explicit_core(text: str) -> str:
    """从"请记住X"这类表述里抽出 X。

    处理顺序（从最可靠到最宽松）：
    1. 元指令前缀（"请调用 remember 工具，把…"）：先剥掉，再取引号里的内容；
    2. 引号内容：带「」/"" 的通常就是要点；
    3. "记住 X" 直接截取；
    4. 兜底：剥前缀后的剩余文本。
    """
    text = _META_PREFIX_RE.sub("", text.strip())
    text = _META_VERB_RE.sub("", text)

    # 长文本里"记住"后面往往跟着一整段叙述：没有引号时就别猜了，直接不存，
    # 否则会把无关内容整段存成记忆。
    if len(text) > 100 and not _QUOTED_RE.search(text):
        return ""

    quoted = _QUOTED_RE.search(text)
    if quoted:
        candidate = _clean(quoted.group(1))
        if _acceptable(candidate):
            return candidate

    match = _MEMORY_CORE_RE.search(text)
    if match:
        candidate = match.group(1)
        candidate = _META_SUFFIX_RE.sub("", candidate)
        candidate = _TRAILING_CLAUSE_RE.sub("", candidate)
        candidate = _TRAILING_NOISE_RE.sub("", candidate)
        candidate = _clean(candidate)
        if _acceptable(candidate):
            return candidate

    core = _STRIP_PREFIX_RE.sub("", text)
    core = _META_VERB_RE.sub("", core)
    core = _META_SUFFIX_RE.sub("", core)
    core = _TRAILING_CLAUSE_RE.sub("", core)
    core = _TRAILING_NOISE_RE.sub("", core)
    return _clean(core)


def extract(message: str, *, allow_sensitive: bool = False) -> List[ExtractedFact]:
    """从一条用户消息里提取候选记忆。allow_sensitive=True 时跳过敏感拦截（默认关闭）。"""
    text = _clean(message)
    if not text:
        return []

    # 注意顺序：先判断"显式要求记住"，再判断敏感信息。
    # 这样"记住我的密码是xxx"会返回 explicit_rejected 让上层如实告知，而不是静默丢弃。
    explicit_requested = has_explicit_request(text)
    if looks_sensitive(text) and not allow_sensitive:
        if explicit_requested:
            return [ExtractedFact("", "explicit_rejected", 0.0, "explicit")]
        return []

    facts: List[ExtractedFact] = []

    # 1) 称呼
    nick = _NICKNAME_RE.search(text)
    if nick:
        name = _clean(nick.group(1))
        if _acceptable(name) or (name and len(name) <= 16):
            facts.append(ExtractedFact(f"用户希望被称为「{name}」", "nickname", 0.95, "explicit"))

    # 2) 明确要求记住 -> 抽出真正要记的内容
    if explicit_requested:
        core = _extract_explicit_core(text)
        if _acceptable(core):
            facts.append(ExtractedFact(f"用户明确要求记住：{core}", "explicit", 0.9, "explicit"))
        else:
            # 内容太短、或涉及敏感信息：返回一个空占位，让上层如实告诉用户"没有保存"
            facts.append(ExtractedFact("", "explicit_rejected", 0.0, "explicit"))

    # 3) 偏好/兴趣
    for match in _INTEREST_RE.finditer(text):
        verb, obj = match.group(1), _clean(match.group(2))
        if not _acceptable(obj):
            continue
        if verb in ("讨厌", "不喜欢", "不擅长"):
            content = f"用户{verb}{obj}"
        elif verb == "擅长":
            content = f"用户擅长{obj}"
        else:
            content = f"用户喜欢{obj}"
        facts.append(ExtractedFact(content, "interest", 0.7, "auto"))

    # 4) 稳定的自我介绍（跳过明显没价值的句子）
    intro = _SELF_INTRO_RE.search(text)
    if intro and len(text) <= 24:
        value = _clean(intro.group(1))
        if _acceptable(value) and value not in ("谁", "什么", "个"):
            facts.append(ExtractedFact(f"用户自称「{value}」", "profile", 0.5, "auto"))

    # 去重，保留最高置信度
    dedup = {}
    for fact in facts:
        if not fact.content:
            continue
        key = fact.content
        if key not in dedup or fact.confidence > dedup[key].confidence:
            dedup[key] = fact
    # 显式要求记忆的条目排在最前，便于上层挑选"需要向用户确认"的记录
    kind_rank = {"explicit": 0, "nickname": 1, "interest": 2, "profile": 3, "fact": 4}
    return sorted(dedup.values(), key=lambda f: (kind_rank.get(f.kind, 9), -f.confidence))


def render_memory_block(facts, limit: int = 30) -> str:
    """把记忆列表渲染成注入提示词的文本块。"""
    if not facts:
        return "（暂无长期记忆）"
    lines = []
    for fact in list(facts)[:limit]:
        tag = {"nickname": "称呼", "interest": "偏好", "explicit": "用户要求记录",
               "profile": "自我描述", "fact": "其他"}.get(fact.kind, "其他")
        lines.append(f"- [{tag}] {fact.content}")
    return "\n".join(lines)
