"""角色风格护栏（目前用于 DeepSeek 小鲸鱼）。

为什么做成确定性代码而不是只写在提示词里：
角色卡把"AI 味"列成了黑名单，还特别警告"括号内容会被当成正式消息发到群里"。
提示词能降低概率，但**拦不住必然出现的漏网**——而这些漏网对用户是直接可见的
（一条"（我不太想接话）"发到群里就露馅了）。所以像诚实护栏一样，
这里做**确定性清洗**：能删的删、能拆的拆，并把命中记进 debug 供排查。

清洗刻意保守：只在明显是 AI 味时才动手，且**宁可保留原文也不把消息清空**。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# 角色卡明确列出的黑名单（出现即失败）
BLACKLIST = (
    "首先", "其次", "综上所述", "总而言之", "值得注意的是",
    "希望这个回答对你有帮助", "有什么可以帮您", "随时找我",
    "作为AI", "作为一个语言模型", "作为人工智能",
    "这是个好问题", "我理解你的感受",
    "大家今天有什么安排吗", "你们觉得呢",
    "保证解压", "一绝", "真管用", "刺激多了", "包你满意",
    "强烈推荐", "绝对有效", "强烈安利", "一绝",
    "本小姐",
)

# 客服式/说教式收尾：整句都是这类话且没有别的句子时，不删（避免发空消息）
SOFT_BLACKLIST = ("祝你", "加油！", "记得", "一定要", "包你", "请放心")

# 有等价正确说法的词：直接替换，比删掉整句更保信息
REPLACEMENTS = (
    ("本小姐", "小鲸鱼"),      # 卡里说这是二创女仆腔，不是群友腔
    # 这几个直接删掉：后面通常本来就跟着"我"，换成"我"会变成"我我觉得"
    ("作为一个语言模型", ""),
    ("作为人工智能", ""),
    ("作为AI", ""),
)

# 每句尾巴都挂的助词（单独出现不算问题）
TAIL_PARTICLES = ("～", "~", "哦", "呢", "啦", "呀")

# 句末句号：群友打字基本不打句号
TRAILING_PERIODS = "。."

# 括号：全角/半角，可能跨行
_PAREN_PATTERNS = (
    re.compile(r"（[^（）]*）"),
    re.compile(r"\([^()]*\)"),
    re.compile(r"【[^【】]*】"),
)

_MD_PATTERNS = (
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),        # 标题
    (re.compile(r"^\s{0,3}[-*+]\s+", re.M), ""),          # 列表
    (re.compile(r"^\s{0,3}>\s?", re.M), ""),              # 引用
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),              # 加粗
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),     # 斜体
    (re.compile(r"^\s*[-*_]{3,}\s*$", re.M), ""),         # 分隔线
)

# 句子切分点（中英文标点都算）
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?…~～])")


@dataclass
class StyleReport:
    """清洗结果与命中的问题，供 debug 与测试使用。"""

    original: str = ""
    cleaned: str = ""
    violations: List[str] = field(default_factory=list)
    stripped_parens: List[str] = field(default_factory=list)
    dropped_sentences: List[str] = field(default_factory=list)
    replaced: List[str] = field(default_factory=list)
    markdown_removed: bool = False
    particles_stripped: int = 0

    @property
    def changed(self) -> bool:
        return self.cleaned != self.original

    def summary(self) -> str:
        bits = []
        if self.violations:
            bits.append("黑名单:" + "/".join(self.violations[:4]))
        if self.stripped_parens:
            bits.append("去括号:%d" % len(self.stripped_parens))
        if self.replaced:
            bits.append("替换:" + "/".join(self.replaced))
        if self.dropped_sentences:
            bits.append("删句:%d" % len(self.dropped_sentences))
        if self.markdown_removed:
            bits.append("去Markdown")
        if self.particles_stripped:
            bits.append("去尾助词:%d" % self.particles_stripped)
        return "；".join(bits) or "未命中"


def detect(text: str) -> List[str]:
    """只检测、不清洗。返回命中的黑名单短语。"""
    if not text:
        return []
    hits = [w for w in dict.fromkeys(BLACKLIST) if w in text]
    # 全大写 AI 之类的变体
    if re.search(r"\bAI\b", text) and ("作为" in text or "我是" in text):
        hits.append("作为AI(变体)")
    return hits


def _split_sentences(text: str) -> List[str]:
    parts = [p for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text] if text.strip() else [])


def _strip_markdown(text: str) -> Tuple[str, bool]:
    before = text
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    return text, text != before


def _strip_parens(text: str) -> Tuple[str, List[str]]:
    removed: List[str] = []
    for pattern in _PAREN_PATTERNS:
        for match in pattern.findall(text):
            removed.append(match.strip())
        text = pattern.sub("", text)
    return text, removed


def _drop_sentences_with(text: str, words) -> Tuple[str, List[str]]:
    """按句删掉含指定词的句子。

    两条保底：单句不删（没有可切分点，删了就整条没了）；
    全删则原样返回（宁留一点 AI 味，也不能发一条空消息）。
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text, []
    kept, dropped = [], []
    for sentence in sentences:
        if any(word in sentence for word in words):
            dropped.append(sentence.strip())
        else:
            kept.append(sentence)
    if not kept:
        return text, []
    return "".join(kept), dropped


def _drop_blacklisted_sentences(text: str) -> Tuple[str, List[str]]:
    return _drop_sentences_with(text, BLACKLIST)


def _strip_uniform_particles(text: str) -> Tuple[str, int]:
    """所有句子尾巴都挂同一个助词时才去掉——单独出现是正常语气，不该动。"""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text, 0
    tails = [s.rstrip()[-1:] for s in sentences if s.strip()]
    uniform = {t for t in tails if t in TAIL_PARTICLES}
    if len(uniform) != 1 or len(tails) != len(sentences):
        return text, 0
    particle = uniform.pop()
    fixed, count = [], 0
    for sentence in sentences:
        stripped = sentence.rstrip()
        if stripped.endswith(particle):
            fixed.append(stripped[: -len(particle)])
            count += 1
        else:
            fixed.append(sentence)
    return "".join(fixed), count


def sanitize(text: str) -> StyleReport:
    """对小鲸鱼的回复做确定性清洗。"""
    report = StyleReport(original=text or "")
    body = (text or "").strip()
    if not body:
        report.cleaned = ""
        return report

    report.violations = detect(body)

    # 有等价说法的先替换（本小姐 -> 小鲸鱼），比删句更保信息
    for bad, good in REPLACEMENTS:
        if bad in body:
            body = body.replace(bad, good)
            report.replaced.append(bad)

    body, report.markdown_removed = _strip_markdown(body)
    body, report.stripped_parens = _strip_parens(body)
    body, report.dropped_sentences = _drop_blacklisted_sentences(body)
    body, soft_dropped = _drop_sentences_with(body, SOFT_BLACKLIST)
    report.dropped_sentences += soft_dropped
    body, report.particles_stripped = _strip_uniform_particles(body)

    # 收尾修饰
    body = re.sub(r"我{2,}", "我", body)              # "我我觉得" -> "我觉得"
    body = re.sub(r"^[，,、；;\s]+", "", body)          # 删掉替换留下的前置标点
    body = re.sub(r"([，。！？])[，、]+", r"\1", body)   # 连续标点收敛
    body = re.sub(r"[ \t]{2,}", " ", body)
    lines = [ln.strip().rstrip(TRAILING_PERIODS).strip() for ln in body.splitlines()]
    body = "\n".join(ln for ln in lines if ln).strip()

    # 清洗后为空则保留原文：宁可有点 AI 味，也不能发一条空消息
    report.cleaned = body or report.original.strip()
    return report


def split_messages(text: str, max_parts: int = 3, max_chars: int = 60) -> List[str]:
    """把一条回复拆成"像真人连发"的多条。

    小鲸鱼的核心习惯是"一句话一条消息"。模型有时会把几句塞进一条，
    这里按句子切开——配合适配层的间隔发送，就等于真人在补刀。

    不截断内容：宁可多发一条，也不砍掉信息（截断会让回复变得没头没尾）。
    """
    body = (text or "").strip()
    if not body:
        return []
    # 模型已经自己换行了，就尊重它的分段
    if "\n" in body:
        parts = [p.strip() for p in body.splitlines() if p.strip()]
    else:
        parts = [p.strip() for p in _SENTENCE_SPLIT.split(body) if p.strip()]
    if not parts:
        return [body]

    # 不再"合并过短碎片"：句子边界是模型有意给的，合并会把它毁掉
    # （实测把"累了就睡。醒了继续累。"并回成了一条，正好违背分条的初衷）
    # 超长部分按长度切，强制保持碎片感
    sized: List[str] = []
    for part in parts:
        while len(part) > max_chars:
            sized.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            sized.append(part)

    if len(sized) <= max_parts:
        return sized
    # 超过上限就合并尾部，而不是丢弃
    head = sized[: max_parts - 1]
    head.append("".join(sized[max_parts - 1:]))
    return head
