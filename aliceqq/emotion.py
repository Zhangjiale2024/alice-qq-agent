"""情感色彩：从对话推断该用什么语气说话。

为什么用**确定性规则**而不是让模型自报情感：
1. 模型自报的情感标签很容易漏进最终回复里（"（开心）你好呀"），
   而这类泄漏用户一眼就能看到；
2. 规则可测、可解释、可复现——情感判断错了好定位；
3. 与现有护栏风格一致：能确定性做的事不交给模型。

情感最终要被 TTS 用上，所以这里输出的是**有限枚举**，不是自由文本。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 情感枚举。key 同时用作 TTS 侧 emotion 参数值。
NEUTRAL = "neutral"
HAPPY = "happy"
EXCITED = "excited"
SAD = "sad"
ANGRY = "angry"
SURPRISED = "surprised"
SHY = "shy"
WORRIED = "worried"
GENTLE = "gentle"

ALL_EMOTIONS = (NEUTRAL, HAPPY, EXCITED, SAD, ANGRY, SURPRISED, SHY, WORRIED, GENTLE)

LABELS = {
    NEUTRAL: "平静",
    HAPPY: "开心",
    EXCITED: "兴奋",
    SAD: "难过",
    ANGRY: "生气",
    SURPRISED: "惊讶",
    SHY: "害羞",
    WORRIED: "担心",
    GENTLE: "温柔",
}

# 用户显式指定语气："用开心的语气说" / "语气开心一点"
_EXPLICIT = [
    (EXCITED, ("激动", "兴奋", "热血", "燃起来")),
    (HAPPY, ("开心", "高兴", "愉快", "欢快", "元气", "活泼")),
    (SAD, ("难过", "伤心", "低落", "悲伤", "沮丧")),
    (ANGRY, ("生气", "愤怒", "凶一点", "严肃")),
    (SURPRISED, ("惊讶", "吃惊", "震惊")),
    (SHY, ("害羞", "不好意思", "娇羞")),
    (WORRIED, ("担心", "忧虑", "关切")),
    (GENTLE, ("温柔", "轻声", "细声", "安慰")),
    (NEUTRAL, ("平静", "正常", "平淡")),
]
_EXPLICIT_PATTERN = re.compile(
    r"(用|以|语气|声音|说|讲)[^，。！？!?]{0,12}?(开心|高兴|愉快|欢快|元气|活泼|激动|兴奋|"
    r"热血|难过|伤心|低落|悲伤|沮丧|生气|愤怒|严肃|惊讶|吃惊|震惊|害羞|不好意思|"
    r"担心|忧虑|温柔|轻声|安慰|平静|正常)")

# 文本里的情感信号。顺序即优先级（强的在前）
_SIGNALS: List[Tuple[str, Tuple[str, ...]]] = [
    (ANGRY, ("气死", "讨厌", "烦人", "别闹", "过分", "可恶")),
    (SAD, ("呜呜", "难过", "伤心", "遗憾", "可惜", "对不起", "抱歉", "失落", "孤独")),
    (EXCITED, ("太棒了", "厉害", "冲啊", "赢了", "成功", "！", "!")),
    (HAPPY, ("哈哈", "开心", "高兴", "太好了", "喜欢", "棒", "笑", "嘿嘿", "嘻")),
    (SURPRISED, ("居然", "竟然", "没想到", "真的吗", "诶", "欸", "哇")),
    (SHY, ("害羞", "脸红", "别、别", "才不是", "不好意思")),
    (WORRIED, ("担心", "没事吧", "要紧吗", "小心", "别勉强", "注意")),
    (GENTLE, ("加油", "慢慢来", "陪着你", "没关系", "会好的", "别怕")),
]

# QQ/Unicode 表情暗示
_FACE_SIGNALS = {
    HAPPY: ("笑", "嘻", "嘿", "愉快", "呲牙", "憨笑"),
    SAD: ("哭", "泪", "委屈", "难过"),
    ANGRY: ("怒", "发火", "打人", "敲打"),
    SURPRISED: ("惊讶", "呆", "疑问", "懵逼"),
    SHY: ("害羞", "红脸"),
}


@dataclass
class EmotionGuess:
    emotion: str = NEUTRAL
    reason: str = ""
    explicit: bool = False

    @property
    def label(self) -> str:
        return LABELS.get(self.emotion, self.emotion)


def _score(text: str) -> List[Tuple[str, int, str]]:
    """给文本里的情感信号打分。返回按分数降序的 (情感, 分数, 命中词)。"""
    if not text:
        return []
    hits: List[Tuple[str, int, str]] = []
    for emotion, words in _SIGNALS:
        for word in words:
            if word in text:
                # 标点类信号权重低一些，避免"！"就压过具体词
                weight = 1 if word in ("！", "!") else 3
                hits.append((emotion, weight, word))
    # CQ 表情码
    for match in re.findall(r"\[CQ:face,[^\]]*\]", text):
        for emotion, words in _FACE_SIGNALS.items():
            if any(w in match for w in words):
                hits.append((emotion, 4, match[:24]))
    # 连续感叹号/问号
    if re.search(r"[！!]{2,}", text):
        hits.append((EXCITED, 3, "多个感叹号"))
    if re.search(r"[？?]{2,}", text):
        hits.append((SURPRISED, 2, "多个问号"))
    hits.sort(key=lambda item: item[1], reverse=True)
    return hits


def detect_explicit(text: str) -> Optional[str]:
    """识别"用开心的语气说"这类显式要求。"""
    if not text:
        return None
    if not re.search(r"(用|以|语气|声音|说|讲|点)", text):
        return None
    match = _EXPLICIT_PATTERN.search(text)
    if not match:
        return None
    keyword = match.group(2)
    for emotion, words in _EXPLICIT:
        if keyword in words:
            return emotion
    return None


def infer(user_text: str = "", reply_text: str = "",
          explicit: Optional[str] = None) -> EmotionGuess:
    """推断该用什么语气。

    优先级：用户显式要求 > 回复正文的信号 > 用户消息的信号 > 平静。
    回复正文优先于用户消息，因为要念出来的是回复本身。
    """
    if explicit and explicit in ALL_EMOTIONS:
        return EmotionGuess(explicit, f"用户指定语气：{LABELS[explicit]}", explicit=True)

    from_request = detect_explicit(user_text)
    if from_request:
        return EmotionGuess(from_request, f"用户要求用{LABELS[from_request]}的语气", True)

    for source_name, text in (("回复", reply_text), ("用户消息", user_text)):
        hits = _score(text)
        if hits:
            emotion, score, word = hits[0]
            return EmotionGuess(emotion, f"{source_name}中的「{word}」", False)
    return EmotionGuess(NEUTRAL, "未发现明显情感信号")


def describe(guess: EmotionGuess) -> str:
    return f"{guess.label}（{guess.reason}）"
