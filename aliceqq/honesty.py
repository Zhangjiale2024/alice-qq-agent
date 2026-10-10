"""诚实护栏：确定性拦截"假装做了没做的事"。

为什么需要它：即使提示词明确写了"不要编造已执行的操作"，模型仍然会因为
web_search 这类工具出现在工具列表里，而顺着用户的期待演一遍"我去查一下"。
这类幻觉在角色扮演场景里格外有害——用户会真的以为机器人联网了。

所以这里做一层**确定性**的兜底：识别"用户要的正是我们做不到的能力" + "回复里
出现了假装在执行/已完成的表达"，两者同时成立就替换成如实说明。
它不是概率性的，也不依赖模型配合。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# 用户请求了哪些当前做不到的能力
# ---------------------------------------------------------------------------

CAPABILITY_PATTERNS = {
    "web_search": re.compile(
        r"(上网|联网|网上搜|搜一下|搜索一下|查一下|帮我查|查查|检索|"
        r"最新的?(新闻|消息|资讯|情况)|今天有什么(新闻|消息)|实时|"
        r"百度一下|google|谷歌一下)", re.IGNORECASE),
    "image_generation": re.compile(
        r"(画一?[张幅个]|画个|生成.{0,6}(图片|图|插画)|来一?张.{0,4}(图|照片)|"
        r"做个表情包|p一?张图|修图)", re.IGNORECASE),
    "voice": re.compile(r"(发(个|条)?语音|用语音说|说句话给我听|voice)", re.IGNORECASE),
}

# 只有回复里出现这些"假装在行动/已完成"的表达时才算幻觉
FABRICATION_MARKERS = (
    "稍等", "请稍等", "等一下", "马上", "正在打开", "正在检索", "正在查询", "正在处理",
    "已为您", "已经查到", "查到了", "搜索结果", "检索结果", "我找到", "爱丽丝找到",
    "根据最新", "已发送", "已经发出", "已经生成", "已经画", "画好了", "生成好了",
    "正在生成", "正在画", "链接如下", "以下是新闻", "刚刚查",
)

# 真话的兜底文案：如实说明，同时保持爱丽丝的语气，并给出替代方案
FALLBACKS = {
    "web_search": (
        "……爱丽丝现在还没接上外部网络，查不到最新消息。\n"
        "如果阿岚把看到的内容贴过来，爱丽丝可以陪你一起看、一起分析。"
    ),
    "image_generation": (
        "……爱丽丝现在还不能生成图片。\n"
        "不过可以帮你想画面、写描述，或者找找现成的表情包。"
    ),
    "voice": "……爱丽丝现在还不能发语音。先用文字聊天可以吗？",
}


# ---------------------------------------------------------------------------
# 内部思考泄漏：模型有时把自己的推理过程也当成回复输出
# ---------------------------------------------------------------------------

LEAK_MARKERS = (
    "长期记忆区块", "记忆区块", "系统已经保存", "系统已保存", "直接自然确认",
    "不需要重复调用", "不需要再调用", "我直接", "我应当", "我应该", "作为AI", "作为 AI",
    "根据系统提示", "系统提示词", "工具返回", "工具结果", "tool_call", "调用工具",
    "按照要求", "根据规则", "提示词要求", "已经出现", "说明系统", "无需调用",
    "用户的请求", "用户的意图", "触发", "内部",
)

# 明显的自述式第一人称（爱丽丝用第三人称自称，出现这些基本可判定是模型自述）
SELF_TALK_RE = re.compile(r"^(我|我们)(需要|应该|应当|要|得|直接|先|可以|来|会|已)")

_STRAY_MARKUP_RE = re.compile(r"^\s*(::|【思考】|【分析】|思考[:：]|推理[:：]|Analysis:|Thinking:).*$",
                              re.IGNORECASE)


@dataclass
class HonestyVerdict:
    replaced: bool
    capability: str = ""
    reason: str = ""
    text: Optional[str] = None


def requested_capabilities(text: str) -> List[str]:
    return [name for name, pattern in CAPABILITY_PATTERNS.items() if pattern.search(text or "")]


def looks_fabricated(reply: str) -> bool:
    return any(marker in (reply or "") for marker in FABRICATION_MARKERS)


def check(user_text: str, reply: str, available_capabilities: Optional[set] = None,
          *, can_search: bool = False, search_failed: bool = False) -> HonestyVerdict:
    """如果回复在假装执行当前做不到的能力，返回替换方案。

    联网检索要分三种情形，不能一刀切：
    1. 没接入检索：说"稍等/正在检索"就是幻觉，必须换成如实说明；
    2. 接入了但这次失败：不能假装查到了，要说明检索失败；
    3. 接入了且成功：正常引用结果，不做拦截（由提示词约束来源标注）。
    """
    available = available_capabilities or set()
    requested = requested_capabilities(user_text)

    if "web_search" in requested:
        if not can_search and looks_fabricated(reply):
            return HonestyVerdict(
                replaced=True, capability="web_search",
                reason="用户请求联网检索，但该能力未接入，回复却出现执行类表达",
                text=FALLBACKS["web_search"],
            )
        if can_search and search_failed and claims_search_success(reply):
            return HonestyVerdict(
                replaced=True, capability="web_search",
                reason="检索失败，但回复仍以已获得最新信息的口吻作答",
                text=None,  # 由调用方用 search_failure_text 生成，以便带上失败原因
            )

    for capability in requested:
        if capability == "web_search":
            continue
        if capability in available:
            continue
        if looks_fabricated(reply):
            return HonestyVerdict(
                replaced=True,
                capability=capability,
                reason=f"用户请求 {capability}，该能力未实现，但回复出现了执行/完成类表达",
                text=FALLBACKS[capability],
            )
    return HonestyVerdict(replaced=False)


def sanitize_reply(reply: str) -> str:
    """删掉模型泄漏进回复的内部思考行。

    做法很保守：逐行判断，只删除"明确属于推理过程"的行，不碰正常台词。
    如果一个段落里出现泄漏标记，且该段整体很短（说明它就是那句思考），也一并删掉。
    """
    if not reply:
        return reply
    kept = []
    for raw_line in reply.split("\n"):
        line = raw_line.strip()
        if not line:
            kept.append(raw_line)
            continue
        if _STRAY_MARKUP_RE.match(line):
            continue
        if SELF_TALK_RE.match(line):
            continue
        if len(line) <= 200 and any(marker in line for marker in LEAK_MARKERS):
            continue
        kept.append(raw_line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# 检索成功时模型会用的表达；检索失败后仍出现这些就是"拿旧知识冒充最新信息"
SEARCH_SUCCESS_MARKERS = (
    "根据最新", "据最新", "我查到", "查到了", "检索到", "搜索到", "找到了相关",
    "来源:", "来源：", "链接:", "链接：", "https://", "http://",
    "据报道", "官方公告显示", "已确认",
)


def claims_search_success(reply: str) -> bool:
    text = reply or ""
    return any(marker in text for marker in SEARCH_SUCCESS_MARKERS)


def search_failure_text(query: str, error: str = "") -> str:
    """检索失败时的如实说明：不拿旧知识冒充最新信息。"""
    reason = f"（失败原因：{error}）" if error else ""
    return (
        f"关于「{query}」的最新信息，爱丽丝这边检索没有成功{reason}，"
        "所以没法给出准确的当前情况。\n"
        "如果不着急，可以过一会儿再让爱丽丝试一次；你也可以直接把看到的页面内容贴过来，"
        "爱丽丝帮你一起看。"
    )


# 学术场景下"凭空报出论文"的特征：看起来像在给具体文献信息
PAPER_CLAIM_MARKERS = (
    "doi:", "DOI", "https://doi.org/", "arxiv:", "arXiv:", "et al",
    "发表于", "期刊", "卷", "第.*期", "被引", "引用次数", "影响因子",
)


def claims_paper_findings(reply: str) -> bool:
    """回复里是否给出了具体的文献细节（这类内容一旦编造，用户很难发现）。"""
    text = reply or ""
    if re.search(r"10\.\d{4,9}/\S+", text):
        return True
    return any(marker in text for marker in PAPER_CLAIM_MARKERS if not marker.startswith("第"))


def academic_failure_text(query: str, error: str = "", rate_limited: bool = False) -> str:
    """学术检索失败时的如实说明。"""
    if rate_limited:
        reason = ("Semantic Scholar 的公共额度被限流了"
                  "（这个接口按 IP 共享，免费额度很容易用满）")
        advice = ("如果你有 Semantic Scholar 的 API Key，填到 .env 的 "
                  "SEMANTIC_SCHOLAR_API_KEY 就能稳定使用；"
                  "或者直接把论文标题/DOI 发给我，可以精确查找。")
    else:
        reason = f"检索没有成功（{error}）" if error else "检索没有成功"
        advice = "可以换个更具体的说法再试，或者直接把论文标题、DOI 发给我。"
    return (f"关于「{query}」，我这边没有查到可用的文献：{reason}。\n"
            f"我不想凭记忆报出论文标题或 DOI——那种信息一旦说错，你很难发现。\n{advice}")


def available_capabilities(config, search_available: bool = False) -> set:
    """当前真正可用的能力集合（与 tools.py 的注册表保持一致）。"""
    caps = set()
    # 只有"服务真的构建成功"才算可用，避免把没实现的工具说成能用
    if search_available:
        caps.add("web_search")
    if getattr(config.memory, "enabled", False):
        caps.add("memory")
    return caps
