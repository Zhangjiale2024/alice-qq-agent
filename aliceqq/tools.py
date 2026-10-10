"""工具注册表。

V1 只暴露两个工具，且默认都不真正执行副作用：
- remember: 由服务端逻辑直接落库，模型只负责决定"要记什么"；
- web_search: 配置开启但 V1 未接入真实检索，明确回复"暂不可用"，绝不编造结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import AppConfig


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    available: bool = True
    unavailable_hint: str = ""

    def to_prompt_line(self) -> str:
        state = "" if self.available else f"（{self.unavailable_hint or '当前不可用'}）"
        return f"- {self.name}: {self.description}{state}"


@dataclass
class ToolResult:
    tool: str
    ok: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


def registry(config: AppConfig, search_available: bool = False,
             academic_available: bool = False) -> Dict[str, ToolSpec]:
    # 注意：WEB_SEARCH_IN_ALICE_MODE 只表达"产品方向希望启用"；
    # 真正决定工具可不可用的是 search 配置 + 服务能否构建成功。
    web_available = bool(config.features.web_search) and bool(search_available)
    return {
        "remember": ToolSpec(
            "remember",
            "把用户明确要求长期保存的信息写入该用户自己的记忆库（仅限当前用户）。",
            {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            available=config.memory.enabled,
            unavailable_hint="记忆功能未启用",
        ),
        "web_search": ToolSpec(
            "web_search",
            "联网检索最新信息。",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            # 注意：WEB_SEARCH_IN_ALICE_MODE 只表示"产品方向希望启用"，
            # V1 并未接入真实检索，因此这里始终标记为不可用——
            # 否则模型会以为工具可用，顺着用户期待演一遍"我去查一下"。
            available=web_available,
            unavailable_hint="未启用联网检索（检查 SEARCH_PROVIDER / SEARCH_ENABLED）",
        ),
        "academic_search": ToolSpec(
            "academic_search",
            "学术文献检索（CrossRef / Semantic Scholar / arXiv）：按标题、作者或 DOI 查论文，"
            "返回标题、摘要、年份与链接。仅适用于学术问题，不适用于游戏/动漫/新闻。",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            available=bool(academic_available),
            unavailable_hint="未启用学术检索（检查 ACADEMIC_ENABLED）",
        ),
    }


def catalog_text(config: AppConfig, search_available: bool = False,
                academic_available: bool = False) -> str:
    reg = registry(config, search_available, academic_available)
    lines = [spec.to_prompt_line() for spec in reg.values()]
    if not any(spec.available for spec in reg.values()):
        lines.append("- （当前没有任何可用工具，遇到超出能力范围的请求要诚实说明）")
    if reg["web_search"].available:
        lines.append(
            "- web_search 可用：涉及最新新闻、活动、价格、规则、时间安排或你不确定的事实时，"
            "应当先检索再回答；引用结论时附上来源名称或链接。"
        )
    if reg["academic_search"].available:
        lines.append(
            "- academic_search 可用：只在用户问论文、文献、研究、DOI 或学术概念时使用；"
            "返回的是文献元数据与摘要，引用时必须给出标题与链接，"
            "**不要编造卷号、页码、引用次数或影响因子**。"
            "游戏、动漫、新闻类问题不要用它。"
        )
    lines.append(
        "- 硬性规则：标注为不可用的工具**绝对不能尝试调用**，也不能说"
        "「我去查一下」「正在检索」「稍等」这类话。"
        "没有联网检索、图片生成和语音能力时，被要求必须直接说明做不到，并给出替代方案。"
    )
    return "\n".join(lines)


def call(config: AppConfig, name: str, arguments: Dict[str, Any]) -> ToolResult:
    """V1 的兜底实现：不在这里做真实网络请求，避免编造结果。"""
    reg = registry(config)
    spec = reg.get(name)
    if spec is None:
        return ToolResult(name, False, f"未知工具：{name}")
    if not spec.available:
        return ToolResult(name, False, f"{name} 当前不可用：{spec.unavailable_hint or '功能未启用'}")
    if name == "web_search":
        return ToolResult(name, False, "联网搜索还没有接入到当前版本，爱丽丝暂时查不到最新的网络信息。", {"deferred": True})
    if name == "remember":
        content = str(arguments.get("content", "")).strip()
        if not content:
            return ToolResult(name, False, "没有给出要记住的内容。")
        return ToolResult(name, True, "已记录。", {"content": content})
    return ToolResult(name, False, f"工具 {name} 未实现")
