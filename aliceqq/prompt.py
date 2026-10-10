"""提示词渲染：把上下文变量注入系统提示词。

支持 {{var}} 形式，默认保留未赋值的占位符以便排查漏注入的变量。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

# 文档"建议注入的上下文变量"
CONTEXT_VARIABLES = (
    "mode", "user_id", "user_name", "group_id", "is_private_chat", "current_time",
    "user_memory", "conversation_summary", "group_context", "knowledge_base", "group_config",
    "available_tools", "quota_status", "permission_context", "media_catalog", "event_type",
)

EMPTY_TEXT = "（暂无）"


class MissingVariableError(KeyError):
    pass


def render(template: str, variables: Mapping[str, Any], *, strict: bool = False,
           keep_unknown: bool = True, empty_text: str = EMPTY_TEXT) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name in variables:
            value = variables[name]
            if value is None or value == "":
                return empty_text
            if isinstance(value, (list, tuple)):
                return "\n".join(str(v) for v in value) or empty_text
            return str(value)
        if strict:
            raise MissingVariableError(name)
        return match.group(0) if keep_unknown else empty_text

    return PLACEHOLDER_RE.sub(replace, template)


def find_placeholders(template: str) -> List[str]:
    return sorted(set(PLACEHOLDER_RE.findall(template)))


def load_template(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def list_variables(template: str, provided: Mapping[str, Any]) -> Dict[str, List[str]]:
    used = find_placeholders(template)
    return {
        "used": used,
        "missing": [name for name in used if name not in provided],
        "unused": [name for name in CONTEXT_VARIABLES if name not in used],
    }


def join_blocks(title: str, body: Optional[str]) -> str:
    if not body:
        return ""
    return f"### {title}\n{body.strip()}"


def compose_sections(sections: Iterable[str]) -> str:
    return "\n\n".join(s for s in sections if s)
