"""知识库管理：角色资料、世界观资料、所有者上传资料。

V2 的要求是"知识来源有优先级"，所以这里用文件名前缀编码优先级：

    00_*.md  项目设定 / 最高优先（例如本 Agent 自己的规则）
    10_*.md  角色资料（爱丽丝的设定）
    20_*.md  世界观资料（蔚蓝档案背景）
    30_*.md  普通资料
    90_*.md  所有者上传（默认最高优先，可用 KNOWLEDGE_UPLOAD_PREFIX 调整）

排序规则：**前缀数字小的优先**；同前缀按文件名排序。
做法上不是"只取前几篇"，而是按优先级拼接，并在总量超限时先丢低优先级的尾部，
这样高优先级的设定永远在提示词里。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

UPLOAD_PREFIX = "90"
DEFAULT_MAX_CHARS = 12000      # 注入提示词的知识总量上限
DEFAULT_MAX_FILE_CHARS = 6000  # 单篇上限
SAFE_NAME_RE = re.compile(r"[^\w\u4e00-\u9fa5\-]+")


@dataclass
class KnowledgeDoc:
    path: Path
    prefix: int
    title: str
    content: str
    uploaded: bool = False
    truncated: bool = False

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class KnowledgeBase:
    docs: List[KnowledgeDoc] = field(default_factory=list)
    total_chars: int = 0
    dropped: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.docs)


def _prefix_of(name: str) -> int:
    match = re.match(r"^(\d{1,3})[_\-]", name)
    if match:
        return int(match.group(1))
    # 没写前缀的按 50（排在角色/世界观之后，上传之前）
    return 50


def safe_filename(stem: str, *, prefix: str = UPLOAD_PREFIX) -> str:
    cleaned = SAFE_NAME_RE.sub("_", (stem or "").strip())[:40].strip("_")
    if not cleaned:
        cleaned = time.strftime("upload_%Y%m%d_%H%M%S")
    return f"{prefix}_{cleaned}.md"


def load(directory: Path, *, max_chars: int = DEFAULT_MAX_CHARS,
         max_file_chars: int = DEFAULT_MAX_FILE_CHARS) -> KnowledgeBase:
    """按优先级加载知识库，并做容量控制。"""
    base = KnowledgeBase()
    if not directory.is_dir():
        return base

    entries: List[Tuple[int, str, Path]] = []
    for path in sorted(directory.glob("*.md")):
        entries.append((_prefix_of(path.name), path.name, path))
    entries.sort(key=lambda item: (item[0], item[1]))

    for prefix, _name, path in entries:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw:
            continue
        truncated = False
        if max_file_chars > 0 and len(raw) > max_file_chars:
            raw = raw[:max_file_chars].rstrip() + "\n…（本文件过长，已截断）"
            truncated = True
        if base.total_chars + len(raw) > max_chars and base.docs:
            # 超总量：丢掉后续（低优先级）文件，保证高优先级内容完整
            base.dropped.append(path.name)
            continue
        title = _first_heading(raw) or path.stem
        base.docs.append(KnowledgeDoc(path=path, prefix=prefix, title=title,
                                      content=raw, uploaded=prefix >= int(UPLOAD_PREFIX),
                                      truncated=truncated))
        base.total_chars += len(raw)
    return base


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def render(base: KnowledgeBase) -> str:
    """渲染成注入提示词的文本块，并标明来源类别。"""
    if not base.docs:
        return "（未加载任何知识资料）"
    blocks: List[str] = []
    for doc in base.docs:
        kind = "所有者上传" if doc.uploaded else _kind_name(doc.prefix)
        blocks.append(f"<!-- 来源：{kind} | {doc.name} -->\n{doc.content}")
    return "\n\n".join(blocks)


def _kind_name(prefix: int) -> str:
    if prefix <= 0:
        return "项目设定"
    if prefix < 20:
        return "角色资料"
    if prefix < 30:
        return "世界观资料"
    return "补充资料"


def catalog_text(base: KnowledgeBase) -> str:
    """给 /资料 与启动横幅用的清单。"""
    if not base.docs:
        return "（知识库为空）"
    lines = []
    for doc in base.docs:
        kind = "上传" if doc.uploaded else _kind_name(doc.prefix)
        flag = "（已截断）" if doc.truncated else ""
        lines.append(f"- [{kind}] {doc.name}｜{doc.title}｜{len(doc.content)} 字{flag}")
    if base.dropped:
        lines.append(f"- 因超出容量未加载：{'、'.join(base.dropped)}")
    lines.append(f"合计 {base.count} 篇 / {base.total_chars} 字")
    return "\n".join(lines)


def save_upload(directory: Path, title: str, content: str,
                *, prefix: str = UPLOAD_PREFIX) -> Path:
    """保存所有者上传的资料。返回写入路径。"""
    directory.mkdir(parents=True, exist_ok=True)
    body = content.strip()
    if not body.startswith("#"):
        body = f"# {title.strip()}\n\n{body}"
    path = directory / safe_filename(title, prefix=prefix)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def delete_doc(directory: Path, name: str) -> bool:
    """删除一篇资料。只允许删除目录下的 .md，避免路径穿越。"""
    candidate = (directory / Path(name).name)
    if candidate.suffix != ".md" or not candidate.is_file():
        return False
    try:
        candidate.unlink()
        return True
    except OSError:
        return False


def pending_upload_path(directory: Path) -> Path:
    """适配层收到文件时暂存到这里的约定位置。"""
    return directory / ".pending"


def record_pending(directory: Path, user_id: str, name: str, content: str) -> Path:
    pending = pending_upload_path(directory)
    pending.mkdir(parents=True, exist_ok=True)
    target = pending / f"{SAFE_NAME_RE.sub('_', str(user_id))}.md"
    target.write_text(f"<!-- upload-name: {name} -->\n{content}", encoding="utf-8")
    return target


def take_pending(directory: Path, user_id: str) -> Optional[Tuple[str, str]]:
    """取出并清理暂存文件，返回 (名字, 内容)。"""
    pending = pending_upload_path(directory)
    target = pending / f"{SAFE_NAME_RE.sub('_', str(user_id))}.md"
    if not target.is_file():
        return None
    try:
        raw = target.read_text(encoding="utf-8")
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    name = "上传资料"
    lines = raw.splitlines()
    if lines and lines[0].startswith("<!-- upload-name:"):
        name = lines[0].split(":", 1)[1].replace("-->", "").strip() or name
        raw = "\n".join(lines[1:])
    return name, raw.strip()
