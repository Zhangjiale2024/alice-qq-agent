"""媒体素材库（V4）：QQ 表情包与图片。

设计要点：
- 素材放在 media/ 下，manifest.json 记录元数据（标签、说明、来源）；
- **只在标签与话题匹配时才选**——V4 明确要求"不要发送与当前话题无关的图片"，
  所以不做"随便挑一张"的兜底；
- 区分来源：builtin（自备素材）/ generated（AI 生成）。
  V4 要求"不要把 AI 生成图片说成官方素材"，所以来源必须记录下来并传给模型。
"""
from __future__ import annotations

import json
import os
import urllib.request
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

STICKER = "sticker"
IMAGE = "image"
GENERATED = "generated"
VOICE = "voice"

KINDS = (STICKER, IMAGE, GENERATED, VOICE)

# 允许的扩展名（QQ 图片/表情/语音）
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
VOICE_EXT = {".mp3", ".amr", ".silk", ".wav", ".m4a", ".ogg"}

MAX_BYTES = 8 * 1024 * 1024


@dataclass
class MediaItem:
    id: str
    kind: str
    file: str                     # 相对 media 目录的路径
    tags: List[str] = field(default_factory=list)
    desc: str = ""
    origin: str = "builtin"       # builtin | generated
    created_at: float = 0.0

    def to_dict(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "file": self.file,
                "tags": list(self.tags), "desc": self.desc,
                "origin": self.origin, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: Dict) -> "MediaItem":
        return cls(id=str(data.get("id") or ""), kind=str(data.get("kind") or ""),
                   file=str(data.get("file") or ""),
                   tags=[str(t) for t in (data.get("tags") or [])],
                   desc=str(data.get("desc") or ""),
                   origin=str(data.get("origin") or "builtin"),
                   created_at=float(data.get("created_at") or 0.0))

    @property
    def is_generated(self) -> bool:
        return self.origin == "generated"


def normalize_tags(raw) -> List[str]:
    """把 "困, 累；睡觉 打哈欠" 这类输入切成干净标签列表。"""
    if isinstance(raw, (list, tuple, set)):
        parts = [str(x) for x in raw]
    else:
        parts = re.split(r"[,，;；、\s/|]+", str(raw or ""))
    seen: List[str] = []
    for part in parts:
        tag = part.strip().lower()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


class MediaLibrary:
    """素材库：加载、查询、增删、按话题挑选。"""

    def __init__(self, root: str):
        # 必须是绝对路径：CQ 码里用相对路径会变成 file:///media/...，
        # 那会被解析成"主机名 media"，NapCat 根本找不到文件。
        self.root = Path(root).expanduser().absolute()
        self.manifest_path = self.root / "manifest.json"
        self._items: List[MediaItem] = []
        self._loaded = False

    # -- 存取 ----------------------------------------------------------
    def load(self) -> List[MediaItem]:
        self._items = []
        if self.manifest_path.is_file():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for entry in (data.get("items") or []):
                    item = MediaItem.from_dict(entry)
                    if item.id and item.kind:
                        self._items.append(item)
            except (json.JSONDecodeError, OSError):
                self._items = []
        self._loaded = True
        return self._items

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"items": [i.to_dict() for i in self._items]}
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    # -- 查询 ----------------------------------------------------------
    def all(self, kind: Optional[str] = None) -> List[MediaItem]:
        self._ensure()
        if kind is None:
            return list(self._items)
        return [i for i in self._items if i.kind == kind]

    def get(self, item_id: str) -> Optional[MediaItem]:
        self._ensure()
        return next((i for i in self._items if i.id == item_id), None)

    def path_of(self, item: MediaItem) -> Path:
        return self.root / item.file

    def exists(self, item: MediaItem) -> bool:
        return self.path_of(item).is_file()

    # -- 增删 ----------------------------------------------------------
    def add(self, kind: str, filename: str, data: bytes, tags, desc: str = "",
            origin: str = "builtin") -> MediaItem:
        if kind not in KINDS:
            raise ValueError(f"未知素材类型：{kind}")
        suffix = Path(filename).suffix.lower()
        allowed = VOICE_EXT if kind == VOICE else IMAGE_EXT
        if suffix not in allowed:
            raise ValueError(f"不支持的格式 {suffix}（允许：{'、'.join(sorted(allowed))}）")
        if len(data) > MAX_BYTES:
            raise ValueError(f"文件过大（{len(data) // 1024 // 1024}MB > "
                             f"{MAX_BYTES // 1024 // 1024}MB）")
        subdir = self.root / kind
        subdir.mkdir(parents=True, exist_ok=True)
        item_id = f"{kind}_{uuid.uuid4().hex[:10]}"
        target = subdir / f"{item_id}{suffix}"
        target.write_bytes(data)
        item = MediaItem(id=item_id, kind=kind, file=str(target.relative_to(self.root)),
                         tags=normalize_tags(tags), desc=desc, origin=origin,
                         created_at=time.time())
        self._ensure()
        self._items.append(item)
        self.save()
        return item

    def remove(self, item_id: str) -> bool:
        self._ensure()
        item = self.get(item_id)
        if item is None:
            return False
        try:
            self.path_of(item).unlink(missing_ok=True)
        except OSError:
            pass
        self._items = [i for i in self._items if i.id != item_id]
        self.save()
        return True

    # -- 挑选 ----------------------------------------------------------
    def match_tags(self, text: str, kind: Optional[str] = None) -> List[MediaItem]:
        """返回标签出现在文本里的素材（只做匹配，不做随机兜底）。"""
        self._ensure()
        lowered = (text or "").lower()
        if not lowered:
            return []
        hits = []
        for item in self._items:
            if kind is not None and item.kind != kind:
                continue
            if any(tag and tag in lowered for tag in item.tags):
                hits.append(item)
        return hits

    def pick(self, text: str, kind: Optional[str] = None,
             rng=None) -> Optional[MediaItem]:
        """按话题选一个素材；没有匹配就返回 None。

        绝不"随便挑一张"——V4 要求不要发与话题无关的图片。
        """
        hits = [i for i in self.match_tags(text, kind) if self.exists(i)]
        if not hits:
            return None
        if rng is not None and len(hits) > 1:
            return hits[rng.randrange(len(hits))]
        return hits[0]

    # -- 提示词渲染 -----------------------------------------------------
    def catalog_text(self, allow_sticker: bool = True, allow_image: bool = True,
                     allow_voice: bool = False, limit: int = 40) -> str:
        """渲染成给模型看的素材清单。

        只列"标签+说明"，不列文件路径——路径属于内部实现细节，不该进提示词。
        """
        self._ensure()
        lines: List[str] = []
        if allow_sticker:
            stickers = [i for i in self._items if i.kind == STICKER][:limit]
            if stickers:
                lines.append("QQ 表情包（按话题标签选择）：")
                for item in stickers:
                    label = item.desc or "表情"
                    tags = "、".join(item.tags) if item.tags else "无标签"
                    lines.append(f"  - {label}｜标签：{tags}")
        if allow_image:
            images = [i for i in self._items if i.kind == IMAGE][:limit]
            if images:
                lines.append("已有图片素材：")
                for item in images:
                    label = item.desc or "图片"
                    tags = "、".join(item.tags) if item.tags else "无标签"
                    origin = "（AI 生成，非官方素材）" if item.is_generated else ""
                    lines.append(f"  - {label}{origin}｜标签：{tags}")
        if allow_voice:
            voices = [i for i in self._items if i.kind == VOICE][:limit]
            if voices:
                lines.append("已有语音素材：")
                for item in voices:
                    lines.append(f"  - {item.desc or '语音'}｜标签：{'、'.join(item.tags) or '无标签'}")
        if not lines:
            lines.append("（当前没有可用素材；不要声称已发送图片或语音）")
        return "\n".join(lines)


def build_library(config) -> Optional[MediaLibrary]:
    """按配置构建素材库；未启用时返回 None。"""
    settings = getattr(config, "media", None)
    if settings is None or not settings.enabled:
        return None
    library = MediaLibrary(settings.path)
    try:
        library.load()
    except Exception:
        return library
    return library


# ----------------------------------------------------------------------
# 素材抓取：从 OneBot 消息段里取出可入库的媒体
#
# 这两件事放在这里而不是适配层，是因为它们曾在适配层里悄悄坏掉：
#   1. 只认本地文件路径 —— 而 NapCat 给照片的是 http URL，照片全被跳过；
#   2. 只认 image/record 段 —— 而 QQ 表情是 face 段，完全不进分支。
# 适配层（if HAS_NONEBOT 块内）在没装 NoneBot 的环境里根本测不到，
# 所以判定逻辑必须沉到这里，才能被单元测试覆盖。
# ----------------------------------------------------------------------

FACE_REASON = "qq_face"

_REASON_TEXT = {
    FACE_REASON: "这是 QQ 内置表情，拿不到图片文件，存不进素材库",
    "unsupported": "这种类型的消息不能作为素材",
    "no_source": "消息里没有可取的文件地址",
}


def classify_segment(segment_type: str, data: dict):
    """判断一个消息段能否作为素材入库。

    返回 (kind, source, reason)：
    - 可入库：(kind, source, "")
    - 不可入库：(None, None, reason)，reason 见 _REASON_TEXT
    """
    seg = str(segment_type or "")
    data = data or {}
    if seg == "face":
        return None, None, FACE_REASON
    if seg not in ("image", "record"):
        return None, None, "unsupported"
    source = str(data.get("url") or data.get("file") or "")
    if not source:
        return None, None, "no_source"
    return ("voice" if seg == "record" else "sticker"), source, ""


def reason_text(reason: str) -> str:
    return _REASON_TEXT.get(reason, reason or "未知原因")


def read_media_bytes(source: str, timeout: int = 20, retries: int = 2) -> bytes:
    """读取媒体字节：base64://、本地路径、http(s) URL 都支持。

    必须支持 URL —— 真机日志显示 NapCat 给照片的 file 字段是 http 链接，
    早期只认本地文件，导致所有照片被静默跳过。

    **也必须支持 `base64://`**：OneBot v11 规范里 image / record / video
    三段的 `file` 字段都允许这个形式（见规范里"发送时 file 还支持
    base64://…"那段）。而 NapCat 配置里如果 enableLocalFile2Url 为 false，
    它就更可能直接给 base64 而不是可访问的网址。
    实测过不支持时的表现：返回 0 字节，于是**图片、语音、视频三种模态
    全部静默失效**，而且没有任何报错——只有"它看不见我发的东西"。

    显式禁用系统代理：这台机器的 Windows 代理指向一个没在运行的端口，
    走代理会直接失败；而 QQ 的图片 CDN 在国内，直连即可。
    """
    if not source:
        return b""
    text = str(source)

    # base64 形式：`base64://<数据>`。注意数据里可能含 `+` `/` `=`，
    # 也可能带换行，所以解码前先把空白去掉。
    if text.startswith("base64://"):
        payload = text[len("base64://"):]
        payload = "".join(payload.split())
        try:
            import base64 as _b64
            data = _b64.b64decode(payload, validate=False)
        except Exception:
            return b""
        return data[:MAX_BYTES] if len(data) > MAX_BYTES else data

    path = text
    if path.startswith("file:///"):
        path = path[8:]
    elif path.startswith("file://"):
        path = path[7:]
    try:
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(MAX_BYTES)
    except OSError:
        pass
    if not text.lower().startswith(("http://", "https://")):
        return b""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/124.0 Safari/537.36")}
    for attempt in range(max(1, retries)):
        try:
            request = urllib.request.Request(source, headers=headers)
            with opener.open(request, timeout=timeout) as response:
                return response.read(MAX_BYTES)
        except Exception:
            if attempt + 1 < max(1, retries):
                time.sleep(1)
    return b""


def store_media(library, source: str, kind: str, blob: bytes, tags, desc: str,
                hint_name: str = ""):
    """把字节存进素材库。返回 MediaItem。

    hint_name 是消息段里的原始文件名（如 xxx.jpg）——必须传，
    否则 QQ 那种不带扩展名的 URL 会导致"认不出格式"而入库失败。
    """
    name = guess_media_name(source, kind, blob, hint_name)
    return library.add(kind, name, blob, tags=tags, desc=desc, origin="builtin")

# ----------------------------------------------------------------------
# 格式嗅探：URL 常常不带扩展名，必须能从字节认出来
#
# 真机踩过：QQ 图片的 URL 是 https://multimedia.nt.qq.com.cn/download?appid=...
# 路径里没有扩展名，按文件名推出来是 "download"，被格式校验直接拒掉
# （ValueError: 不支持的格式）。而真正的文件名其实在同一条消息的 file 字段里。
# ----------------------------------------------------------------------

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)
_AUDIO_MAGIC = (
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"#!AMR", ".amr"),
    (b"#!SILK", ".silk"),
    (b"\x02#!SILK", ".silk"),
    (b"OggS", ".ogg"),
)


def sniff_extension(blob: bytes, kind: str = IMAGE) -> str:
    """从字节判断扩展名；认不出返回空串。

    优先级低于真实文件名——只在文件名给不出可用后缀时才用。
    """
    if not blob or len(blob) < 12:
        return ""
    head = blob[:16]
    for magic, ext in (_AUDIO_MAGIC if kind == VOICE else _IMAGE_MAGIC):
        if head.startswith(magic):
            return ext
    if head.startswith(b"RIFF"):
        # RIFF 容器：WAV 与 WebP 共用，看偏移 8
        if blob[8:12] == b"WEBP":
            return ".webp"
        if blob[8:12] == b"WAVE":
            return ".wav"
        return ".wav" if kind == VOICE else ".webp"
    if b"ftyp" in blob[4:12]:
        return ".m4a" if kind == VOICE else ".png"
    return ""


def segment_hint_name(data: dict) -> str:
    """从消息段里取出"原始文件名"，用来保留扩展名。

    QQ 的 image 段同时给 file（如 xxx.jpg）和 url（无扩展名），
    文件名要从 file 取，字节要从 url 下——两者不能混。
    """
    data = data or {}
    return str(data.get("file") or data.get("filename") or "")


def guess_media_name(source: str, kind: str, blob: bytes, hint_name: str = "") -> str:
    """给素材起一个带合法扩展名的文件名。

    顺序：原始文件名 -> 来源路径 -> 字节嗅探。
    三者都认不出则抛 ValueError（由调用方如实上报，不静默丢弃）。
    """
    allowed = VOICE_EXT if kind == VOICE else IMAGE_EXT
    for candidate in (hint_name, source):
        raw = str(candidate or "").split("?")[0].split("#")[0]
        ext = os.path.splitext(raw)[1].lower()
        if ext in allowed:
            stem = os.path.basename(raw) or kind
            return stem
    sniffed = sniff_extension(blob, kind)
    if sniffed:
        return kind + "_" + str(int(time.time())) + sniffed
    raise ValueError(
        "认不出这个素材的格式（文件名没有后缀，字节也不是常见图片/音频格式）——"
        "请换一张普通图片，或先转成 png/jpg 再发")



def cq_code_for(outcome, exists=None) -> str:
    """把 Outcome.media 转成 OneBot 的 CQ 码字符串。

    **为什么这段逻辑在核心包、而不是适配层**：适配层整块包在
    `if HAS_NONEBOT` 里，没装 NoneBot 时那个模块根本不存在，
    于是**没法离线测试**。而这里的处理恰恰是只在真机上暴露过问题的地方。
    项目里 media.py 的其它函数也是同一个理由放在这里的
    （见 tests/test_media_capture.py 的模块说明）。

    两个踩过的坑都由这里兜住：

    1. **必须转绝对路径**。`file:///` 后面如果不是 `/D:/...` 这种形式，
       会被当成主机名，媒体**静默发不出去**。
    2. **逗号与方括号必须转义**。CQ 码用逗号分隔参数、方括号定界，
       路径里带这些字符会把整条 CQ 码解析错。

    表情包和图片都用 `[CQ:image]`（QQ 表情包本质就是图片），
    语音用 `[CQ:record]`。文件不存在就跳过——绝不能因为一个坏路径
    让整条回复都发不出去。

    `exists` 可注入（默认用 os.path.isfile），便于测试构造两种情形。
    """
    check = exists or os.path.isfile
    parts = []
    for item in getattr(outcome, "media", []) or []:
        path = str(getattr(item, "path", "") or "")
        if not path:
            continue
        path = os.path.abspath(path)
        try:
            if not check(path):
                continue
        except OSError:
            continue
        uri = "file:///" + path.replace("\\", "/").lstrip("/")
        uri = uri.replace(",", "%2C").replace("[", "%5B").replace("]", "%5D")
        tag = "record" if str(getattr(item, "kind", "")) == "voice" else "image"
        parts.append("[CQ:%s,file=%s]" % (tag, uri))
    return "".join(parts)
