"""图片理解：把用户发来的图交给视觉模型，拿回描述与文字。

**这个模块补的是项目里最大的一个名不副实的地方**：
在此之前，用户发来的图片会被归为 `sticker` 存进素材库，
但**从来没有送给模型看过**——所以它能"回声"式地把图发回去，
却看不懂图里是什么。`Message = Dict[str, str]` 这个类型注解
也装不下 OpenAI 多模态消息的 `content` 数组。

设计取舍（都是被本项目既有约束逼出来的）：

1. **走云端 OpenAI 兼容接口，不做本地推理。**
   核心包必须零第三方依赖（715 项测试靠这一点才能离线跑），
   所以不能为了看图去 import transformers/torch。
   视觉模型通过一个 HTTP 端点接入，端点可配——换成哪家都行。

2. **它是一个"感知"步骤，不是"工具"。**
   工具是模型自己决定要不要调的（比如要不要联网查）。
   但"用户发了图"这件事是**事实**，模型没法"决定不看"——
   它得先知道图里有什么，才有得选。所以视觉结果直接注入对话上下文，
   而不是挂成一个 `describe_image` 工具等模型来调。

3. **没配就没配，如实说。** 没有视觉模型时要明确告诉模型
   "你现在看不到图"，这样它会说"我看不到图片内容"，
   而不是根据文件名或者用户的话瞎猜图里有什么。
   这一点和项目里其它护栏是同一个原则。

配置（全部可选，不配就退化成"看不见"）：

    VISION_ENABLED=1
    VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    VISION_API_KEY=sk-xxx
    VISION_MODEL=qwen-vl-max
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import net

# 一次请求里最多看几张图。
# 不设上限的话，有人连发 9 张图就会产生 9 倍成本与延迟；
# 而且群聊里刷图的场景很常见。
MAX_IMAGES_PER_TURN = 3

# 单张图的字节上限。QQ 的图常见几 MB，超过这个就不看了——
# base64 之后体积还要再涨 1/3，太大既费流量也容易超时。
MAX_IMAGE_BYTES = 6 * 1024 * 1024

VISION_PROMPT = (
    "请描述这张图片的内容。要求：\n"
    "1. 如果有文字（截图、表情包文字、弹幕等），逐字抄下来，不要改写；\n"
    "2. 描述画面里发生了什么，谁在做什么；\n"
    "3. 不要猜测图片之外的背景，不确定的地方直接说不确定；\n"
    "4. 用简洁的中文，控制在 200 字以内。"
)


@dataclass
class ImageRef:
    """一张待理解的图片。

    `url` 可以是远程地址，也可以是 data URI。
    `path` 是本地文件——适配层拿到的常常是本地缓存路径。
    两者至少有其一。`blob` 是已经读到的字节，优先级最高
    （避免同一个文件被读两次）。
    """

    url: str = ""
    path: str = ""
    blob: bytes = b""
    kind: str = "image"          # image / sticker
    source_hint: str = ""        # 原始文件名之类，仅用于日志

    def to_data_uri(self) -> str:
        """转成 OpenAI 多模态协议能吃的 data URI。失败返回空串。"""
        if self.url.startswith("data:"):
            return self.url
        data = self.blob
        if not data and self.path:
            try:
                with open(self.path, "rb") as fh:
                    data = fh.read(MAX_IMAGE_BYTES + 1)
            except OSError:
                return ""
        if not data or len(data) > MAX_IMAGE_BYTES:
            return ""
        mime = _guess_mime(data)
        return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))

    def to_url_or_uri(self) -> str:
        """优先用原始 http(s) 地址（省一次编码），否则转 data URI。"""
        if self.url.startswith(("http://", "https://", "data:")):
            return self.url
        return self.to_data_uri()


def _guess_mime(data: bytes) -> str:
    """按魔数判断图片类型。判不出来就按 png 处理。"""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"BM":
        return "image/bmp"
    return "image/png"


@dataclass
class VisionResult:
    ok: bool
    description: str = ""
    error: str = ""
    provider: str = ""
    model: str = ""
    images_seen: int = 0
    elapsed: float = 0.0
    skipped: int = 0            # 因为超过张数上限而被丢弃的图
    note: str = ""              # 给用户看的补充说明（如"只看了前 3 张"）


class VisionClient:
    """调用 OpenAI 兼容的视觉接口。

    不绑定任何一家厂商：`qwen-vl`、`glm-4v`、`kimi`、本地 ollama
    都提供同样的 `/chat/completions` + `image_url` 协议，
    换个 base_url 和 model 就能切。
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 60, max_tokens: int = 400):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def health(self) -> Dict[str, Any]:
        return {"configured": self.configured, "model": self.model,
                "base_url": self.base_url}

    def describe(self, images: List[ImageRef],
                 prompt: str = VISION_PROMPT) -> VisionResult:
        """让视觉模型描述这批图片。"""
        started = time.time()
        if not self.configured:
            return VisionResult(False, error="没有配置视觉模型",
                                note="当前看不到图片内容")
        if not images:
            return VisionResult(False, error="没有图片")

        usable: List[str] = []
        for item in images[:MAX_IMAGES_PER_TURN]:
            uri = item.to_url_or_uri()
            if uri:
                usable.append(uri)
        if not usable:
            return VisionResult(
                False, error="图片读取失败",
                note="图片可能是格式不支持或者超过体积上限（%.0f MB）"
                     % (MAX_IMAGE_BYTES / 1048576))

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for uri in usable:
            content.append({"type": "image_url", "image_url": {"url": uri}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,          # 描述事实，不要发散
        }
        url = "%s/chat/completions" % self.base_url
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", "Bearer %s" % self.api_key)
        try:
            with net.open(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            return VisionResult(False, error="视觉接口返回 %s: %s" % (exc.code, detail),
                                provider="openai-compatible", model=self.model,
                                elapsed=time.time() - started)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return VisionResult(False, error="视觉接口连不上：%s" % exc,
                                provider="openai-compatible", model=self.model,
                                elapsed=time.time() - started)

        try:
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = (message.get("content") or "").strip()
        except (AttributeError, IndexError):
            text = ""
        if not text:
            return VisionResult(False, error="视觉接口没有返回内容",
                                provider="openai-compatible", model=self.model,
                                elapsed=time.time() - started)

        skipped = max(0, len(images) - len(usable))
        note = ""
        if skipped:
            note = "（一次最多看 %d 张，后面 %d 张没看）" % (MAX_IMAGES_PER_TURN, skipped)
        return VisionResult(True, description=text, provider="openai-compatible",
                            model=self.model, images_seen=len(usable),
                            elapsed=time.time() - started, skipped=skipped, note=note)


def build_client(config) -> Optional[VisionClient]:
    """按配置造一个客户端；没配置就返回 None。

    参数是 **VisionConfig**（`config.vision`），所以字段名是
    `enabled` / `base_url` / `model`，**不带 `vision_` 前缀**。

    我第一版写成了 `getattr(config, "vision_enabled")` —— 那是把
    AppConfig 的字段命名习惯套到了子配置上。更糟的是 `getattr` 带了默认值，
    取不到就静默返回 False，于是 build_client 永远返回 None，
    而**没有任何报错**。这个错和项目里那个"漏传 tools 参数"
    是同一类：防御性写法把真正的错误盖住了。
    """
    if not getattr(config, "enabled", False):
        return None
    base = getattr(config, "base_url", "") or ""
    key = getattr(config, "api_key", "") or ""
    model = getattr(config, "model", "") or ""
    client = VisionClient(base, key, model,
                          timeout=int(getattr(config, "timeout", 60) or 60),
                          max_tokens=int(getattr(config, "max_tokens", 400) or 400))
    return client if client.configured else None


def render_for_model(result: VisionResult, mode_is_roleplay: bool = True) -> str:
    """把视觉结果渲染成能注入提示词的一段文字。

    **没成功的时候也要注入一段话**，而且要明确说明"看不到"。
    这是刻意的：如果不注入任何东西，模型会以为没有图片这回事，
    于是要么忽略图、要么根据用户的话瞎猜图里有什么。
    明确告诉它"你这次看不到图"，它才会如实说看不到。
    """
    if not result.ok:
        why = result.error or "未知原因"
        return ("（用户这次发了图片，但**你没有看到图片内容**——%s。"
                "请如实告诉用户你看不到，不要猜测图里是什么。）" % why)
    head = "（以下是用户所发图片的内容，由视觉模型识别，可能有误差）"
    if mode_is_roleplay:
        head = ("（以下是用户刚发的图片的内容，由视觉模型识别。"
                "请用你的口吻自然地回应，但**只依据这段描述**，"
                "不要补充描述里没有的画面细节。识别可能有误差，"
                "涉及关键信息时可以说明这是识别结果。）")
    body = result.description
    if result.note:
        body = body + "\n" + result.note
    return head + "\n" + body
