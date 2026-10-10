"""AI 图片生成与语音合成（V4）。

两个客户端都走 OpenAI 兼容接口，零第三方依赖：
- 图片：POST {base}/images/generations  -> b64_json 或 url
- 语音：POST {base}/audio/speech       -> 音频字节

诚实原则（V4 明确要求）：
- 失败一律如实返回，绝不假装"已发送"；
- 生成的图片标记为 AI 生成，绝不说是官方素材（来源字段在 media.MediaItem.origin）。
"""
from __future__ import annotations

from . import net
import base64
import io
import json
import os
import re
import shutil
import struct
import subprocess
import wave

from .asr import build_multipart
from .ffmpeg import find_ffmpeg
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ImageResult:
    ok: bool
    data: bytes = b""
    mime: str = "image/png"
    provider: str = "openai_compatible"
    error: str = ""
    revised_prompt: str = ""


@dataclass
class SpeechResult:
    ok: bool
    data: bytes = b""
    mime: str = "audio/mpeg"
    provider: str = "openai_compatible"
    error: str = ""
    # 情感相关：emotion_applied 为 True 才表示情感真的被表达出来了。
    # 免费 TTS（百度/Google）没有情感参数，这时必须如实标 False，
    # 不能因为"设置了情感"就让上层以为语音带上了情绪。
    emotion_applied: bool = False
    emotion: str = ""
    emotion_note: str = ""


def _post_json(url: str, api_key: str, payload: dict, timeout: int) -> Tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")
    with net.open(request, timeout=timeout) as response:
        return response.status, response.read(12 * 1024 * 1024)


def _error_text(exc: Exception, what: str) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().decode("utf-8", "ignore")[:200]
        hint = ""
        if exc.code in (401, 403):
            hint = f"（{what}的 API Key 无效或无权访问）"
        elif exc.code == 429:
            hint = f"（{what}频率或额度限制）"
        elif exc.code == 404:
            hint = f"（{what}接口地址不存在，请检查 BASE_URL）"
        return f"HTTP {exc.code}{hint}: {detail}"
    return f"{type(exc).__name__}: {exc}"


class ImageClient:
    """AI 图片生成客户端。"""

    def __init__(self, base_url: str, api_key: str, model: str = "dall-e-3",
                 size: str = "1024x1024", timeout: int = 120):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.size = size
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def generate(self, prompt: str) -> ImageResult:
        if not self.available:
            return ImageResult(False, error="未配置图片生成（需要 IMAGE_GEN_API_KEY 与 BASE_URL）")
        if not (prompt or "").strip():
            return ImageResult(False, error="图片描述为空")
        payload = {"model": self.model, "prompt": prompt, "n": 1, "size": self.size}
        try:
            status, raw = _post_json(f"{self.base_url}/images/generations",
                                     self.api_key, payload, self.timeout)
        except Exception as exc:
            return ImageResult(False, error=_error_text(exc, "图片生成"))
        try:
            body = json.loads(raw.decode("utf-8", "ignore"))
        except json.JSONDecodeError:
            return ImageResult(False, error="图片生成返回了非 JSON 内容")
        entries = body.get("data") or []
        if not entries:
            return ImageResult(False, error=f"图片生成没有返回数据：{str(body)[:150]}")
        first = entries[0] or {}
        revised = str(first.get("revised_prompt") or "")
        if first.get("b64_json"):
            try:
                data = base64.b64decode(first["b64_json"])
            except Exception:
                return ImageResult(False, error="图片 base64 解码失败")
            return ImageResult(True, data=data, revised_prompt=revised)
        url = str(first.get("url") or "")
        if url:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "alice-qq-agent"})
                with net.open(request, timeout=self.timeout) as response:
                    return ImageResult(True, data=response.read(12 * 1024 * 1024),
                                       mime=response.headers.get("Content-Type") or "image/png",
                                       revised_prompt=revised)
            except Exception as exc:
                return ImageResult(False, error=f"下载生成图片失败：{_error_text(exc, '图片')}")
        return ImageResult(False, error="图片生成返回里既没有 b64_json 也没有 url")


class OpenAITTSClient:
    """OpenAI 兼容的语音合成（/audio/speech）。

    需要 API Key。适合已有付费 TTS 服务的情况。
    V4 要求：不声称使用官方配音、不声称复制了官方演员声音。
    """
    name = "openai_compatible"
    # gpt-4o-mini-tts 等模型支持 instructions 控制语气
    supports_emotion = True

    def __init__(self, base_url: str, api_key: str, model: str = "tts-1",
                 voice: str = "alloy", fmt: str = "mp3", timeout: int = 120):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.voice = voice
        self.fmt = fmt or "mp3"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        if not self.available:
            return SpeechResult(False, error="未配置语音合成（需要 TTS_API_KEY 与 BASE_URL）")
        if not (text or "").strip():
            return SpeechResult(False, error="语音文本为空")
        payload = {"model": self.model, "input": text, "voice": self.voice,
                   "response_format": self.fmt}
        # 情感靠 instructions 传递（gpt-4o-mini-tts 等模型支持）
        from .emotion import LABELS as _EMO_LABELS
        applied = False
        if emotion:
            label = _EMO_LABELS.get(emotion, emotion)
            payload["instructions"] = f"请用{label}的语气说这段话。"
            applied = True
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/audio/speech", data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with net.open(request, timeout=self.timeout) as response:
                data = response.read(12 * 1024 * 1024)
        except Exception as exc:
            return SpeechResult(False, error=_error_text(exc, "语音合成"))
        if not data:
            return SpeechResult(False, error="语音合成返回空内容")
        mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus",
                "aac": "audio/aac", "flac": "audio/flac"}.get(self.fmt, "audio/mpeg")
        note = "" if applied else ""
        if emotion and not applied:
            note = "该模型可能不支持 instructions，语气控制未必生效"
        return SpeechResult(True, data=data, mime=mime,
                            emotion=emotion, emotion_applied=applied, emotion_note=note)


def build_image_client(config) -> Optional[ImageClient]:
    settings = getattr(config, "media", None)
    if settings is None or not settings.image_gen_enabled:
        return None
    if not (settings.image_gen_api_key and settings.image_gen_base_url):
        return None
    return ImageClient(settings.image_gen_base_url, settings.image_gen_api_key,
                       settings.image_gen_model, settings.image_gen_size)


class BaiduTTSClient:
    """百度翻译的免费 TTS（无需 Key）。

    实测（本机）：HTTP 200，返回 audio/mpeg，约 19KB / 短句。
    优点：免费、无需注册、国内可达。
    缺点：音色是通用女声，非爱丽丝风格；长文本需切分。
    """
    name = "baidu"
    ENDPOINT = "https://fanyi.baidu.com/gettts"

    def __init__(self, language: str = "zh", speed: int = 5, timeout: int = 30):
        self.language = language or "zh"
        self.speed = max(1, min(9, int(speed or 5)))
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return True   # 无需凭据

    supports_emotion = False   # 百度端点没有情感参数

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        body = (text or "").strip()
        if not body:
            return SpeechResult(False, provider=self.name, error="语音文本为空")
        query = urllib.parse.urlencode({
            "lan": self.language, "text": body,
            "spd": str(self.speed), "source": "web"})
        request = urllib.request.Request(
            f"{self.ENDPOINT}?{query}",
            headers={"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/124.0 Safari/537.36"),
                     "Referer": "https://fanyi.baidu.com/"})
        try:
            with net.open(request, timeout=self.timeout) as response:
                data = response.read(6 * 1024 * 1024)
        except Exception as exc:
            return SpeechResult(False, provider=self.name, error=_error_text(exc, "百度语音"))
        if not data or not _looks_like_audio(data):
            return SpeechResult(False, provider=self.name,
                                error=f"返回内容不是音频（{len(data)} 字节）")
        note = ""
        if emotion:
            note = f"{self.name} 不支持情感参数，本次语气为默认"
        return SpeechResult(True, data=data, provider=self.name,
                            emotion=emotion, emotion_applied=False, emotion_note=note)


class GoogleTTSClient:
    """Google 翻译的免费 TTS（无需 Key）。

    实测（本机）：HTTP 200，audio/mpeg。同样不保证长期可用、也不保证所有网络可达。
    """
    name = "google"
    ENDPOINT = "https://translate.google.com/translate_tts"

    def __init__(self, language: str = "zh-CN", timeout: int = 30):
        self.language = language or "zh-CN"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return True

    supports_emotion = False   # Google 端点没有情感参数

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        body = (text or "").strip()
        if not body:
            return SpeechResult(False, provider=self.name, error="语音文本为空")
        # Google 这个端点对长文本会截断，所以这里也做长度保护
        body = body[:200]
        query = urllib.parse.urlencode({"ie": "UTF-8", "q": body,
                                        "tl": self.language, "client": "tw-ob"})
        request = urllib.request.Request(
            f"{self.ENDPOINT}?{query}",
            headers={"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/124.0 Safari/537.36"),
                     "Referer": "https://translate.google.com/"})
        try:
            with net.open(request, timeout=self.timeout) as response:
                data = response.read(6 * 1024 * 1024)
        except Exception as exc:
            return SpeechResult(False, provider=self.name, error=_error_text(exc, "Google 语音"))
        if not data or not _looks_like_audio(data):
            return SpeechResult(False, provider=self.name,
                                error=f"返回内容不是音频（{len(data)} 字节）")
        note = ""
        if emotion:
            note = f"{self.name} 不支持情感参数，本次语气为默认"
        return SpeechResult(True, data=data, provider=self.name,
                            emotion=emotion, emotion_applied=False, emotion_note=note)


def _looks_like_audio(data: bytes) -> bool:
    """粗略判断是不是音频字节。

    实测踩过：只看 b"ID3" 或 ÿû 会漏判——百度返回的帧头是 ÿó，
    只判断前 3 字节相等会失败。这里按 MPEG 帧同步位判断。
    """
    if len(data) < 4:
        return False
    if data[:3] == b"ID3":
        return True
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True                      # MPEG 音频帧同步
    if b"ftyp" in data[:32]:
        return True                      # MP4/M4A
    if data[:4] == b"RIFF":
        return True                      # WAV
    if data[:3] == b"Ogg":
        return True
    return False


class GPTSoVITSTTSClient:
    """接自建 GPT-SoVITS 服务（本地 HTTP），按官方 api_v2 契约。

    **混合中日配音色的正确做法**是用官方参数 aux_ref_audio_paths
    （多说话人音色融合），而不是把两个路径用 | 拼起来——
    后者在官方 api_v2 上不会生效（实测读文档才发现）。

    契约：
        POST {base}/tts  JSON 或 GET {base}/tts?query
        text / text_lang / ref_audio_path / prompt_text / prompt_lang
        aux_ref_audio_paths = [第二个参考音频]      # 音色融合
        media_type = wav|mp3
    成功返回音频字节；失败返回 JSON + HTTP 400。
    """
    name = "gptsovits"

    def __init__(self, base_url: str, ref_audio: str = "", prompt_text: str = "",
                 text_lang: str = "zh", prompt_lang: str = "zh",
                 media_type: str = "wav", timeout: int = 60, use_post: bool = True,
                 ref_audio_secondary: str = "", emotion_refs=None,
                 speed_factor: float = 1.0, seed: int = -1,
                 text_split_method: str = "cut5"):
        self.base_url = (base_url or "").rstrip("/")
        self.ref_audio = ref_audio or ""
        self.prompt_text = prompt_text or ""
        self.text_lang = text_lang or "zh"
        self.prompt_lang = prompt_lang or "zh"
        # 官方 api_v2 只支持 wav/raw/ogg/aac（见 api_v2.py:338），**不支持 mp3**。
        # 而共享的 TTS_FORMAT 默认是 mp3（对 baidu/google 是对的），
        # 直接透传会返回 400 "media_type: mp3 is not supported"。
        # 这里自己兜底，而不是要求用户为每个 provider 改配置。
        wanted = (media_type or "wav").strip().lower()
        self.media_type = wanted if wanted in ("wav", "raw", "ogg", "aac") else "wav"
        self.timeout = timeout
        self.use_post = use_post
        # 第二个参考音频 -> 走官方 aux_ref_audio_paths 做音色融合
        self.ref_audio_secondary = ref_audio_secondary or ""
        # 情感 -> {情感名: (参考音频路径, 该音频的文本)}
        self.emotion_refs = dict(emotion_refs or {})
        self.speed_factor = float(speed_factor or 1.0)
        self.seed = int(seed)
        self.text_split_method = text_split_method or "cut5"

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    @property
    def blends_references(self) -> bool:
        """是否配置了第二个参考音频（即会做音色融合）。"""
        return bool(self.ref_audio_secondary)

    @property
    def supports_emotion(self) -> bool:
        """有情感参考音频才能换语气。"""
        return bool(self.emotion_refs)

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        body = (text or "").strip()
        if not body:
            return SpeechResult(False, provider=self.name, error="语音文本为空")
        if not self.available:
            return SpeechResult(False, provider=self.name,
                                error="未配置克隆音色服务地址（TTS_CLONE_URL）")

        # 情感靠**换参考音频**实现：换成带该情绪的参考音频及其文本。
        # 官方 api_v2 没有 emo_ref 参数，这是在任何版本上都成立的做法。
        ref_audio, prompt_text = self.ref_audio, self.prompt_text
        applied = False
        if emotion and emotion in self.emotion_refs:
            entry = self.emotion_refs[emotion]
            if isinstance(entry, (tuple, list)):
                ref_audio, prompt_text = entry[0], (entry[1] if len(entry) > 1 else "")
            else:
                ref_audio = str(entry)
            applied = True

        payload = {
            "text": body,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text,
            "prompt_lang": self.prompt_lang,
            "media_type": self.media_type,
            "text_split_method": self.text_split_method,
            "speed_factor": self.speed_factor,
            "seed": self.seed,
        }
        # 官方参数 aux_ref_audio_paths：多说话人音色融合。
        #
        # 重复项的作用（**实测结论，不是官方文档承诺**）：
        # 官方没有任何权重参数，源码里 refer_spec = [主参考] + [每个 aux...]
        # 逐个 append 后交给模型（TTS.py:1246-1252）。
        # 实测让日配在 aux 里重复 1/2/3 次，用 MFCC 指纹测与两个参考的距离：
        #     距日配 0.1150 -> 0.1084 -> 0.1061 -> 0.1054   （单调变近）
        #     距中配 0.0425 -> 0.0478 -> 0.0509 -> 0.0528   （单调变远）
        # 所以重复**确实**单调地把音色推向那段参考，但它**不是校准过的比例旋钮**：
        # 即使重复 3 次，结果仍更接近主参考。
        # 想要真正"偏日配"，应该把日配设为主参考（那需要 prompt_lang=ja，
        # 也就需要 pyopenjtalk）。
        #
        # 注意：**不能靠混音实现**——两段不同内容的人声叠加会糊成听不清。
        # 这里支持逗号/竖线分隔的重复写法：
        #   TTS_CLONE_REF_AUDIO_2=D:\...\ja_ref.wav,D:\...\ja_ref.wav
        aux_refs = [p.strip() for p in re.split(r"[,|]", self.ref_audio_secondary or "")
                    if p.strip()]
        if aux_refs:
            payload["aux_ref_audio_paths"] = aux_refs

        try:
            if self.use_post:
                request = urllib.request.Request(
                    f"{self.base_url}/tts",
                    data=json.dumps(payload).encode("utf-8"), method="POST")
                request.add_header("Content-Type", "application/json")
            else:
                request = urllib.request.Request(
                    f"{self.base_url}/tts?{urllib.parse.urlencode(payload)}")
            with net.open(request, timeout=self.timeout) as response:
                data = response.read(12 * 1024 * 1024)
        except urllib.error.HTTPError as exc:
            # 官方约定：失败返回错误 json + 400
            detail = exc.read().decode("utf-8", "ignore")[:200]
            return SpeechResult(False, provider=self.name,
                                error=f"克隆音色服务返回 HTTP {exc.code}：{detail}")
        except Exception as exc:
            return SpeechResult(False, provider=self.name,
                                error=_error_text(exc, "克隆音色服务"))

        if not data or not _looks_like_audio(data):
            return SpeechResult(False, provider=self.name,
                                error=f"返回内容不是音频（{len(data)} 字节）")

        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                "aac": "audio/aac", "flac": "audio/flac"}.get(self.media_type, "audio/wav")
        note = ""
        if emotion and not applied:
            note = (f"未配置「{emotion}」的情感参考音频，本次语气由主参考决定")
        return SpeechResult(True, data=data, mime=mime, provider=self.name,
                            emotion=emotion, emotion_applied=applied, emotion_note=note)


def parse_blend(spec: str) -> dict:
    """解析 "alice_ja:0.7,alice_zh:0.3" 这类混合权重。

    返回 {模型名: 权重}。用于校验与展示——如果权重加起来不是 1，
    说明配置可能写错了（RVC 融合通常要求归一化）。
    """
    weights = {}
    for chunk in str(spec or "").replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, value = chunk.rpartition(":")
        name = name.strip()
        if not name:
            # 没写冒号，就当作"只有模型名、权重默认 1"
            weights[chunk] = 1.0
            continue
        try:
            weights[name] = float(value)
        except ValueError:
            raise ValueError(f"权重不是数字：{chunk!r}")
    return weights


def describe_voice_pipeline(settings) -> str:
    """渲染当前语音链路，供启动横幅展示；发现配置矛盾时明确提示。"""
    if settings is None or not settings.tts_enabled:
        return "语音合成 = 关闭"
    provider = getattr(settings, "tts_provider", "baidu")
    parts = [f"语音合成 = {provider}"]
    fallback = getattr(settings, "tts_fallback", "") or ""
    if fallback and fallback.lower() != str(provider).lower():
        parts.append(f"备用 = {fallback}（克隆服务不可用时自动退回）")
    # 澄清一个容易误解的点：GPT-SoVITS 的参考融合没有权重参数
    if (provider in ("gptsovits", "clone", "custom")
            and getattr(settings, "tts_clone_ref_audio_secondary", "")):
        parts.append("注意 = GPT-SoVITS 参考融合无权重参数，"
                     "要按比例混合需靠 RVC 模型融合或按比例准备微调数据集")
    blend_spec = getattr(settings, "tts_vc_blend", "") or ""
    vc_url = getattr(settings, "tts_vc_url", "") or ""
    if vc_url:
        parts.append(f"音色转换 = {vc_url}")
        if blend_spec:
            try:
                weights = parse_blend(blend_spec)
            except ValueError as exc:
                return "｜".join(parts + [f"混合权重解析失败：{exc}"])
            shown = " + ".join(f"{name} {w:.0%}" for name, w in weights.items())
            parts.append(f"音色混合 = {shown}")
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                parts.append(f"警告：混合权重合计为 {total:.2f}，通常应为 1.0")
    elif blend_spec:
        # 这是最容易踩的配置坑：写了权重却没写转换服务地址，权重根本不生效
        parts.append("警告：设置了 TTS_VC_BLEND 但没有 TTS_VC_URL，"
                     "混合权重不会生效（音色转换未启用）")

    # 情感能力必须如实展示：免费端点做不到，别让人误以为语音带情绪
    if getattr(settings, "tts_emotion_enabled", True):
        refs = parse_emotion_refs(getattr(settings, "tts_emotion_refs", ""))
        if provider in ("gptsovits", "clone", "custom"):
            if refs:
                parts.append("情感 = " + "、".join(sorted(refs.keys())))
            else:
                parts.append("情感 = 未配置情感参考音频（语气由主参考决定）")
        elif provider in ("openai_compatible", "openai"):
            parts.append("情感 = 通过 instructions 传递（取决于模型是否支持）")
        else:
            parts.append(f"情感 = {provider} 端点不支持情感参数，语气为默认")
    return "｜".join(parts)


def _fix_wav_sizes(blob: bytes) -> bytes:
    """修正 ffmpeg 管道输出的 WAV 长度字段。

    ffmpeg 写管道时无法回填长度，于是把 RIFF 和 data 的长度都写成 0xFFFFFFFF。
    这种文件在内存里能读，但**播放器可能直接拒收**——写进字节流发给 QQ
    就会出问题。所以按实际字节数把两个长度改回真实值。
    """
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob
    out = bytearray(blob)
    struct.pack_into("<I", out, 4, len(out) - 8)     # RIFF chunk size
    pos = 12
    while pos + 8 <= len(out):
        chunk_id = bytes(out[pos:pos + 4])
        size = struct.unpack_from("<I", out, pos + 4)[0]
        if chunk_id == b"data":
            struct.pack_into("<I", out, pos + 4, len(out) - pos - 8)
            break
        if size == 0xFFFFFFFF or size > len(out):
            break
        pos += 8 + size + (size & 1)
    return bytes(out)


class SentenceSplitTTS:
    """按句拆开、逐句**独立**合成、再拼接。

    **为什么需要**：GPT-SoVITS 在**一次请求里合成多句**时，同一段文本的前后句
    演绎强度会不一致——实测同一句话第一句与第二句的 F0 差 35 Hz，听起来像
    "后半段换了个声音"。参考音频、随机种子都没变，是模型跨句状态的问题。

    逐句独立合成，每句都从同样的干净状态开始，句间停顿由我们插入固定长度，
    一致性就完全可控了。

    拼接用标准库 wave + array（GPT-SoVITS 输出固定 32kHz 单声道 16bit），
    不额外依赖 ffmpeg。
    """
    name = "sentence_split"

    # 中英文句末标点都算句子边界
    _BOUNDARY = "。！？!?；;…\n"

    def __init__(self, inner, gap_seconds: float = 0.25, min_chars: int = 6):
        self.inner = inner
        self.gap_seconds = max(0.0, float(gap_seconds or 0.0))
        self.min_chars = max(1, int(min_chars or 1))

    @property
    def available(self) -> bool:
        return bool(getattr(self.inner, "available", False))

    def split(self, text: str):
        """按句末标点切分，标点跟着前一句。太短的碎片并回上一句。"""
        text = (text or "").strip()
        if not text:
            return []
        parts, current = [], ""
        for char in text:
            current += char
            if char in self._BOUNDARY:
                if current.strip():
                    parts.append(current.strip())
                current = ""
        if current.strip():
            parts.append(current.strip())
        # 极短句（如"唔。"）并回上一句，避免碎成太多段
        merged = []
        for part in parts:
            if merged and len(part) < self.min_chars:
                merged[-1] += part
            else:
                merged.append(part)
        return merged or [text]

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        sentences = self.split(text)
        # 只有一句就没什么可拆的，直接交给内层（少一次拼接损耗）
        if len(sentences) <= 1:
            try:
                return self.inner.synthesize(text, emotion)
            except TypeError:
                return self.inner.synthesize(text)

        blobs, mime, note = [], "audio/wav", ""
        for sentence in sentences:
            try:
                piece = self.inner.synthesize(sentence, emotion)
            except TypeError:
                piece = self.inner.synthesize(sentence)
            if not piece.ok:
                # 任何一句失败就如实报错，不用残缺结果冒充成功
                return SpeechResult(False, provider=self.name,
                                    error="第 %d 句合成失败：%s"
                                          % (len(blobs) + 1, piece.error))
            blobs.append(piece.data)
            mime = piece.mime or mime
            note = piece.emotion_note or note

        joined = _join_wavs(blobs, self.gap_seconds)
        if joined is None:
            # 拼接不了（格式不统一）就退回拼接前的最后一段，并说明
            return SpeechResult(True, data=blobs[-1], mime=mime,
                                provider=getattr(self.inner, "name", "?"),
                                error="逐句拼接失败，只返回最后一句",
                                emotion=emotion, emotion_note=note)
        return SpeechResult(True, data=joined, mime="audio/wav",
                            provider="%s+sentences(%d)"
                                     % (getattr(self.inner, "name", "?"), len(blobs)),
                            emotion=emotion, emotion_note=note)


def _join_wavs(blobs, gap_seconds: float):
    """把多个 WAV 字节按顺序拼起来，中间插入 gap_seconds 静音。

    要求格式一致（GPT-SoVITS 固定 32kHz/单声道/16bit）；不一致时返回 None，
    由调用方决定怎么降级——不猜、不硬拼。
    """
    if not blobs:
        return None
    if len(blobs) == 1:
        return blobs[0]
    try:
        metas, chunks = [], []
        for blob in blobs:
            with wave.open(io.BytesIO(blob)) as wf:
                metas.append((wf.getnchannels(), wf.getsampwidth(), wf.getframerate()))
                chunks.append(wf.readframes(wf.getnframes()))
        if len(set(metas)) != 1:
            return None
        channels, width, rate = metas[0]
        silence_frames = int(rate * gap_seconds)
        silence = b"\x00" * (silence_frames * channels * width)
        body = silence.join(chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(body)
        return buf.getvalue()
    except Exception:
        return None


class PitchShiftTTS:
    """在合成结果上做**变调不变速**，用于把音色推得更"高/夹/元气"。

    **为什么需要单独一层**：GPT-SoVITS 的输出音调会跟随参考音频，但有上限。
    实测参考 F0 提到 526 Hz 时，输出只到 399 Hz——模型不会完全跟随。
    想再往上，只能直接对输出变调。

    **实现**：ffmpeg 的 asetrate + aresample + atempo 组合——
    先按比率改采样率（音调变了，时长也跟着变），再用 atempo 把时长拉回来。
    净效果是**只改音调、不改语速**。

    找不到 ffmpeg 时不做任何处理，并如实在 error 里说明（不假装变过调）。
    """
    name = "pitch_shift"

    def __init__(self, inner, semitones: float = 0.0, ffmpeg: str = "",
                 timeout: int = 120):
        self.inner = inner
        self.semitones = float(semitones or 0.0)
        self.ffmpeg = ffmpeg or ""
        self.timeout = timeout

    def _ffmpeg_path(self):
        # 定位逻辑统一收在 aliceqq.ffmpeg 里（原先这里有一份**硬编码**路径的副本，
        # search 那边又因为没有这个能力而放弃了长视频识别）。
        return find_ffmpeg(explicit=self.ffmpeg)

    @property
    def available(self) -> bool:
        if abs(self.semitones) < 1e-6:
            return False
        if not self._ffmpeg_path():
            return False
        return bool(getattr(self.inner, "available", False))

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        # 情感传给内层（语气在合成阶段就定了，变调只改音高）
        try:
            base = self.inner.synthesize(text, emotion)
        except TypeError:
            base = self.inner.synthesize(text)
        if not base.ok:
            return base
        if abs(self.semitones) < 1e-6:
            return base
        shifted = self._shift(base.data)
        if shifted is None:
            # 诚实：变调失败就说清楚，不冒充成功
            return SpeechResult(base.ok, data=base.data, mime=base.mime,
                                provider=getattr(base, "provider", "?"),
                                error="变调失败，已使用未变调音频",
                                emotion=getattr(base, "emotion", emotion),
                                emotion_applied=getattr(base, "emotion_applied", False),
                                emotion_note=getattr(base, "emotion_note", ""))
        return SpeechResult(True, data=shifted, mime="audio/wav",
                            provider="%s+pitch%+.1f" % (getattr(base, "provider", "?"),
                                                        self.semitones),
                            emotion=getattr(base, "emotion", emotion),
                            emotion_applied=getattr(base, "emotion_applied", False),
                            emotion_note=getattr(base, "emotion_note", ""))

    def _shift(self, blob: bytes):
        exe = self._ffmpeg_path()
        if not exe or not blob:
            return None
        try:
            with wave.open(io.BytesIO(blob)) as wf:
                sr = wf.getframerate()
        except Exception:
            return None
        if not sr:
            return None
        ratio = 2.0 ** (self.semitones / 12.0)
        # asetrate 改音调（时长同时被压缩），atempo 把时长还原
        filters = "asetrate=%d,aresample=%d,atempo=%.6f" % (int(sr * ratio), sr, 1.0 / ratio)
        # 走 stdin/stdout 管道，**不碰文件系统**：
        # 早先用 TemporaryDirectory 写临时文件，但在受限环境下系统临时目录
        # 不可写（PermissionError: WinError 5），整个变调会失败。
        # 管道方式既避开权限问题，也少一次磁盘往返。
        try:
            proc = subprocess.run(
                [exe, "-y", "-loglevel", "error", "-i", "pipe:0",
                 "-filter:a", filters, "-f", "wav", "pipe:1"],
                input=blob, capture_output=True, timeout=self.timeout)
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout:
            return None
        return _fix_wav_sizes(proc.stdout)


class VoiceConvertedTTS:
    """先合成、再做音色转换（RVC / so-vits-svc / 同类）的包装层。

    **为什么要有这一层**（这是"两个 CV 音色混合"的正确做法）：

    直接用日配参考去克隆，说中文会带日式口音——因为发音习惯来自参考音频。
    正确顺序是：

        中文 TTS（保证中文发音正确） -> 音色转换模型（提供目标音色）

    这样发音与音色解耦。"混合两个 CV" 就体现在 VC 侧：
    RVC 的**模型融合**（model fusion / ckpt merge）可以把两个音色模型按权重相加：
        merged = w * model_日配 + (1 - w) * model_中配
    调 w 就是在两者之间滑动，例如 0.5 是各占一半。

    本类只负责"把音频送去转换、把转换后的音频拿回来"，不关心服务端用什么模型。
    """
    name = "voice_converted"

    def __init__(self, inner, convert_url: str, model: str = "", blend: str = "",
                 timeout: int = 60, pitch: int = 0):
        self.inner = inner
        self.convert_url = (convert_url or "").rstrip("/")
        self.model = model
        self.blend = blend        # 例如 "alice_ja:0.5,alice_zh:0.5"
        self.timeout = timeout
        self.pitch = pitch

    @property
    def available(self) -> bool:
        return bool(getattr(self.inner, "available", False) and self.convert_url)

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        # 情感要传给**内层**（合成阶段），不是转换阶段——
        # 语气在合成时就定了；音色转换只改音色，会保留原有的语调起伏。
        try:
            base = self.inner.synthesize(text, emotion)
        except TypeError:
            base = self.inner.synthesize(text)
        if not base.ok:
            return base
        if not self.convert_url:
            return base
        converted = self._convert(base.data)
        if converted is None:
            # 转换失败就退回未转换的音频，但把 provider 标明，
            # 让上层能如实告知"音色转换没成功"。
            return SpeechResult(base.ok, data=base.data, mime=base.mime,
                                provider=f"{getattr(base, 'provider', '?')}(未转换)",
                                error="音色转换失败，已使用未转换音频",
                                emotion=getattr(base, "emotion", emotion),
                                emotion_applied=getattr(base, "emotion_applied", False),
                                emotion_note=getattr(base, "emotion_note", ""))
        return SpeechResult(True, data=converted, mime="audio/wav",
                            provider=f"{getattr(base, 'provider', '?')}+vc",
                            emotion=getattr(base, "emotion", emotion),
                            emotion_applied=getattr(base, "emotion_applied", False),
                            emotion_note=getattr(base, "emotion_note", ""))

    @property
    def supports_emotion(self) -> bool:
        return bool(getattr(self.inner, "supports_emotion", False))

    def _convert(self, audio: bytes):
        fields = {}
        if self.model:
            fields["model"] = self.model
        if self.blend:
            fields["blend"] = self.blend
        if self.pitch:
            fields["pitch"] = str(self.pitch)
        body, content_type = build_multipart(fields, "audio", "input.wav", audio,
                                             content_type="audio/wav")
        request = urllib.request.Request(f"{self.convert_url}/convert", data=body, method="POST")
        request.add_header("Content-Type", content_type)
        try:
            with net.open(request, timeout=self.timeout) as response:
                data = response.read(12 * 1024 * 1024)
        except Exception:
            return None
        return data if data and _looks_like_audio(data) else None


TTS_PROVIDERS = {
    "openai_compatible": "openai_compatible",
    "openai": "openai_compatible",
    "baidu": "baidu",
    "google": "google",
    "gtts": "google",
    # 自建克隆音色服务（GPT-SoVITS 等）
    "gptsovits": "gptsovits",
    "clone": "gptsovits",
    "custom": "gptsovits",
}


def build_tts_client(config):
    """按 TTS_PROVIDER 构建语音合成客户端。

    若配置了 TTS_VC_URL，会在合成之后**再套一层音色转换**——
    这是"中文发音正确 + 目标音色"的正确组合方式。

    baidu / google 免费且无需 Key，所以即使 tts_enabled=1 而没配 Key 也能用；
    openai_compatible 必须有 Key。
    """
    settings = getattr(config, "media", None)
    if settings is None or not settings.tts_enabled:
        return None
    provider = TTS_PROVIDERS.get(str(settings.tts_provider or "baidu").strip().lower())
    if provider == "baidu":
        return BaiduTTSClient(language=settings.tts_language or "zh",
                              speed=settings.tts_speed or 5)
    if provider == "google":
        return GoogleTTSClient(language=settings.tts_language or "zh-CN")
    if provider == "gptsovits":
        return GPTSoVITSTTSClient(
            settings.tts_clone_url,
            ref_audio=settings.tts_clone_ref_audio,
            prompt_text=settings.tts_clone_prompt_text,
            ref_audio_secondary=settings.tts_clone_ref_audio_secondary,
            text_lang=settings.tts_language or "zh",
            media_type=settings.tts_format or "wav",
            # 以前漏传这个，配置了语速也不生效（客户端本身支持 speed_factor）
            speed_factor=getattr(settings, "tts_clone_speed", 1.0) or 1.0,
            # 种子必须显式传：客户端默认 -1（随机），会让同一角色每次声音都不一样
            seed=getattr(settings, "tts_clone_seed", 42),
            # 注意：这里**不能**传 emotion_alpha——客户端是用"换参考音频"实现情感的，
            # 没有 alpha 混合这回事。曾经传过这个参数，导致 build_tts_client 直接
            # TypeError，运行时会静默退回 baidu（有测试才发现）。
            emotion_refs=parse_emotion_refs(getattr(settings, "tts_emotion_refs", "")))
    if provider == "openai_compatible":
        if not (settings.tts_api_key and settings.tts_base_url):
            return None
        return OpenAITTSClient(settings.tts_base_url, settings.tts_api_key,
                               settings.tts_model, settings.tts_voice, settings.tts_format)
    return None


class FallbackTTS:
    """主用克隆音色，不灵时退回免费 TTS。

    为什么需要：本地克隆服务是**你自己起的一个进程**——没开机、崩了、
    模型还在加载，都会让语音直接消失。这类"服务没起来"是最常见的情况，
    而那时候你只想要"有声音"，并不在乎是不是克隆音色。

    退回时会如实记录原因，不会假装还是克隆音色。
    """
    name = "fallback"

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup

    @property
    def available(self) -> bool:
        return bool(getattr(self.primary, "available", False)
                    or getattr(self.backup, "available", False))

    @property
    def supports_emotion(self) -> bool:
        return bool(getattr(self.primary, "supports_emotion", False))

    def synthesize(self, text: str, emotion: str = "") -> SpeechResult:
        primary_error = ""
        if getattr(self.primary, "available", False):
            try:
                result = self.primary.synthesize(text, emotion)
            except TypeError:
                result = self.primary.synthesize(text)
            if result.ok:
                return result
            primary_error = result.error
        else:
            primary_error = "克隆音色服务未配置或不可用"

        if not getattr(self.backup, "available", False):
            return SpeechResult(False, provider=self.name,
                                error=f"主用语音失败（{primary_error}），且没有可用的备用语音")
        try:
            result = self.backup.synthesize(text, emotion)
        except TypeError:
            result = self.backup.synthesize(text)
        if result.ok:
            # 明确标出"这是备用音色"，不要让人以为克隆生效了
            tail = f"克隆音色不可用（{primary_error}），已退回 {getattr(self.backup, 'name', '备用')}"
            result.emotion_note = ((result.emotion_note + "；") if result.emotion_note else "") + tail
            result.provider = f"{getattr(result, 'provider', '?')}(备用)"
        return result


def parse_emotion_refs(spec: str) -> dict:
    """解析情感参考表。

    格式：happy=/voices/happy.wav|开心地说这句话,sad=/voices/sad.wav

    竖线后面是**该参考音频里说的话**（prompt_text），必须与音频内容一致——
    换参考音频做情感时，文本也要跟着换，否则语气会飘。
    省略竖线时只换音频、沿用主 prompt_text。
    """
    refs = {}
    for chunk in str(spec or "").replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, target = chunk.partition("=")
        name, target = name.strip(), target.strip()
        if not (name and target):
            continue
        audio, _, prompt = target.partition("|")
        audio = audio.strip()
        if audio:
            refs[name] = (audio, prompt.strip()) if prompt.strip() else audio
    return refs


def _with_voice_conversion(client, settings):
    """给任意 TTS 客户端套上音色转换层（配置了 TTS_VC_URL 时）。"""
    if client is None or settings is None:
        return client
    convert_url = getattr(settings, "tts_vc_url", "") or ""
    if not convert_url:
        return client
    if not getattr(client, "available", False):
        return client
    return VoiceConvertedTTS(client, convert_url,
                             model=getattr(settings, "tts_vc_model", "") or "",
                             blend=getattr(settings, "tts_vc_blend", "") or "",
                             pitch=int(getattr(settings, "tts_vc_pitch", 0) or 0))


def build_tts(config):
    """构建最终使用的 TTS 客户端（含可选的音色转换层）。

    引擎应该用这个而不是 build_tts_client：
    只用 build_tts_client 会绕过音色转换。
    """
    settings = getattr(config, "media", None)
    primary = _with_voice_conversion(build_tts_client(config), settings)
    # 逐句独立合成放在变调之前：拼接好之后再统一变调，省一次 ffmpeg 调用，
    # 也避免每句各自变调引入细微差异。
    if primary is not None and getattr(settings, "tts_split_sentences", False):
        primary = SentenceSplitTTS(
            primary, gap_seconds=getattr(settings, "tts_sentence_gap", 0.25) or 0.0)
    # 变调层放在最外侧：音色转换之后再变调，顺序无所谓（都是改波形），
    # 但放最外侧能保证"最终听到的"就是变调后的。
    # 半音为 0 时 available 为 False，等价于不加这一层。
    semitones = float(getattr(settings, "tts_pitch", 0.0) or 0.0)
    if primary is not None and abs(semitones) > 1e-6:
        primary = PitchShiftTTS(primary, semitones=semitones,
                                ffmpeg=getattr(settings, "tts_ffmpeg", "") or "")
    fallback_name = str(getattr(settings, "tts_fallback", "") or "").strip().lower()
    primary_name = str(getattr(settings, "tts_provider", "") or "").strip().lower()
    # 备用与主用是同一个 provider 时没有意义（例如两个都写 baidu）
    if primary is None or not fallback_name or fallback_name == primary_name:
        return primary
    # 备用 provider 走同样的解析逻辑，但不套音色转换
    # （备用本就是为了"没有克隆音色时至少有声音"）
    original = settings.tts_provider
    try:
        settings.tts_provider = fallback_name
        backup = build_tts_client(config)
    finally:
        settings.tts_provider = original
    if backup is None or backup is primary:
        return primary
    return FallbackTTS(primary, backup)
