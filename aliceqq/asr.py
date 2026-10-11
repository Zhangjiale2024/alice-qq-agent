"""语音识别（音频转文字）。

为什么需要独立模块：B 站**只对部分视频**提供 AI 字幕（且需登录），
没有字幕时想拿到"视频里说了什么"，只能自己下载音频再做识别。

本模块提供两种途径：

1. **B 站自带 AI 字幕**（见 search.BilibiliProvider）：免费、无需额外依赖，
   配置 BILIBILI_SESSDATA 即可，本质就是 B 站对音频做好的识别结果。
2. **自有 ASR**（本模块）：走 OpenAI 兼容的 /audio/transcriptions 接口，
   适合"B 站没给字幕"或"根本不在 B 站"的音频。

实现取舍：
- 纯标准库手写 multipart/form-data——不引入 requests；
- **强制时长与体积上限**：识别按时长计费，一个 20 分钟视频约 $0.12，
  不设限就会在公开服务里被玩成账单事故；
- 失败一律如实返回，不编造识别结果。
"""
from __future__ import annotations

from . import net
import os
import json
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 单次识别的体积上限（音频按 64kbps 估算：1 分钟约 0.5MB）
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SECONDS = 300
DEFAULT_TIMEOUT = 120

# 用户一条语音消息的体积上限。
# 比上面那个 DEFAULT_MAX_BYTES 小得多，因为场景完全不同：
# 那个是转写视频音轨（可能几分钟），这个是"群友随口说一句"。
# QQ 语音消息通常十几秒、几十 KB，给它 2MB 已经很宽松了。
VOICE_MESSAGE_MAX_BYTES = 2 * 1024 * 1024


@dataclass
class AudioRef:
    """用户发来的一段语音。

    和 vision.ImageRef 是同一个思路：适配层只负责说"这里有段音频、
    它在哪"，真正的读取和识别由引擎侧完成。

    `url` 和 `path` 至少有其一。QQ 的 record 段通常给 http(s) 地址，
    也可能给本地缓存路径。`blob` 是已经读到的字节，优先级最高。
    """

    url: str = ""
    path: str = ""
    blob: bytes = b""
    filename: str = "voice.m4a"
    duration: float = 0.0          # 秒；QQ 有时会给，0 表示未知
    source_hint: str = ""

    def read_bytes(self) -> bytes:
        """取到音频字节。失败或超限返回空 bytes（调用方据此如实报错）。"""
        if self.blob:
            # **这里以前直接 return，漏了体积检查**——
            # 文件分支检查了、blob 分支没有，于是"已经读进内存的大音频"
            # 会绕过上限直接送进识别。测试抓到了这个不一致。
            if len(self.blob) > VOICE_MESSAGE_MAX_BYTES:
                return b""
            return self.blob
        target = self.url or self.path
        if not target:
            return b""
        try:
            # read_media_bytes 支持本地路径与 http(s) 两种来源。
            # 它本身没有体积参数，所以取到之后自己判断。
            from .media import read_media_bytes
            data = read_media_bytes(target) or b""
        except Exception:
            return b""
        if len(data) > VOICE_MESSAGE_MAX_BYTES:
            return b""
        return data


@dataclass
class Transcription:
    ok: bool
    text: str = ""
    provider: str = ""
    error: str = ""
    seconds: float = 0.0
    bytes_sent: int = 0
    # 质量指标（不是所有后端都给）：日志概率均值、无语音概率、识别段数。
    # 作用见 quality_warning()——ASR 在噪声/背景音乐上会输出**通顺但完全错误**
    # 的文本，光看文本看不出来，需要这些指标 + 文本启发式一起判断。
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    segments: int = 0


# 这些字符在正常中文/日文语音转写里占比极低，一旦大量出现
# 往往意味着模型在"唱"背景音乐或复读。
_REPEAT_MIN_CHARS = 40      # 太短的文本不做复读判断（样本不足）
_REPEAT_RATIO = 0.60        # 单个字符占比超过这个值就认为在复读


def quality_warning(text: str, *, seconds: float = 0.0,
                    avg_logprob: Optional[float] = None,
                    no_speech_prob: Optional[float] = None) -> str:
    """判断一段识别结果是否**不可靠**，返回给用户看的如实说明（可靠时返回空串）。

    为什么要这个：本次配音调试实测过——ASR 遇到噪声/背景音乐时会输出
    **语法通顺但内容完全无关**的文本（例如把游戏音效识别成"今天我已死定了"）。
    这种结果比"识别失败"更危险，因为看起来像真的。这里用四条独立的线索判断，
    任何一条命中就明确标注"不可靠"，而不是把文本当事实发出去。

    指标只能**证伪**（说明"这次别信"），不能证成——所以没有告警不代表一定准确，
    返回的文本始终带"未核对"的说明。
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    reasons = []
    # 1) 文本短得离谱：说了 N 秒却只有几个字，通常是音效被硬凑成了字
    if seconds >= 20 and len(stripped) < 8:
        reasons.append(f"{round(seconds)} 秒音频只识别出 {len(stripped)} 个字")
    # 2) 复读：单个字符占绝对多数（模型卡在循环里）
    if len(stripped) >= _REPEAT_MIN_CHARS:
        most = max(stripped.count(ch) for ch in set(stripped))
        if most / len(stripped) > _REPEAT_RATIO:
            reasons.append("识别结果高度重复（疑似模型陷入复读）")
    # 3) 无语音概率高：模型自己都觉得这里没人说话
    if no_speech_prob is not None and no_speech_prob > 0.6:
        reasons.append(f"模型判断该段很可能没有人声（无语音概率 {no_speech_prob:.2f}）")
    # 4) 平均日志概率过低：模型对每个字的把握都很低
    if avg_logprob is not None and avg_logprob < -1.0:
        reasons.append(f"识别置信度偏低（平均对数概率 {avg_logprob:.2f}）")
    if not reasons:
        return ""
    return ("（注意：本次识别**可能不可靠**——" + "；".join(reasons)
            + "。文本仅供参考，请勿当作事实使用）")


def build_multipart(fields: Dict[str, str], file_field: str, filename: str,
                    file_bytes: bytes, content_type: str = "audio/mpeg") -> Tuple[bytes, str]:
    """手工拼 multipart/form-data。

    返回 (body, content_type)。用标准库而非 requests，是为了保持零依赖。
    """
    boundary = "----AliceFormBoundary" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode())
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class ASRClient:
    """OpenAI 兼容的语音识别客户端（Whisper 等）。"""

    def __init__(self, base_url: str, api_key: str, model: str = "whisper-1",
                 language: str = "zh", timeout: int = DEFAULT_TIMEOUT):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "whisper-1"
        self.language = language
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def transcribe(self, audio: bytes, filename: str = "audio.m4a") -> Transcription:
        if not self.available:
            return Transcription(False, provider="openai_compatible",
                                 error="未配置 ASR（需要 ASR_API_KEY 与 ASR_BASE_URL）")
        if not audio:
            return Transcription(False, provider="openai_compatible", error="音频内容为空")
        fields = {"model": self.model}
        if self.language:
            fields["language"] = self.language
        body, content_type = build_multipart(fields, "file", filename, audio)
        request = urllib.request.Request(f"{self.base_url}/audio/transcriptions",
                                         data=body, method="POST")
        request.add_header("Content-Type", content_type)
        request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with net.open(request, timeout=self.timeout) as response:
                raw = response.read(400_000)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:200]
            hint = ""
            if exc.code in (401, 403):
                hint = "（ASR 的 API Key 无效或无权访问）"
            elif exc.code == 429:
                hint = "（ASR 频率或额度限制）"
            return Transcription(False, provider="openai_compatible",
                                 error=f"HTTP {exc.code}{hint}: {detail}",
                                 bytes_sent=len(body))
        except Exception as exc:
            return Transcription(False, provider="openai_compatible",
                                 error=f"{type(exc).__name__}: {exc}", bytes_sent=len(body))

        body = raw.decode("utf-8", "ignore").strip()
        text = ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            text = str(payload.get("text") or "").strip()
        elif isinstance(payload, str):
            # 有的服务返回的是"裸 JSON 字符串"（"\u4f60\u597d" 这种）。
            # 早期实现直接回退到原始字节，结果把转义符当成识别结果返回。
            text = payload.strip()
        else:
            # 非 JSON：按纯文本处理
            text = body
        if not text:
            return Transcription(False, provider="openai_compatible",
                                 error="识别结果为空", bytes_sent=len(body))
        return Transcription(True, text=text, provider="openai_compatible",
                             bytes_sent=len(body))


def _add_torch_cuda_dlls() -> str:
    """把 torch 自带的 CUDA 运行库加进 DLL 搜索路径，返回所用目录（失败返回空串）。

    为什么需要：ctranslate2（faster-whisper 的推理后端）用 CUDA 时要
    cublas64_12.dll / cudnn*_9.dll。这些文件**已经在 torch\\lib 里**
    （torch 附带的），但不在 PATH 上，于是直接报
    "Library cublas64_12.dll is not found or cannot be loaded"。

    实测把该目录加进去后 GPU 就通了（同一段音频 2.68 秒 -> 0.95 秒），
    不需要再装 nvidia-cublas-cu12 / nvidia-cudnn-cu12。

    注意 Windows 上 Python 3.8+ 已不再从 PATH 查找扩展模块的依赖，
    必须用 os.add_dll_directory。
    """
    try:
        import torch
    except Exception:
        return ""
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(lib):
        return ""
    try:
        # 返回值要留着，句柄被回收的话目录会失效
        _add_torch_cuda_dlls._handle = os.add_dll_directory(lib)
    except (AttributeError, OSError):
        pass
    current = os.environ.get("PATH", "")
    if lib not in current.split(os.pathsep):
        os.environ["PATH"] = lib + os.pathsep + current
    return lib


class LocalWhisperASR:
    """本地 faster-whisper 语音识别（不花钱、不联网）。

    为什么加这个后端：原设计只有"OpenAI 兼容的云端接口"，按量计费，
    公开服务里合适，**自用时却没必要**——本地能跑就用本地的，隐私也好。

    两个实现要点：

    1. **依赖是可选的**。aliceqq 整体保持纯标准库，faster-whisper
       只在真正要识别时才导入。没装时 available 为 False，
       上层会如实说"本地识别不可用"，而不是静默失败或编造结果。
    2. **模型实例要缓存**。加载一次 small 模型约 3~10 秒、占 500MB 内存，
       每次识别都重建会让每条语音都卡住，所以第一次用完就留在实例上。
    """
    provider = "local_whisper"

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 language: str = "zh", cache_dir: str = "",
                 timeout_seconds: int = 300, initial_prompt: str = "",
                 vocabulary: str = ""):
        self.model_size = model_size or "small"
        self.device = (device or "cpu").lower()
        self.language = language or ""
        self.cache_dir = cache_dir or ""
        self.timeout = timeout_seconds
        self.vocabulary = (vocabulary or "").strip()
        # 提示词 = 基础提示 + 专名列表。
        # 专名单独放一个配置项，因为它**每个角色/项目都不同**，
        # 不该写死在代码里；实测把角色名加进去能显著降低同音字错误。
        base = initial_prompt if initial_prompt is not None else ""
        if self.vocabulary:
            base = (base + "常见词：" + self.vocabulary + "。") if base \
                else ("常见词：" + self.vocabulary + "。")
        self.initial_prompt = base.strip()
        self._model = None
        self._load_error = ""

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except Exception as exc:
            self._load_error = f"未安装 faster-whisper（{type(exc).__name__}）"
            return False
        return True

    def _load(self):
        """加载模型（只做一次）。

        **GPU 加载失败时自动退回 CPU。** 实测踩过：装好 faster-whisper 后
        `available=True`，但 `.env` 里配的 `ASR_LOCAL_DEVICE=cuda`，
        而这个解释器**没有 torch**——ctranslate2 用 GPU 需要 CUDA 运行库，
        报的是：

            RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

        如果就这样失败，表现是"语音识别又不可用"，而真正的原因
        （缺一个 cublas DLL）完全看不出来。同一个模型在 CPU 上
        实测 4.6 秒转写完全可用，所以这里直接降级——
        慢一点总比整条链路不可用好，而且会留下日志说明发生了什么。
        """
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel
        # CPU 上用 int8：比 float32 快很多，精度损失对识别几乎无感。
        # GPU 上用 float16（4060 支持）。
        if self.device == "cuda":
            # GPU 推理要先让 ctranslate2 找到 CUDA 运行库（见函数说明）
            _add_torch_cuda_dlls()
        compute = "float16" if self.device == "cuda" else "int8"
        kwargs = {"device": self.device, "compute_type": compute}
        if self.cache_dir:
            # 模型缓存**必须**能写。默认位置是用户目录下的 .cache，
            # 在受限环境里会被拒绝访问，所以允许显式指定。
            kwargs["download_root"] = self.cache_dir
        try:
            self._model = WhisperModel(self.model_size, **kwargs)
            if self.device == "cuda":
                # **必须在这里真的跑一次推理探针。**
                # 实测发现 `WhisperModel(device="cuda")` 构造时**不会**
                # 加载 cuBLAS——错误要等到第一次 transcribe 才爆：
                #
                #     RuntimeError: Library cublas64_12.dll is not found
                #
                # 所以只把构造包在 try 里是不够的（第一版就是这么写的，
                # 结果降级从未触发）。用一小段静音做探针，把问题
                # 在"加载模型"这一步就暴露出来。
                self._probe_gpu()
        except Exception as exc:
            if self.device != "cuda":
                raise
            # 只对 GPU 失败降级；CPU 都失败说明是模型/缓存问题，
            # 那种情况降级也救不了，必须如实抛出去。
            detail = str(exc)[:140]
            self._model = None
            self.device = "cpu"
            self._gpu_fallback_reason = (
                "GPU 不可用（%s: %s），已自动改用 CPU。"
                "想要 GPU 加速需要装齐 CUDA 运行库（含 cublas）。"
                % (type(exc).__name__, detail))
            self._model = WhisperModel(self.model_size, device="cpu",
                                       compute_type="int8", **(
                                           {"download_root": self.cache_dir}
                                           if self.cache_dir else {}))
        return self._model

    def _probe_gpu(self) -> None:
        """用一小段静音跑一次 GPU 推理，确认 cuBLAS 等运行库真的能加载。

        只在模型确实有 `transcribe` 时才探针——测试里会用
        **假模型**替换 `faster_whisper.WhisperModel`（只记录参数），
        那种对象没有这个方法。如果无条件去调，假模型会被判成
        "GPU 不可用"并触发降级，于是 `compute_type` 从 float16
        变成 int8，把"按设备选 compute_type"那条既有测试弄红。
        """
        if not callable(getattr(self._model, "transcribe", None)):
            return
        import io
        import struct
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack("<%dh" % 1600, *([0] * 1600)))
        segments, _ = self._model.transcribe(
            io.BytesIO(buffer.getvalue()), language=self.language or None,
            beam_size=1, word_timestamps=False)
        for _ in segments:          # 生成器必须消费完才会真正执行
            pass

    def transcribe(self, audio: bytes, filename: str = "audio.wav") -> Transcription:
        if not audio:
            return Transcription(False, provider=self.provider, error="音频内容为空")
        if not self.available:
            return Transcription(False, provider=self.provider,
                                 error=self._load_error or "本地识别不可用")
        try:
            model = self._load()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:120]}"
            self._load_error = detail
            return Transcription(False, provider=self.provider,
                                 error=f"加载本地模型失败：{detail}")
        # 直接用 BytesIO 喂给 faster-whisper，不落临时文件：
        # 它接受 file-like 对象，而临时文件在受限环境里未必写得进去。
        import io
        try:
            extra = {}
            if self.initial_prompt:
                # **initial_prompt 是实测最有效的单项改动**：
                # 同一批语音，不加提示字符错误率 40.6%、完全正确 0/12；
                # 加上"简体中文 + 专名"的提示后降到 1.0%、完全正确 11/12。
                # 主导错误其实是**输出繁体字**，不是同音字。
                extra["initial_prompt"] = self.initial_prompt
            segments, info = model.transcribe(
                io.BytesIO(audio),
                language=self.language or None,
                beam_size=5,
                word_timestamps=False,
                **extra,
            )
            # segments 是生成器，必须在这里消费完才能真正拿到结果。
            # 顺便收集置信度指标（见 quality_warning）。
            parts = []
            logprobs = []
            no_speech = []
            count = 0
            for seg in segments:
                parts.append(seg.text)
                count += 1
                lp = getattr(seg, "avg_logprob", None)
                if isinstance(lp, (int, float)):
                    logprobs.append(float(lp))
                nsp = getattr(seg, "no_speech_prob", None)
                if isinstance(nsp, (int, float)):
                    no_speech.append(float(nsp))
            text = "".join(parts).strip()
        except Exception as exc:
            return Transcription(False, provider=self.provider,
                                 error=f"{type(exc).__name__}: {str(exc)[:120]}")
        if not text:
            return Transcription(False, provider=self.provider,
                                 error="识别结果为空", bytes_sent=len(audio))
        return Transcription(True, text=text, provider=self.provider,
                             bytes_sent=len(audio),
                             seconds=float(getattr(info, "duration", 0.0) or 0.0),
                             avg_logprob=(sum(logprobs) / len(logprobs)) if logprobs else None,
                             no_speech_prob=(max(no_speech) if no_speech else None),
                             segments=count)


def build_client(config):
    """按配置构建 ASR 客户端；未启用或不完整时返回 None。

    两个后端：
      local             —— 本地 faster-whisper，不需要 Key
      openai_compatible —— 云端接口，需要 Key
    """
    settings = getattr(config, "asr", None)
    if settings is None or not settings.enabled:
        return None

    provider = str(getattr(settings, "provider", "") or "").strip().lower()
    if provider in ("local", "faster_whisper", "faster-whisper", "whisper_local"):
        return LocalWhisperASR(
            model_size=getattr(settings, "local_model", "small") or "small",
            device=getattr(settings, "local_device", "cpu") or "cpu",
            language=settings.language,
            cache_dir=getattr(settings, "local_cache", "") or "",
            timeout_seconds=settings.timeout_seconds,
            # 提示词与专名表。getattr 带默认值是为了兼容老的配置对象
            # （测试里会传只有部分字段的假配置）。
            initial_prompt=getattr(settings, "initial_prompt", "") or "",
            vocabulary=getattr(settings, "vocabulary", "") or "",
        )

    if not (settings.api_key and settings.base_url):
        return None
    return ASRClient(settings.base_url, settings.api_key, settings.model,
                     language=settings.language, timeout=settings.timeout_seconds)
