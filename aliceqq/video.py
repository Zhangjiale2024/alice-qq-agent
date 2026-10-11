# -*- coding: utf-8 -*-
"""视频输入：从一段视频里同时取画面和声音，交给视觉与语音两条链路。

**这是"多模态"的第三块，也是最后一块。**
在此之前项目完全不认 `video` 段——适配层只取 `get_plaintext()`，
图片和语音各自做完之后视频仍然是断的。

设计上刻意复用了已有的两条链路，而不是另起一套：

    视频 --抽帧--> JPEG 帧 --+--> VisionClient（图片理解那条）
         --抽音轨--> WAV   --+--> ASR（语音理解那条）

这样视频不需要新的模型、新的额度类型、新的护栏——
`vision` 和 `asr_voice` 的额度与门禁自动生效。

为什么**音轨往往比画面更有信息量**：QQ 上转发的视频多是短视频
（梗图、片段），"说了什么"通常比"画面长什么样"更能说明内容，
所以音轨转写失败时也不算全输——帧还能看。

没有音轨的视频（纯画面）是常见情况，不当成错误处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from . import ffmpeg as ffmpeg_mod

# 一条视频消息的体积上限。QQ 上转发的短视频通常几 MB，
# 给到 24MB 已经很宽松；再大就不是"随手转发"而是刻意上传了。
VIDEO_MESSAGE_MAX_BYTES = 24 * 1024 * 1024

# 抽几帧。**不是越多越好**：视觉接口按图算钱，
# 而且三帧已经足够看出"画面里在发生什么"。
DEFAULT_FRAME_COUNT = 3

# 缩到多宽。1080p 一帧可能好几 MB，缩到 640 宽后通常几十 KB，
# 对"看懂内容"完全够用。
DEFAULT_FRAME_WIDTH = 640

# 音轨最多转写多久。QQ 短视频一般几十秒；太长的视频转写既慢又贵，
# 截取前 N 秒通常已经包含主要信息（开场就说话）。
MAX_AUDIO_SECONDS = 120.0


@dataclass
class VideoRef:
    """用户发来的一段视频。

    和 ImageRef / AudioRef 同一个思路：适配层只说"有段视频、它在哪"，
    读取与解析由引擎侧完成。
    """

    url: str = ""
    path: str = ""
    blob: bytes = b""
    duration: float = 0.0          # 秒；QQ 有时会给，0 表示未知
    filename: str = "video.mp4"
    source_hint: str = ""

    def read_bytes(self) -> bytes:
        """取到视频字节。失败或超限返回空 bytes。"""
        if self.blob:
            if len(self.blob) > VIDEO_MESSAGE_MAX_BYTES:
                return b""
            return self.blob
        target = self.url or self.path
        if not target:
            return b""
        try:
            from .media import read_media_bytes
            data = read_media_bytes(target) or b""
        except Exception:
            return b""
        if len(data) > VIDEO_MESSAGE_MAX_BYTES:
            return b""
        return data


@dataclass
class VideoAnalysis:
    """视频解析结果。两部分可以各自独立成功或失败。"""

    ok: bool = False
    frames: List[bytes] = field(default_factory=list)
    audio: bytes = b""
    audio_seconds: float = 0.0
    has_audio: bool = False
    error: str = ""
    note: str = ""
    elapsed: float = 0.0


def analyze(data: bytes, ffmpeg: str = "",
            frame_count: int = DEFAULT_FRAME_COUNT,
            frame_width: int = DEFAULT_FRAME_WIDTH,
            max_audio_seconds: float = MAX_AUDIO_SECONDS,
            duration_hint: float = 0.0,
            timeout: int = 180) -> VideoAnalysis:
    """把视频拆成"帧"和"音轨"两部分。

    **只要任一部分拿到内容就算成功**（ok=True），因为：
      * 纯画面视频（没有音轨）仍然值得看画面；
      * 画面全黑但有人说话的音频仍然值得听。
    两部分都拿不到才是真失败。
    """
    import time

    started = time.time()
    if not data:
        return VideoAnalysis(False, error="视频内容为空")
    if not ffmpeg:
        return VideoAnalysis(False, error="没有可用的 ffmpeg，无法解析视频")

    # 时长：优先用调用方给的（QQ 的 video 段带 duration，省一次解码）
    duration = float(duration_hint or 0.0)
    if duration <= 0:
        probed, _ = ffmpeg_mod.probe_duration(data, ffmpeg, timeout=min(timeout, 120))
        duration = probed

    frames, frame_err = ffmpeg_mod.extract_video_frames(
        data, ffmpeg, count=frame_count, width=frame_width,
        duration=duration or float(frame_count), timeout=timeout)

    # 音轨单独截断：长视频只转写前一段，避免又慢又贵
    clip = 0.0
    if max_audio_seconds and duration > max_audio_seconds:
        clip = float(max_audio_seconds)
    audio, audio_err = ffmpeg_mod.extract_video_audio(
        data, ffmpeg, start=0.0, duration=clip, timeout=timeout)

    has_audio = bool(audio)
    notes = []
    if has_audio and clip:
        notes.append("音轨只转写了前 %.0f 秒" % clip)
    if not has_audio:
        # 没有音轨是常见情况，不当错误
        notes.append("这段视频没有音轨")
    if frames and len(frames) < frame_count:
        notes.append("只抽到 %d 帧" % len(frames))

    ok = bool(frames) or has_audio
    error = ""
    if not ok:
        error = frame_err or audio_err or "视频解析失败"
    return VideoAnalysis(
        ok=ok, frames=frames, audio=audio or b"",
        audio_seconds=clip or duration,
        has_audio=has_audio, error=error,
        note="；".join(notes), elapsed=time.time() - started)


def render_for_model(analysis: VideoAnalysis,
                     frame_result: Optional[Any] = None,
                     transcript: str = "",
                     transcript_note: str = "") -> str:
    """把视频解析结果渲染成注入提示词的一段文字。

    **两部分都要如实标注来源**：画面来自抽帧后的视觉识别，
    声音来自音轨转写。模型不该以为自己在"看视频"。
    """
    if not analysis.ok:
        why = analysis.error or "解析失败"
        return ("（用户这次发了一段**视频**，但你没有拿到内容——%s。"
                "请如实告诉用户你看不了这个视频，"
                "**不要猜测视频里是什么**。）" % why)

    lines = ["（用户这次发的是一段**视频**，下面是机器从里面提取的信息。"
             "你并没有从头到尾看过这段视频，只是看了几帧画面和一段音轨转写。）"]

    if frame_result is not None and getattr(frame_result, "ok", False):
        lines.append("")
        lines.append("【画面】从视频里抽了几帧做了识别：")
        lines.append(getattr(frame_result, "description", "").strip())
        if getattr(frame_result, "note", ""):
            lines.append(str(frame_result.note))
    elif analysis.frames:
        lines.append("")
        lines.append("【画面】抽到了 %d 帧，但识别没有成功。" % len(analysis.frames))
    else:
        lines.append("")
        lines.append("【画面】没有抽到可用的画面帧。")

    if transcript:
        lines.append("")
        lines.append("【声音】视频音轨转写：")
        lines.append(transcript)
        if transcript_note:
            lines.append("注意：" + transcript_note)
    elif analysis.has_audio:
        lines.append("")
        lines.append("【声音】有音轨，但转写没有成功。")
    else:
        lines.append("")
        lines.append("【声音】这段视频没有音轨。")

    if analysis.note:
        lines.append("")
        lines.append("（%s）" % analysis.note)

    lines.append("")
    lines.append("**只依据上面这些说**。不要补充没有提到的画面细节或对话内容；"
                 "可以说『从抽帧看』『从音轨听』，让用户知道这是机器提取的。")
    return "\n".join(lines)
