# -*- coding: utf-8 -*-
"""从 OneBot 消息段里抽取媒体输入（图片、语音）。

**为什么这些函数不在适配层里。**

适配层的所有代码都包在 `if HAS_NONEBOT:` 里面（没有 NoneBot 时整个模块
不会定义那些函数），后果是：**适配层的逻辑一行都测不到**，
而它的测试环境要求注定了 CI 里永远装不上 NoneBot。

项目在 `test_media_capture.py` 的模块说明里早就写过这条原则：

> 逻辑放在 aliceqq.media 而不是适配层，就是为了能被这里测到。

但后来写图片/语音抽取时我又把逻辑直接写进了适配层，于是重蹈覆辙。
现在把它移出来：这些函数只用到 segment 的 `type` 和 `data` 两个属性，
用 duck typing 就能测，不需要 NoneBot。

适配层里剩下的只是薄薄一层转发。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

from . import asr as asr_mod
from . import video as video_mod
from . import vision


def _text(value: Any) -> str:
    """把 segment 里的字段安全转成字符串。"""
    if value is None:
        return ""
    return str(value)


def segments_to_images(segments: Iterable[Any],
                       debug: Optional[Callable[[str], None]] = None) -> List[Any]:
    """把 image 段抽成 `vision.ImageRef` 列表。

    只取 `image` 段。QQ 的 `face`（内置表情）拿不到图片数据，
    所以不在这里处理——那属于"素材入库"流程，会另外提示用户。

    抽不到要留日志：静默返回空会让"为什么它看不见我发的图"
    变成没有任何线索的问题（`_was_mentioned` 那个坑就是这么来的）。
    """
    refs: List[Any] = []
    try:
        for segment in segments:
            if _text(getattr(segment, "type", "")) != "image":
                continue
            data = dict(getattr(segment, "data", {}) or {})
            url = _text(data.get("url"))
            path = _text(data.get("file"))
            if not url and not path:
                continue
            refs.append(vision.ImageRef(
                url=url, path=path, kind="sticker",
                source_hint=(path or url)[:80]))
    except Exception as exc:
        if debug:
            debug("segments_to_images 解析失败，本次按『没有图片』处理："
                  "%s: %s" % (type(exc).__name__, exc))
        return []
    if refs and debug:
        debug("本次收到 %d 张图片" % len(refs))
    return refs


def segments_to_videos(segments: Iterable[Any],
                       debug: Optional[Callable[[str], None]] = None) -> List[Any]:
    """把 video 段抽成 `video.VideoRef` 列表。

    QQ 的 video 段给的字段和其他段类似（`url` / `file`），
    另外通常带 `duration`。有的客户端还会给一个 `thumb`（封面图），
    这里**不用它**——抽帧能拿到更真实的画面，而且封面常常是无关的网图。
    """
    refs: List[Any] = []
    try:
        for segment in segments:
            if _text(getattr(segment, "type", "")) != "video":
                continue
            data = dict(getattr(segment, "data", {}) or {})
            url = _text(data.get("url"))
            path = _text(data.get("file"))
            if not url and not path:
                continue
            try:
                duration = float(data.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            raw_name = (path or "video.mp4").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            refs.append(video_mod.VideoRef(
                url=url, path=path, duration=duration,
                filename=raw_name[:40] or "video.mp4",
                source_hint=(path or url)[:80]))
    except Exception as exc:
        if debug:
            debug("segments_to_videos 解析失败，本次按『没有视频』处理："
                  "%s: %s" % (type(exc).__name__, exc))
        return []
    if refs and debug:
        debug("本次收到 %d 段视频" % len(refs))
    return refs


def segments_to_audios(segments: Iterable[Any],
                       debug: Optional[Callable[[str], None]] = None) -> List[Any]:
    """把 record 段抽成 `asr.AudioRef` 列表。

    `duration` 单独用 try 包住：QQ 有时给字符串、有时缺失、
    格式未必规范，而**一个坏字段不该让整条语音被丢掉**。
    """
    refs: List[Any] = []
    try:
        for segment in segments:
            if _text(getattr(segment, "type", "")) != "record":
                continue
            data = dict(getattr(segment, "data", {}) or {})
            url = _text(data.get("url"))
            path = _text(data.get("file"))
            if not url and not path:
                continue
            try:
                duration = float(data.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            raw_name = (path or "voice.m4a").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            refs.append(asr_mod.AudioRef(
                url=url, path=path, duration=duration,
                filename=raw_name[:40] or "voice.m4a",
                source_hint=(path or url)[:80]))
    except Exception as exc:
        if debug:
            debug("segments_to_audios 解析失败，本次按『没有语音』处理："
                  "%s: %s" % (type(exc).__name__, exc))
        return []
    if refs and debug:
        debug("本次收到 %d 条语音" % len(refs))
    return refs
