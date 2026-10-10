"""参考音频与克隆服务自检。

自己部署克隆音色时，最常见的失败不是模型问题，而是**参考音频不合规**：
带 BGM、多人声、太短、格式不对、路径服务端看不到。
这些都能在跑之前查出来，不必等到生成出怪声音再回头找。

用标准库 wave 解析 WAV 头，不引入任何依赖。
"""
from __future__ import annotations

from . import net
import os
import urllib.request
import wave
from typing import List

# 零样本克隆的推荐区间
IDEAL_MIN_SECONDS = 3.0
IDEAL_MAX_SECONDS = 15.0


def inspect_audio(path: str) -> dict:
    """检查单个参考音频文件。返回 {ok, notes, info}。"""
    notes: List[str] = []
    info: dict = {}
    if not path:
        return {"ok": False, "notes": ["未配置"], "info": info}
    if not os.path.isfile(path):
        return {"ok": False, "notes": ["文件不存在"], "info": info}

    size = os.path.getsize(path)
    info["size"] = size
    if size < 1024:
        notes.append(f"文件只有 {size} 字节，几乎肯定是坏的")

    suffix = os.path.splitext(path)[1].lower()
    info["suffix"] = suffix
    if suffix == ".wav":
        try:
            with wave.open(path, "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                seconds = frames / float(rate or 1)
                info.update({"seconds": round(seconds, 2), "rate": rate,
                             "channels": channels, "width": width})
                if channels != 1:
                    notes.append(f"是 {channels} 声道，建议转成单声道")
                if rate < 16000:
                    notes.append(f"采样率 {rate}Hz 偏低，建议 16k 以上")
                if seconds < IDEAL_MIN_SECONDS:
                    notes.append(f"只有 {seconds:.1f} 秒，太短（建议 "
                                 f"{IDEAL_MIN_SECONDS:.0f}~{IDEAL_MAX_SECONDS:.0f} 秒）")
                elif seconds > IDEAL_MAX_SECONDS:
                    notes.append(f"有 {seconds:.1f} 秒，偏长（建议 "
                                 f"{IDEAL_MIN_SECONDS:.0f}~{IDEAL_MAX_SECONDS:.0f} 秒；"
                                 "长音频里更容易混进换气与杂音）")
        except wave.Error as exc:
            notes.append(f"WAV 解析失败（{exc}）——扩展名是 wav 但内容不是？")
        except Exception as exc:
            notes.append(f"读取失败：{type(exc).__name__}")
    else:
        notes.append(f"不是 WAV（{suffix}）。GPT-SoVITS 对 WAV 最稳，"
                     "建议先转成 16k 单声道 wav")

    # 无法用标准库判断的项，明确交给人工
    notes.append("需人工确认：没有背景音乐、没有第二个人说话、没有爆音")
    return {"ok": not any("不存在" in n or "坏的" in n or "失败" in n for n in notes),
            "notes": notes, "info": info}


def check_clone_service(base_url: str, timeout: int = 4) -> dict:
    """看克隆服务在不在。"""
    if not base_url:
        return {"ok": False, "note": "未配置 TTS_CLONE_URL"}
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/docs",
                                        headers={"User-Agent": "alice-qq-agent"})
        with net.open(request, timeout=timeout) as response:
            return {"ok": True, "note": f"服务可达（HTTP {response.status}）"}
    except Exception as exc:
        # 服务可能没有 /docs，连接被拒才说明没起来
        return {"ok": False,
                "note": f"连不上（{type(exc).__name__}）——服务没启动？"
                        f"启动命令：python api_v2.py -a 127.0.0.1 -p 9880"}


def check_voice_setup(config) -> str:
    """渲染一份完整的自检报告。"""
    settings = getattr(config, "media", None)
    lines: List[str] = ["【语音链路自检】"]
    if settings is None:
        return "（没有 media 配置）"

    from .generation import (describe_voice_pipeline, parse_blend, parse_emotion_refs)

    lines.append("  " + describe_voice_pipeline(settings))
    provider = (getattr(settings, "tts_provider", "") or "").lower()
    lines.append("")

    if provider not in ("gptsovits", "clone", "custom"):
        lines.append(f"  当前 provider 是 {provider}，不使用本地克隆服务。")
        lines.append("  想用克隆音色：把 TTS_PROVIDER 改成 gptsovits 并启动本地服务。")
        return "\n".join(lines)

    # 服务
    service = check_clone_service(getattr(settings, "tts_clone_url", ""))
    lines.append(f"  [服务] {'OK  ' if service['ok'] else '问题'} {service['note']}")

    # 主参考
    primary = getattr(settings, "tts_clone_ref_audio", "") or ""
    result = inspect_audio(primary)
    lines.append(f"  [主参考] {primary or '(未配置)'}")
    for note in result["notes"]:
        lines.append(f"      - {note}")

    # 第二参考（音色融合）
    secondary = getattr(settings, "tts_clone_ref_audio_secondary", "") or ""
    if secondary:
        lines.append(f"  [第二参考（音色融合）] {secondary}")
        for note in inspect_audio(secondary)["notes"]:
            lines.append(f"      - {note}")
        lines.append("      注意：官方 api_v2 的 aux_ref_audio_paths **没有权重参数**，")
        lines.append("            要按 0.7/0.3 混合得靠 RVC 模型融合，")
        lines.append("            或者按 7:3 时长比例准备微调数据集。")
    else:
        lines.append("  [第二参考] 未配置（单音色，不做融合）")

    # 情感
    refs = parse_emotion_refs(getattr(settings, "tts_emotion_refs", ""))
    if refs:
        lines.append(f"  [情感参考] 共 {len(refs)} 个")
        for name, entry in refs.items():
            path = entry[0] if isinstance(entry, tuple) else entry
            has_prompt = isinstance(entry, tuple) and bool(entry[1])
            issues = [n for n in inspect_audio(path)["notes"]
                      if "人工确认" not in n]
            mark = "OK" if not issues else "问题"
            lines.append(f"      {name}: {mark} {path}"
                         + ("" if has_prompt else "  （缺少该音频对应的文本，"
                                                  "建议写成 name=路径|音频里说的话）"))
            for issue in issues:
                lines.append(f"          - {issue}")
    else:
        lines.append("  [情感参考] 未配置（语气会由主参考音频决定）")

    # 转换层
    blend = parse_blend(getattr(settings, "tts_vc_blend", "") or "")
    vc_url = getattr(settings, "tts_vc_url", "") or ""
    if blend and not vc_url:
        lines.append("")
        lines.append("  [警告] 配了 TTS_VC_BLEND 但没有 TTS_VC_URL，混合权重不会生效。")
    if vc_url:
        lines.append(f"  [音色转换] {vc_url}")
        if blend:
            lines.append("      权重：" + " ".join(f"{k}={v:.0%}" for k, v in blend.items()))
    return "\n".join(lines)
