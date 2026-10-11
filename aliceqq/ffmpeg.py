"""ffmpeg 定位与音频片段裁剪（可复用）。

为什么单独抽一个模块：这段"去哪儿找 ffmpeg"的逻辑原先散在两个地方——
`generation.PitchShifter` 里有一份（含一条**硬编码**的 imageio-ffmpeg 路径），
而 `search.BilibiliProvider._transcribe` 的注释还写着"当前没有 ffmpeg 无法裁剪音频"，
于是超过 ASR 时长上限的视频**直接放弃**。实际上 ffmpeg 就在这台机器上
（GPT-SoVITS venv 里 imageio-ffmpeg 自带的那个二进制）。

本模块只做两件事，保持纯标准库 + subprocess：

1. `find_ffmpeg(config=None, explicit="")` —— 按 显式路径 -> 配置 -> PATH -> 常见位置
   的顺序找一个可用的 ffmpeg；找不到就返回空串，调用方如实说明，不假装成功。
2. `extract_segment(data, start, duration, ...)` —— 从内存里的音频字节裁出一段
   （默认用 `-c copy`，只丢不重编码，秒级完成），返回新字节。

设计取舍：
- **不落临时文件**：B 站音频的容器头有偏移，写临时文件反而更容易出错；
  直接把字节喂给 ffmpeg 的 stdin、从 stdout 收结果。
- **裁剪失败不抛异常**：返回 (None, 原因)，让上层决定怎么如实说明。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Optional, Tuple

# ffmpeg 的进度行形如 "time=00:00:03.05"（-f null 解码结束时给的就是总时长）
_TIME_PATTERN = re.compile(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)")

# 已知的 imageio-ffmpeg 安装位置（GPT-SoVITS venv 里那个）。
# 这里列出来是因为该 venv 不在 PATH 上，而它的 ffmpeg 是实测可用的。
KNOWN_DIRS = (
    r"D:\DSHWORK\envs\gptsovits\Lib\site-packages\imageio_ffmpeg\binaries",
)

DEFAULT_TIMEOUT = 120


def _scan_dir(directory: str) -> str:
    """在一个目录里找 ffmpeg 可执行文件（imageio 的文件名带版本号）。"""
    if not directory or not os.path.isdir(directory):
        return ""
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return ""
    for name in names:
        low = name.lower()
        if low.startswith("ffmpeg") and low.endswith(".exe"):
            return os.path.join(directory, name)
    return ""


def _from_config(config: Any) -> str:
    """从配置里取显式路径：先 FFMPEG，再兼容历史的 TTS_FFMPEG。"""
    if config is None:
        return ""
    for name in ("ffmpeg_path", "tts_ffmpeg"):
        value = getattr(config, name, "") or ""
        if value and os.path.isfile(str(value)):
            return str(value)
    raw = getattr(config, "raw_env", None)
    if isinstance(raw, dict):
        for key in ("FFMPEG", "TTS_FFMPEG"):
            value = str(raw.get(key, "") or "")
            if value and os.path.isfile(value):
                return value
    return ""


def find_ffmpeg(config: Any = None, explicit: str = "") -> str:
    """按优先级找一个可用的 ffmpeg，找不到返回空串。

    顺序：显式参数 -> 配置里的路径 -> PATH -> 已知安装目录。
    """
    if explicit and os.path.isfile(explicit):
        return explicit
    from_config = _from_config(config)
    if from_config:
        return from_config
    found = shutil.which("ffmpeg")
    if found:
        return found
    for directory in KNOWN_DIRS:
        found = _scan_dir(directory)
        if found:
            return found
    return ""


def run(args, input_bytes: Optional[bytes] = None,
        timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[bytes], str]:
    """跑一次 ffmpeg，返回 (stdout 字节, 错误说明)。成功时错误为空串。"""
    try:
        proc = subprocess.run(args, input=input_bytes, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError:
        return None, "找不到 ffmpeg 可执行文件"
    except subprocess.TimeoutExpired:
        return None, f"ffmpeg 超时（{timeout} 秒）"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "ignore").strip().splitlines()
        detail = tail[-1][:160] if tail else ""
        return None, f"ffmpeg 退出码 {proc.returncode}" + (f"：{detail}" if detail else "")
    if not proc.stdout:
        return None, "ffmpeg 没有输出音频数据"
    return proc.stdout, ""


def extract_segment(data: bytes, start: float, duration: float,
                    ffmpeg: str = "", sample_rate: int = 16000,
                    timeout: int = DEFAULT_TIMEOUT,
                    ) -> Tuple[Optional[bytes], str]:
    """从音频字节里裁出 [start, start+duration) 一段，输出 16kHz 单声道 WAV。

    **两个踩过的坑（别改回去）**：

    1. 不能用 `-c copy`。流复制只能从**关键帧**开始切，实测一段 435.54 秒的
       B 站 AAC 音频，请求 [300, +300) 会得到 430.56 秒——切出来的边界根本不对，
       而"时长看起来像是有内容"会让错误更难发现。这里一律重编码：
       整段 7 分钟音频切一段只要 0.13 秒，代价可以忽略。
    2. `-ss` 在 `-i` 之前时，`-t` 的含义会变成**绝对结束时间**而不是时长。
       实测请求 [200, +300) 只得到 235.54 秒（= 435.54 - 200，被截到文件尾），
       于是"每段 5 分钟"会退化成"到 300 秒处为止"，第 2 段之后全被截断。
       这里统一用 `-ss 起点 -to 终点`（都是绝对秒）。
       结束位置超过文件尾时 ffmpeg 会在 EOF 处自然停下，不需要特殊处理。

    返回 (WAV 字节, 错误说明)；成功时第二项为空串。
    """
    if not ffmpeg:
        return None, "没有可用的 ffmpeg"
    if not data:
        return None, "音频内容为空"
    if duration <= 0:
        return None, "裁剪时长必须为正数"
    begin = max(0.0, float(start))
    end = begin + float(duration)
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", "%.3f" % begin,
            "-to", "%.3f" % end,
            "-i", "pipe:0",
            "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
            "-f", "wav", "pipe:1"]
    return run(args, input_bytes=data, timeout=timeout)


def probe_duration(data: bytes, ffmpeg: str = "",
                   timeout: int = 120) -> Tuple[float, str]:
    """读出一段音频的时长（秒）。拿不到就返回 (0.0, 原因)。

    **为什么不能用 `ffmpeg -i pipe:0` 解析 Duration 行**：从管道读时
    没有可 seek 的文件头，ffmpeg 只报 `Duration: N/A`（实测过）。
    所以这里改用**解码到底、读最后一帧的 time 值**——虽然要跑一遍解码，
    但结果是可信的；本机也没有 ffprobe（imageio-ffmpeg 不带），
    所以就按这条唯一可靠的路走。

    调用方优先用 B 站接口给的视频时长，这个函数是"拿不到时才兜底"。
    """
    if not ffmpeg:
        return 0.0, "没有可用的 ffmpeg"
    if not data:
        return 0.0, "音频内容为空"
    # 先落到临时文件：从管道读的音频没有可 seek 的文件头，
    # ffmpeg 会报 `Duration: N/A`，用 -f null 解码到底也拿不到 time= 行。
    # 写文件后 ffmpeg 就能读到真实时长（实测 435.54 秒的 m4s 一次读对）。
    handle = None
    try:
        handle = tempfile.NamedTemporaryFile(prefix="alice_probe_", suffix=".bin",
                                             delete=False)
        handle.write(data)
        handle.flush()
        handle.close()
        path = handle.name
    except OSError as exc:
        return 0.0, f"无法写临时文件：{type(exc).__name__}"
    try:
        # -f null -：解码但不写任何输出，最后 stderr 会给出 "time=00:00:03.00"。
        # 必须带 -stats：不带的话 ffmpeg 不打印这一行（实测），也就读不到时长。
        proc = subprocess.run([ffmpeg, "-hide_banner", "-stats", "-nostdin",
                               "-i", path, "-f", "null", "-"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout)
        text = (proc.stderr or b"").decode("utf-8", "ignore")
    except subprocess.TimeoutExpired:
        return 0.0, f"读取时长超时（{timeout} 秒）"
    except FileNotFoundError:
        return 0.0, "找不到 ffmpeg 可执行文件"
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"
    finally:
        # 临时文件必须删掉。之前忘了删，工作区里攒了 5 个 tmp* 目录、155MB，
        # 所以这里用 finally 保证任何分支都会清理。
        try:
            os.unlink(path)
        except OSError:
            pass
    times = _TIME_PATTERN.findall(text)
    if not times:
        return 0.0, "解析不出音频时长"
    hours, minutes, seconds = times[-1]
    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds), ""
    except ValueError:
        return 0.0, "解析不出音频时长"


# ---------------------------------------------------------------------------
# 视频
# ---------------------------------------------------------------------------

def _with_temp_file(data: bytes, suffix: str):
    """把字节落到临时文件，返回 (路径, 错误说明)。

    **视频不能用管道喂给 ffmpeg**（和 probe_duration 里记的是同一个原因）：
    mp4 的 moov box 可能在文件尾部，需要可 seek 才能解析；
    从管道读会得到"moov atom not found"或者直接解不出流。
    """
    if not data:
        return "", "视频内容为空"
    handle = None
    try:
        handle = tempfile.NamedTemporaryFile(prefix="alice_video_", suffix=suffix,
                                             delete=False)
        handle.write(data)
        handle.flush()
        handle.close()
        return handle.name, ""
    except OSError as exc:
        if handle is not None:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
        return "", f"无法写临时文件：{type(exc).__name__}"


def extract_video_audio(data: bytes, ffmpeg: str = "", sample_rate: int = 16000,
                        start: float = 0.0, duration: float = 0.0,
                        timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[bytes], str]:
    """抽出视频的音轨，输出 16kHz 单声道 WAV。

    很多视频**没有音轨**（纯画面）。这种情况 ffmpeg 会因为没有输出流而
    返回非零，这里把它当成"没有音频"而不是错误——调用方据此只做画面理解。
    """
    if not ffmpeg:
        return None, "没有可用的 ffmpeg"
    path, err = _with_temp_file(data, ".mp4")
    if err:
        return None, err
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start > 0:
        args += ["-ss", "%.3f" % float(start)]
    if duration > 0:
        args += ["-t", "%.3f" % float(duration)]
    args += ["-i", path, "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
             "-f", "wav", "pipe:1"]
    try:
        out, run_err = run(args, timeout=timeout)
        if run_err:
            # 没有音轨是最常见的原因，单独说明，别让调用方以为"坏了"
            return None, "视频没有音轨或音轨无法解码"
        return out, ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def extract_video_frames(data: bytes, ffmpeg: str = "", count: int = 3,
                         width: int = 640, start: float = 0.0,
                         duration: float = 0.0, quality: int = 4,
                         timeout: int = DEFAULT_TIMEOUT) -> Tuple[List[bytes], str]:
    """从视频里均匀抽 `count` 帧，返回 JPEG 字节列表。

    **为什么压缩到 width 宽**：视觉接口按图算钱，一张 1080p JPEG 可能
    好几 MB，base64 之后还要涨 1/3。等比例缩到 640 宽之后每帧通常
    几十 KB，画面内容对"看懂发生了什么"完全够用。

    为什么要抽多帧而不是一张：短视频里画面变化很快
    （"谁在做什么"往往看不出来），多抽几帧成本增加有限。
    """
    if not ffmpeg:
        return [], "没有可用的 ffmpeg"
    if count <= 0:
        return [], ""
    path, err = _with_temp_file(data, ".mp4")
    if err:
        return [], err
    try:
        frames: List[bytes] = []
        seek = max(0.0, float(start)) + 0.1     # 避开可能的黑场首帧
        step = (float(duration) / (count + 1)) if duration and duration > 0 else 1.0
        for i in range(count):
            args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-ss", "%.3f" % (seek + step * i),
                    "-i", path,
                    "-frames:v", "1",
                    "-vf", "scale=%d:-2" % int(width),
                    "-q:v", str(int(quality)),
                    "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
            out, _ = run(args, timeout=timeout)
            if out:
                frames.append(out)
        if not frames:
            return [], "抽不出画面帧（可能不是有效视频）"
        return frames, ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
