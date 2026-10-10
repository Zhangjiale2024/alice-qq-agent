"""ffmpeg 定位、裁剪、时长探测的测试。

全部离线，不依赖真实 ffmpeg：
- 定位逻辑用临时目录和假的目录内容验证优先级；
- 裁剪用打桩的 subprocess.run 验证命令拼装（真实 ffmpeg 只在装了的时候跑一条冒烟用例）。

**可选依赖**：只有"用真实音频验证切段位置"那一条用例需要
`numpy` 与 `soundfile`（要解码 WAV 做滑窗比对）。核心逻辑本身是纯标准库，
所以这两个库**故意做成可选**——没装时那几条用例跳过，其余照跑。

这个处理是被 CI 逼出来的：在干净环境里跑 `python -m unittest discover`
会因为在模块顶层 `import numpy` 而**整文件收集失败**，
表现为"test_ffmpeg 加载不出来"，而不是清晰的"缺 numpy"。
"""
from __future__ import annotations

import io
import math
import os
import struct
import subprocess
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

# 可选依赖：没装就跳过少数需要解码音频的用例，而不是让整个文件收集失败
try:
    import numpy as np
except ImportError:                      # pragma: no cover
    np = None
try:
    import soundfile as sf
except ImportError:                      # pragma: no cover
    sf = None

HAS_AUDIO_LIBS = np is not None and sf is not None
NO_AUDIO_LIBS = "需要 numpy 与 soundfile（用于解码音频做比对），当前未安装"

from aliceqq import ffmpeg as ffmpeg_mod
from aliceqq.ffmpeg import extract_segment, find_ffmpeg, probe_duration, run


class _FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FindFfmpegTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        # 把"已知目录"限制到临时目录，避免测试机上的真实 ffmpeg 干扰断言
        patcher = mock.patch.object(ffmpeg_mod, "KNOWN_DIRS", (self.dir,))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make(self, name="ffmpeg-win-x86_64-v7.1.exe"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(b"MZ")
        return path

    def test_explicit_path_wins(self):
        path = self._make()
        self.assertEqual(find_ffmpeg(explicit=path), path)

    def test_falls_back_to_known_dir(self):
        """imageio-ffmpeg 的文件名带版本号，要能扫出来。"""
        path = self._make()
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(), path)

    def test_path_takes_priority_over_known_dir(self):
        self._make()
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=r"C:\x\ffmpeg.exe"):
            self.assertEqual(find_ffmpeg(), r"C:\x\ffmpeg.exe")

    def test_config_path_is_used(self):
        path = self._make()
        config = SimpleNamespace(raw_env={"FFMPEG": path})
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(config), path)

    def test_legacy_tts_ffmpeg_key_still_honoured(self):
        """老配置项 TTS_FFMPEG 也要认，不能因为改名就失效。"""
        path = self._make()
        config = SimpleNamespace(raw_env={"TTS_FFMPEG": path})
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(config), path)

    def test_missing_config_path_is_ignored(self):
        config = SimpleNamespace(raw_env={"FFMPEG": os.path.join(self.dir, "没有这个.exe")})
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(config), "")

    def test_nothing_found_returns_empty(self):
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(), "")

    def test_config_without_raw_env_is_safe(self):
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(SimpleNamespace()), "")

    def test_non_ffmpeg_files_in_dir_are_ignored(self):
        self._make("ffprobe.exe")
        self._make("README.md")
        with mock.patch.object(ffmpeg_mod.shutil, "which", return_value=None):
            self.assertEqual(find_ffmpeg(), "")


class RunTest(unittest.TestCase):
    def test_success(self):
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               return_value=_FakeCompleted(0, b"audio", b"")):
            out, err = run(["ffmpeg"])
        self.assertEqual(out, b"audio")
        self.assertEqual(err, "")

    def test_nonzero_exit_reports_last_stderr_line(self):
        stderr = b"line one\nInvalid argument\n"
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               return_value=_FakeCompleted(1, b"", stderr)):
            out, err = run(["ffmpeg"])
        self.assertIsNone(out)
        self.assertIn("Invalid argument", err)
        self.assertIn("退出码 1", err)

    def test_empty_output_is_reported(self):
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               return_value=_FakeCompleted(0, b"", b"")):
            out, err = run(["ffmpeg"])
        self.assertIsNone(out)
        self.assertIn("没有输出", err)

    def test_missing_binary(self):
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               side_effect=FileNotFoundError()):
            out, err = run(["ffmpeg"])
        self.assertIsNone(out)
        self.assertIn("找不到", err)

    def test_timeout(self):
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ffmpeg", 5)):
            out, err = run(["ffmpeg"], timeout=5)
        self.assertIsNone(out)
        self.assertIn("超时", err)


class ExtractSegmentTest(unittest.TestCase):
    def test_uses_absolute_end_and_reencodes(self):
        """用 `-ss`+`-to`（绝对秒），并且**不做** `-c copy`。

        两个回归点（见 aliceqq/ffmpeg.py 的说明）：
        1. `-t` 在 `-ss` 之后会被当成绝对结束时间，导致第 2 段起被截断；
        2. `-c copy` 只能从关键帧切，实测 [300, +300) 会得到 430 秒而不是 135 秒。
        """
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return _FakeCompleted(0, b"wav", b"")

        with mock.patch.object(ffmpeg_mod.subprocess, "run", side_effect=fake_run):
            out, err = extract_segment(b"data", 300.0, 300.0, ffmpeg="ffmpeg.exe")
        self.assertEqual(out, b"wav")
        self.assertEqual(err, "")
        args = calls[0]
        self.assertIn("-to", args)
        self.assertNotIn("-t", args)
        self.assertNotIn("copy", args)
        self.assertIn("wav", args)
        self.assertEqual(args[args.index("-to") + 1], "600.000")
        self.assertEqual(args[args.index("-ss") + 1], "300.000")
        self.assertLess(args.index("-ss"), args.index("-i"))
        self.assertLess(args.index("-to"), args.index("-i"))

    def test_no_ffmpeg_returns_reason(self):
        out, err = extract_segment(b"data", 0, 10, ffmpeg="")
        self.assertIsNone(out)
        self.assertIn("ffmpeg", err)

    def test_empty_audio_rejected(self):
        out, err = extract_segment(b"", 0, 10, ffmpeg="ffmpeg.exe")
        self.assertIsNone(out)
        self.assertIn("空", err)

    def test_non_positive_duration_rejected(self):
        out, err = extract_segment(b"data", 0, 0, ffmpeg="ffmpeg.exe")
        self.assertIsNone(out)
        self.assertIn("正数", err)

    def test_negative_start_is_clamped(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return _FakeCompleted(0, b"x", b"")

        with mock.patch.object(ffmpeg_mod.subprocess, "run", side_effect=fake_run):
            extract_segment(b"data", -5, 10, ffmpeg="ffmpeg.exe")
        self.assertEqual(calls[0][calls[0].index("-ss") + 1], "0.000")
        self.assertEqual(calls[0][calls[0].index("-to") + 1], "10.000")


class ProbeDurationTest(unittest.TestCase):
    def test_parses_time_from_decode_run(self):
        """时长靠解码到底读 time= 行——从管道读时 ffmpeg 只报 Duration: N/A。"""
        stderr = (b"Input #0, mov,mp4,m4a, from 'pipe:0':\n"
                  b"  Duration: N/A, bitrate: 68 kb/s\n"
                  b"size=N/A time=00:00:10.00 bitrate=N/A speed= 195x\n"
                  b"size=N/A time=00:03:42.50 bitrate=N/A speed= 368x\n")
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               return_value=_FakeCompleted(0, b"", stderr)):
            seconds, err = probe_duration(b"data", ffmpeg="ffmpeg.exe")
        self.assertEqual(err, "")
        # 取**最后一处** time=：那是解码结束的总时长
        self.assertAlmostEqual(seconds, 222.5, places=2)

    def test_input_is_passed_as_a_real_file(self):
        """要从**文件**读，不能走管道（管道上 ffmpeg 读不出时长）。"""
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return _FakeCompleted(0, b"", b"time=00:00:03.00\n")

        with mock.patch.object(ffmpeg_mod.subprocess, "run", side_effect=fake_run):
            probe_duration(b"data", ffmpeg="ffmpeg.exe")
        args = calls[0]
        self.assertNotIn("pipe:0", args)
        # -i 后面跟的是一个真实存在的临时文件路径
        path = args[args.index("-i") + 1]
        self.assertTrue(os.path.isabs(path))

    def test_temporary_file_is_cleaned_up(self):
        """回归：临时文件必须删掉。

        早期实现忘了删，工作区里攒了 5 个 tmp* 目录共 155MB。
        """
        seen = {}

        def fake_run(args, **kwargs):
            seen["path"] = args[args.index("-i") + 1]
            self.assertTrue(os.path.isfile(seen["path"]), "跑之前文件应存在")
            return _FakeCompleted(0, b"", b"time=00:00:03.00\n")

        with mock.patch.object(ffmpeg_mod.subprocess, "run", side_effect=fake_run):
            probe_duration(b"data", ffmpeg="ffmpeg.exe")
        self.assertFalse(os.path.exists(seen["path"]), "临时文件应被删除")

    def test_temporary_file_is_cleaned_up_on_failure(self):
        """出错路径也要清理，不能只在成功时清。"""
        seen = {}

        def fake_run(args, **kwargs):
            seen["path"] = args[args.index("-i") + 1]
            raise subprocess.TimeoutExpired("ffmpeg", 5)

        with mock.patch.object(ffmpeg_mod.subprocess, "run", side_effect=fake_run):
            seconds, err = probe_duration(b"data", ffmpeg="ffmpeg.exe", timeout=5)
        self.assertEqual(seconds, 0.0)
        self.assertFalse(os.path.exists(seen["path"]), "失败时也要删临时文件")

    def test_duration_na_alone_is_not_enough(self):
        """只有 Duration: N/A 时必须如实说解析不出，而不是返回 0 当成功。"""
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               return_value=_FakeCompleted(0, b"",
                                                           b"Duration: N/A, bitrate: 1 kb/s\n")):
            seconds, err = probe_duration(b"data", ffmpeg="ffmpeg.exe")
        self.assertEqual(seconds, 0.0)
        self.assertTrue(err)

    def test_no_ffmpeg(self):
        seconds, err = probe_duration(b"data", ffmpeg="")
        self.assertEqual(seconds, 0.0)
        self.assertIn("ffmpeg", err)

    def test_empty_audio(self):
        seconds, err = probe_duration(b"", ffmpeg="ffmpeg.exe")
        self.assertEqual(seconds, 0.0)
        self.assertIn("空", err)

    def test_timeout_reported(self):
        with mock.patch.object(ffmpeg_mod.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ffmpeg", 5)):
            seconds, err = probe_duration(b"data", ffmpeg="ffmpeg.exe", timeout=5)
        self.assertEqual(seconds, 0.0)
        self.assertIn("超时", err)


@unittest.skipUnless(find_ffmpeg(), "本机没有 ffmpeg，跳过真实冒烟测试")
class RealFfmpegSmokeTest(unittest.TestCase):
    """真 ffmpeg 的冒烟测试。

    刻意复现生产里的形状：B 站音频是 **AAC 装在 mp4/m4s 容器里**。
    一开始我拿 wave 模块生成的裸 PCM 当输入，`-c copy` 会产出一个坏掉的
    matroska（裸 PCM 没有容器头可直接重封装），测试因此失败——
    这不是产品代码的问题，而是测试样本不真实。所以这里先用 ffmpeg
    把正弦波编码成 AAC/mp4，再走被测代码。
    """

    @classmethod
    def setUpClass(cls):
        cls.exe = find_ffmpeg()

    def _tone_wav(self, seconds=3.0, rate=16000, silent_head=0.0):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            frames = []
            for i in range(int(seconds * rate)):
                amp = 0 if i < int(silent_head * rate) else 12000
                frames.append(struct.pack(
                    "<h", int(amp * math.sin(2 * math.pi * 440 * i / rate))))
            fh.writeframes(b"".join(frames))
        return buf.getvalue()

    def _aac_mp4(self, seconds=3.0):
        """把正弦波编成 AAC/mp4 —— 与 B 站音频同构。"""
        wav = self._tone_wav(seconds)
        proc = subprocess.run(
            [self.exe, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "wav", "-i", "pipe:0", "-c:a", "aac", "-b:a", "64k",
             "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
            input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "ignore")[-300:])
        return proc.stdout

    def test_segment_has_exact_requested_length(self):
        """切出来的片段时长必须**精确**等于请求值，不能只是"差不多"。"""
        data = self._aac_mp4(6.0)
        self.assertTrue(data)
        chunk, err = extract_segment(data, 2.0, 2.0, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertTrue(chunk)
        length, err = probe_duration(chunk, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 2.0, delta=0.25)

    def test_output_is_wav(self):
        chunk, err = extract_segment(self._tone_wav(3.0), 1.0, 1.0, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertTrue(chunk.startswith(b"RIFF"), "输出应是标准 WAV")

    def test_probe_reads_real_duration(self):
        data = self._aac_mp4(4.0)
        length, err = probe_duration(data, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 4.0, delta=0.5)

    def test_second_segment_is_not_truncated(self):
        """回归：第 2 段不能被截断（`-t` 被当成绝对结束时间那个坑）。

        用 8 秒音频取 [5, +5)：正确结果是 3 秒（到文件尾）；
        如果误用 `-t`，会得到 0 秒或整段，都会在这里暴露。
        """
        data = self._aac_mp4(8.0)
        chunk, err = extract_segment(data, 5.0, 5.0, ffmpeg=self.exe)
        self.assertEqual(err, "")
        length, err = probe_duration(chunk, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 3.0, delta=0.6)

    def test_back_to_back_segments_cover_the_whole_file(self):
        """连续切段不应丢内容：各段时长之和要接近整段时长。"""
        data = self._aac_mp4(6.0)
        total, _ = probe_duration(data, ffmpeg=self.exe)
        covered = 0.0
        for start in (0.0, 2.0, 4.0):
            chunk, err = extract_segment(data, start, 2.0, ffmpeg=self.exe)
            self.assertEqual(err, "")
            length, err = probe_duration(chunk, ffmpeg=self.exe)
            self.assertEqual(err, "")
            covered += length
        self.assertAlmostEqual(covered, total, delta=1.0)


_REAL_BILI = r"D:\DSHWORK\models_shared\bili\ja_tanaka_BV14v4y1e7QC.m4s"


@unittest.skipUnless(find_ffmpeg() and os.path.isfile(_REAL_BILI),
                     "本机没有真实 B 站音频样本，跳过")
class RealBilibiliFileTest(unittest.TestCase):
    """用真实 B 站 m4s 验证切段——当初就是这个文件暴露了 `-t` 那个坑。"""

    @classmethod
    def setUpClass(cls):
        cls.exe = find_ffmpeg()
        with open(_REAL_BILI, "rb") as fh:
            cls.data = fh.read()

    def test_duration_matches_known_length(self):
        length, err = probe_duration(self.data, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 435.5, delta=2.0)

    def test_middle_segment_exact_length(self):
        """取 [200, +120) 必须是 120 秒——用 `-t` 的话只会得到 120 秒也算对，
        所以这里再取一个"结束位置超过文件尾"的区间来卡住那个坑。"""
        chunk, err = extract_segment(self.data, 200.0, 120.0, ffmpeg=self.exe)
        self.assertEqual(err, "")
        length, err = probe_duration(chunk, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 120.0, delta=2.0)

    def test_segment_running_past_eof_stops_at_eof(self):
        """[300, +300) 应该得到 135.5 秒（到文件尾），而不是 600 秒或 0 秒。"""
        chunk, err = extract_segment(self.data, 300.0, 300.0, ffmpeg=self.exe)
        self.assertEqual(err, "")
        length, err = probe_duration(chunk, ffmpeg=self.exe)
        self.assertEqual(err, "")
        self.assertAlmostEqual(length, 135.5, delta=2.0)

    @unittest.skipUnless(HAS_AUDIO_LIBS, NO_AUDIO_LIBS)
    def test_seek_lands_at_the_right_place(self):
        """切出来的内容必须真的来自 start 处。

        做法：切 [299, +1)，再把整段解码出来，用滑窗找这一秒最像的位置——
        它应该落回 299.0 秒。只比时长是不够的（ffmpeg 可能返回了别处的 1 秒）。
        """
        seg, err = extract_segment(self.data, 299.0, 1.0, ffmpeg=self.exe,
                                   sample_rate=16000)
        self.assertEqual(err, "")
        whole, err = extract_segment(self.data, 0.0, 435.5, ffmpeg=self.exe,
                                     sample_rate=16000)
        self.assertEqual(err, "")
        target, _ = sf.read(io.BytesIO(seg), dtype="float32", always_2d=True)
        full, sr = sf.read(io.BytesIO(whole), dtype="float32", always_2d=True)
        target = target.mean(axis=1)
        full = full.mean(axis=1)
        n = len(target)
        best = None
        step = sr // 4                      # 0.25 秒的分辨率足够
        for offset in range(0, len(full) - n, step):
            diff = float(np.abs(full[offset:offset + n] - target).mean())
            if best is None or diff < best[1]:
                best = (offset, diff)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best[0] / sr, 299.0, delta=0.3)
        self.assertLess(best[1], 1e-4, "切出来的内容与源音频对应位置不一致")


if __name__ == "__main__":
    unittest.main(verbosity=2)
