"""变调层 PitchShiftTTS 的测试。

背景：GPT-SoVITS 的输出音调跟随参考有上限（实测参考 F0 526 Hz 时
输出只到 399 Hz），想把音色推得更"夹/元气"只能直接变调。
实现走 ffmpeg 的 asetrate+aresample+atempo，应当**只改音调、不改语速**。

只用标准库：本项目测试环境是纯标准库的 Python，numpy/soundfile 不可用。
纯正弦可以用**过零率**精确测频率（每周期两次过零），比 FFT 还准。
"""
from __future__ import annotations

import array
import io
import math
import os
import shutil
import struct
import unittest
import wave

from aliceqq.generation import PitchShiftTTS, SpeechResult


def _sine_wav(freq=200.0, seconds=2.0, sr=32000) -> bytes:
    frames = int(sr * seconds)
    data = array.array("h")
    for i in range(frames):
        value = math.sin(2 * math.pi * freq * i / sr) * 0.8
        data.append(int(value * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


def _measure(blob: bytes):
    """返回 (主频, 时长秒)。用带滞回的过零计数，避免噪声抖动。"""
    with wave.open(io.BytesIO(blob)) as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
    duration = frames / sr
    if not samples:
        return 0.0, duration
    peak = max(abs(min(samples)), abs(max(samples))) or 1
    high, low = peak * 0.5, -peak * 0.5
    # 带滞回地统计上下穿越总数；一个完整周期有 2 次，所以频率 = 穿越数 / 2 / 时长。
    # 早先写成 state<=0 才计下行穿越，而 state=1 时该条件恒假，于是一次都没计到。
    crossings, state = 0, 0        # state: 1=已过上半，-1=已过下半
    for value in samples:
        if value > high and state != 1:
            if state == -1:
                crossings += 1
            state = 1
        elif value < low and state != -1:
            if state == 1:
                crossings += 1
            state = -1
    return ((crossings / 2.0 / duration) if duration else 0.0), duration


class _FakeTTS:
    """返回固定频率正弦波，便于精确验证频率变化。"""

    name = "fake"
    available = True

    def __init__(self, freq=200.0, seconds=2.0, sr=32000):
        self.freq, self.seconds, self.sr = freq, seconds, sr
        self.blob = _sine_wav(freq, seconds, sr)

    def synthesize(self, text, emotion=""):
        return SpeechResult(True, data=self.blob, mime="audio/wav", provider="fake")


def _has_ffmpeg() -> bool:
    try:
        return bool(PitchShiftTTS(_FakeTTS(), semitones=1).available)
    except Exception:
        return False


@unittest.skipUnless(_has_ffmpeg(), "环境里没有 ffmpeg")
class PitchShiftTest(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "test-pitch")
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _render(self, semitones):
        shifter = PitchShiftTTS(_FakeTTS(), semitones=semitones)
        result = shifter.synthesize("测试")
        self.assertTrue(result.ok, result.error)
        freq, duration = _measure(result.data)
        return freq, duration, result

    def test_pitch_rises(self):
        """+2 半音应当是 200 * 2^(2/12) ≈ 224.5 Hz。"""
        freq, _, _ = self._render(2.0)
        self.assertAlmostEqual(freq, 200.0 * (2 ** (2 / 12)), delta=6.0)

    def test_pitch_falls(self):
        freq, _, _ = self._render(-3.0)
        self.assertAlmostEqual(freq, 200.0 * (2 ** (-3 / 12)), delta=6.0)

    def test_duration_is_preserved(self):
        """关键：只改音调、不改语速。"""
        _, duration, _ = self._render(3.0)
        self.assertAlmostEqual(duration, 2.0, delta=0.08)

    def test_provider_marks_pitch(self):
        _, _, result = self._render(2.0)
        self.assertIn("pitch", result.provider)


class PitchShiftGuardTest(unittest.TestCase):
    """半音为 0 / 没有 ffmpeg / 内层失败时的行为必须诚实。"""

    def test_zero_semitones_is_transparent(self):
        self.assertFalse(PitchShiftTTS(_FakeTTS(), semitones=0.0).available)

    def test_missing_ffmpeg_reports_honestly(self):
        shifter = PitchShiftTTS(_FakeTTS(), semitones=2.0, ffmpeg="D:/不存在/ffmpeg.exe")
        if shifter._ffmpeg_path():
            self.skipTest("环境 PATH 里有 ffmpeg，无法构造缺失场景")
        self.assertFalse(shifter.available)
        result = shifter.synthesize("测试")
        self.assertTrue(result.ok, "应退回未变调音频而不是失败")
        self.assertIn("变调失败", result.error)

    def test_inner_failure_passes_through(self):
        class Broken:
            name = "broken"
            available = True

            def synthesize(self, text, emotion=""):
                return SpeechResult(False, provider="broken", error="内层失败")

        result = PitchShiftTTS(Broken(), semitones=2.0).synthesize("测试")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "内层失败")


if __name__ == "__main__":
    unittest.main(verbosity=2)
