# -*- coding: utf-8 -*-
"""`--trace` 决策链输出的测试（全部离线）。

**为什么要专门测这个。**
决策链是"可观测性"，而多模态是最容易**静默失效**的地方——
最近两轮各抓到一类，两次都没有任何报错：

  * 纯媒体消息文字为空，被当成"空消息"直接丢掉，
    图片理解和语音识别一次都没执行
  * `base64://` 形式的图片读不出来，三种模态一起失效

两次都只能靠"看引擎内部状态"发现。所以 trace 必须把多模态的处理
结果、**以及失败原因**打出来。只打"没看到"是不够的——
"看不见"和"看错了"处理方式完全不同。

这里的原则也和别处一致：**只打印真实发生过的事**，
没发图就不打图片那一行。
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aliceqq import asr as asr_mod
from aliceqq import video, vision
from aliceqq.cli import print_trace
from aliceqq.config import VisionConfig
from aliceqq.engine import IncomingMessage
from tests.test_engine import make_engine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _FakeASR:
    available = True

    def __init__(self, ok=True, text="爱丽丝今天想去打游戏"):
        self.ok, self.text = ok, text

    def transcribe(self, audio, filename="a.m4a"):
        return asr_mod.Transcription(self.ok, text=self.text if self.ok else "",
                                     provider="fake",
                                     error="" if self.ok else "模型加载失败",
                                     seconds=3.0, bytes_sent=len(audio or b""))


def capture(outcome, config=None):
    """跑一次 print_trace，把 stdout 收回来。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_trace(outcome, config)
    return buf.getvalue()


class TraceBase(unittest.TestCase):
    def _engine(self):
        engine = make_engine()
        engine.config.search.enabled = False
        engine.config.features.web_search = False
        return engine

    def _with_vision(self, description="写着 TRACE-8801"):
        import json

        engine = self._engine()
        engine.config.vision = VisionConfig(enabled=True, base_url="https://x/v1",
                                            api_key="k", model="m")
        engine.vision_client = vision.build_client(engine.config.vision)
        original = vision.net.open

        class _Resp:
            def read(self):
                return json.dumps({"choices": [
                    {"message": {"content": description}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        vision.net.open = lambda request, timeout=None: _Resp()
        self.addCleanup(lambda: setattr(vision.net, "open", original))
        return engine


class ImageTraceTest(TraceBase):
    def test_success_line_is_printed(self):
        engine = self._with_vision()
        out = engine.handle(IncomingMessage(
            user_id="1", text="看图", is_private=True, user_name="x",
            images=[vision.ImageRef(blob=PNG)]))
        text = capture(out, engine.config)
        self.assertIn("图片", text)
        self.assertIn("识别成功", text)

    def test_failure_reason_is_printed(self):
        # 只打"没看到"没用——必须说明为什么
        engine = self._engine()
        engine.vision_client = None
        out = engine.handle(IncomingMessage(
            user_id="1", text="看图", is_private=True, user_name="x",
            images=[vision.ImageRef(blob=PNG)]))
        text = capture(out, engine.config)
        self.assertIn("没有看到内容", text)
        self.assertIn("没有配置视觉模型", text)

    def test_no_image_means_no_line(self):
        # 只打印真实发生过的事
        engine = self._engine()
        out = engine.handle(IncomingMessage(user_id="1", text="你好",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        self.assertNotIn("图片     收到", text)


class AudioTraceTest(TraceBase):
    def test_success_line_is_printed(self):
        engine = self._engine()
        engine.asr_client = _FakeASR()
        out = engine.handle(IncomingMessage(
            user_id="1", text="", is_private=True, user_name="x",
            audios=[asr_mod.AudioRef(blob=b"OggS" + b"\x00" * 60)]))
        text = capture(out, engine.config)
        self.assertIn("语音", text)
        self.assertIn("转写", text)

    def test_failure_reason_is_printed(self):
        engine = self._engine()
        engine.asr_client = _FakeASR(ok=False)
        out = engine.handle(IncomingMessage(
            user_id="1", text="", is_private=True, user_name="x",
            audios=[asr_mod.AudioRef(blob=b"OggS" + b"\x00" * 60)]))
        text = capture(out, engine.config)
        self.assertIn("没有听清", text)
        self.assertIn("模型加载失败", text)

    def test_empty_text_marker_is_printed(self):
        # 这条是那个"纯媒体消息被当空消息丢掉"的 bug 的痕迹，
        # 留着让以后再出现时一眼能认出来
        engine = self._engine()
        engine.asr_client = _FakeASR()
        out = engine.handle(IncomingMessage(
            user_id="1", text="", is_private=True, user_name="x",
            audios=[asr_mod.AudioRef(blob=b"OggS" + b"\x00" * 60)]))
        text = capture(out, engine.config)
        self.assertIn("只有媒体", text)


class VideoTraceTest(TraceBase):
    def test_failure_reason_is_printed(self):
        import aliceqq.ffmpeg as ffm
        original = ffm.find_ffmpeg
        ffm.find_ffmpeg = lambda config=None, explicit="": ""
        self.addCleanup(lambda: setattr(ffm, "find_ffmpeg", original))

        engine = self._engine()
        out = engine.handle(IncomingMessage(
            user_id="1", text="看视频", is_private=True, user_name="x",
            videos=[video.VideoRef(blob=b"\x00\x00\x00\x18ftypmp42")]))
        text = capture(out, engine.config)
        self.assertIn("视频", text)
        self.assertIn("没有拿到内容", text)
        self.assertIn("ffmpeg", text)

    def test_success_separates_frames_and_audio(self):
        import aliceqq.ffmpeg as ffm
        saved = (ffm.find_ffmpeg, ffm.probe_duration,
                 ffm.extract_video_frames, ffm.extract_video_audio)
        ffm.find_ffmpeg = lambda config=None, explicit="": "ffmpeg"
        ffm.probe_duration = lambda data, exe, timeout=120: (5.0, "")
        ffm.extract_video_frames = (
            lambda data, exe, count=3, width=640, start=0.0, duration=0.0,
            quality=4, timeout=180: ([b"\xff\xd8\xff" + b"\x00" * 40], ""))
        ffm.extract_video_audio = (
            lambda data, exe, sample_rate=16000, start=0.0, duration=0.0,
            timeout=180: (b"RIFF" + b"\x00" * 40, ""))
        self.addCleanup(lambda: (
            setattr(ffm, "find_ffmpeg", saved[0]),
            setattr(ffm, "probe_duration", saved[1]),
            setattr(ffm, "extract_video_frames", saved[2]),
            setattr(ffm, "extract_video_audio", saved[3])))

        engine = self._engine()
        engine.asr_client = _FakeASR()
        out = engine.handle(IncomingMessage(
            user_id="1", text="看视频", is_private=True, user_name="x",
            videos=[video.VideoRef(blob=b"\x00\x00\x00\x18ftypmp42")]))
        text = capture(out, engine.config)
        # 视频最容易被误认为"看过全片"，所以画面和声音要分开写
        self.assertIn("抽到", text)
        self.assertIn("音轨", text)


class ToolLoopTraceTest(TraceBase):
    def test_rounds_are_printed(self):
        from tests.test_tool_loop import ScriptedLLM, _calls, _text, _tool_call
        engine = self._engine()
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"})),
                                  _text("好了。")])
        out = engine.handle(IncomingMessage(user_id="1", text="记住 x",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        self.assertIn("工具循环", text)
        self.assertIn("1 轮", text)

    def test_limit_is_disclosed(self):
        from tests.test_tool_loop import ScriptedLLM, _calls, _tool_call
        engine = self._engine()
        engine.config.llm.max_tool_rounds = 1
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"}))])
        out = engine.handle(IncomingMessage(user_id="1", text="记住 x",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        self.assertIn("撞上限", text)
        self.assertIn("可能没查完", text)

    def test_loop_error_is_printed(self):
        # 以前这里套着 `except LLMError: pass`，失败完全无声
        from aliceqq.llm import LLMError
        from tests.test_tool_loop import ScriptedLLM, _calls, _tool_call
        engine = self._engine()
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"})),
                                  LLMError("连接中断")])
        out = engine.handle(IncomingMessage(user_id="1", text="记住 x",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        self.assertIn("循环出错", text)
        self.assertIn("连接中断", text)


class TraceHonestyTest(TraceBase):
    """trace 只能打印真实发生过的事。"""

    def test_plain_text_has_no_multimodal_lines(self):
        engine = self._engine()
        out = engine.handle(IncomingMessage(user_id="1", text="你好",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        for label in ("图片     收到", "语音     收到", "视频     收到"):
            self.assertNotIn(label, text, "纯文本不该出现 %r" % label)

    def test_no_tool_call_has_no_fake_rounds(self):
        engine = self._engine()
        out = engine.handle(IncomingMessage(user_id="1", text="你好",
                                            is_private=True, user_name="x"))
        text = capture(out, engine.config)
        self.assertNotIn("工具循环", text)

    def test_trace_does_not_crash_on_minimal_outcome(self):
        class Bare:
            decision = "silent"
            reason = ""
            debug = {}
            tool_results = []
            memory_saved = []
            memory_rejected = []

        text = capture(Bare(), None)          # 不该抛异常
        self.assertIn("决策链", text)


class DeadConfigFieldTest(unittest.TestCase):
    """配置字段必须真的被用到。

    实测抓到两个**死字段**：
      * `VisionConfig.max_images_per_turn` —— 配了但 `describe` 用的是
        模块常量，改了没用
      * `vision.video_frames` —— engine 在 `getattr` 它，但字段根本不存在，
        永远拿到默认值

    两个都不会报错，只是"调不动"。这类问题比崩溃更难发现，
    所以加一组用例专门盯它。
    """

    def test_max_images_per_turn_actually_takes_effect(self):
        client = vision.build_client(VisionConfig(
            enabled=True, base_url="https://x/v1", api_key="k", model="m",
            max_images_per_turn=5))
        self.assertEqual(client.max_images, 5)

    def test_max_images_defaults_sensibly(self):
        client = vision.build_client(VisionConfig(
            enabled=True, base_url="https://x/v1", api_key="k", model="m"))
        self.assertEqual(client.max_images, vision.MAX_IMAGES_PER_TURN)

    def test_max_images_is_clamped_to_at_least_one(self):
        for bad in (0, -3):
            client = vision.build_client(VisionConfig(
                enabled=True, base_url="https://x/v1", api_key="k", model="m",
                max_images_per_turn=bad))
            self.assertGreaterEqual(client.max_images, 1)

    def test_video_frames_field_exists_and_is_used(self):
        from aliceqq.config import AppConfig

        config = AppConfig.load()
        self.assertTrue(hasattr(config.vision, "video_frames"),
                        "engine 在 getattr video_frames，字段必须存在")
        src = (ROOT / "aliceqq" / "engine.py").read_text(encoding="utf-8")
        self.assertIn("video_frames", src)

    def test_video_frames_reaches_the_analyzer(self):
        """改 `video_frames` 应当改变实际抽帧数（不是死字段）。"""
        import aliceqq.ffmpeg as ffm

        seen = {}
        saved = (ffm.find_ffmpeg, ffm.probe_duration,
                 ffm.extract_video_frames, ffm.extract_video_audio)
        ffm.find_ffmpeg = lambda config=None, explicit="": "ffmpeg"
        ffm.probe_duration = lambda data, exe, timeout=120: (5.0, "")

        def spy_frames(data, exe, count=3, width=640, start=0.0,
                       duration=0.0, quality=4, timeout=180):
            seen["count"] = count
            return [b"\xff\xd8\xff" + b"\x00" * 20], ""

        ffm.extract_video_frames = spy_frames
        ffm.extract_video_audio = (
            lambda data, exe, sample_rate=16000, start=0.0, duration=0.0,
            timeout=180: (None, "no audio"))
        self.addCleanup(lambda: (
            setattr(ffm, "find_ffmpeg", saved[0]),
            setattr(ffm, "probe_duration", saved[1]),
            setattr(ffm, "extract_video_frames", saved[2]),
            setattr(ffm, "extract_video_audio", saved[3])))

        engine = make_engine()
        engine.config.vision.video_frames = 6
        engine.handle(IncomingMessage(
            user_id="1", text="看视频", is_private=True, user_name="x",
            videos=[video.VideoRef(blob=b"\x00\x00\x00\x18ftypmp42")]))
        self.assertEqual(seen.get("count"), 6,
                         "改了 video_frames 却没改变实际抽帧数——死字段")


if __name__ == "__main__":
    unittest.main()
