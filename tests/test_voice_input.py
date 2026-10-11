# -*- coding: utf-8 -*-
"""语音理解（多模态输入的另一半）测试。

**全部离线**：ASR 用假的转写器，不发真实请求。

这一套测试守着的是：用户发的语音要真的到达模型。
在此之前，`asr.py` 的能力只用在"转写 B 站视频音轨"上，
用户直接发来的语音消息在适配层就没被抽出来——
模型收到的是一条空文本，表现是"发语音它不回"。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq import asr as asr_mod
from aliceqq import policy
from aliceqq.config import AppConfig, VisionConfig
from aliceqq.engine import IncomingMessage
from tests.test_engine import make_engine

FAKE_AUDIO = b"OggS" + b"\x00" * 200


class _FakeASR:
    """假的转写器。记录调用，返回预设结果。"""

    def __init__(self, text="艾利斯今天想去打游戏", ok=True, error="",
                 avg_logprob=None, no_speech_prob=None):
        self.text = text
        self.ok = ok
        self.error = error
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.calls = []

    available = True

    def transcribe(self, audio, filename="audio.m4a"):
        self.calls.append({"bytes": len(audio), "filename": filename})
        return asr_mod.Transcription(
            self.ok, text=self.text, provider="fake", error=self.error,
            seconds=3.0, bytes_sent=len(audio),
            avg_logprob=self.avg_logprob, no_speech_prob=self.no_speech_prob)


class AudioRefTest(unittest.TestCase):
    def test_blob_is_used_directly(self):
        ref = asr_mod.AudioRef(blob=FAKE_AUDIO)
        self.assertEqual(ref.read_bytes(), FAKE_AUDIO)

    def test_local_path_is_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.m4a"
            p.write_bytes(FAKE_AUDIO)
            self.assertEqual(asr_mod.AudioRef(path=str(p)).read_bytes(), FAKE_AUDIO)

    def test_missing_source_yields_empty(self):
        self.assertEqual(asr_mod.AudioRef().read_bytes(), b"")

    def test_nonexistent_file_yields_empty(self):
        self.assertEqual(asr_mod.AudioRef(path=r"D:\nope\v.m4a").read_bytes(), b"")

    def test_oversized_audio_is_rejected(self):
        # 语音消息场景给 2MB 上限（视频音轨那套是 8MB，场景不同）
        big = FAKE_AUDIO + b"\x00" * asr_mod.VOICE_MESSAGE_MAX_BYTES
        ref = asr_mod.AudioRef(blob=big)
        self.assertEqual(ref.read_bytes(), b"",
                         "超过上限的音频不该被送进识别")


class MessageCarriesAudioTest(unittest.TestCase):
    def test_message_can_carry_audio(self):
        msg = IncomingMessage(user_id="1", text="", audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        self.assertTrue(msg.has_audios)
        self.assertTrue(msg.has_media_input)

    def test_plain_message_has_no_media_input(self):
        msg = IncomingMessage(user_id="1", text="你好")
        self.assertFalse(msg.has_audios)
        self.assertFalse(msg.has_media_input)


class EmptyTextWithMediaTest(unittest.TestCase):
    """**这一组是实测 bug 的回归守卫。**

    用户直接甩一张图、或者发语音不说话，text 都是空的。
    原来的 `if not text: return` 会让这类消息被当成"空消息"丢掉，
    后面的图片理解与语音识别一次都不会执行。
    """

    def test_voice_only_message_is_not_dropped(self):
        engine = make_engine()
        engine.asr_client = _FakeASR()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        outcome = engine.handle(msg)
        self.assertNotEqual(outcome.reason, "空消息",
                            "纯语音消息被当成空消息丢掉了")
        self.assertTrue(outcome.debug.get("asr", {}).get("ok"))

    def test_image_only_message_is_not_dropped(self):
        from aliceqq import vision

        engine = make_engine()
        engine.vision_client = None          # 看不到也要走完流程
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)])
        outcome = engine.handle(msg)
        self.assertNotEqual(outcome.reason, "空消息",
                            "纯图片消息被当成空消息丢掉了")
        self.assertIn("vision", outcome.debug)

    def test_truly_empty_message_is_still_dropped(self):
        # 没有媒体、也没有文字，仍然要丢——这个行为不能被上面的修复改坏
        engine = make_engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x")
        outcome = engine.handle(msg)
        self.assertEqual(outcome.reason, "空消息")


class TranscriptBlockTest(unittest.TestCase):
    def _engine(self, fake=None):
        engine = make_engine()
        engine.asr_client = fake or _FakeASR()
        return engine

    def test_no_audio_means_no_block(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="你好", is_private=True, user_name="x")
        self.assertEqual(engine._transcript_block(msg), "")

    def test_transcript_is_injected(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        block = engine._transcript_block(msg)
        self.assertIn("艾利斯今天想去打游戏", block)
        self.assertIn("语音", block)

    def test_unavailable_asr_tells_model_it_could_not_hear(self):
        # 听不到也必须注入，否则模型会以为用户没说话，转而瞎猜
        engine = make_engine()
        engine.asr_client = None
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        block = engine._transcript_block(msg)
        self.assertIn("没有听到内容", block)
        self.assertIn("不要猜", block)

    def test_transcription_failure_reports_instead_of_guessing(self):
        engine = self._engine(_FakeASR(ok=False, error="模型加载失败"))
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        block = engine._transcript_block(msg)
        self.assertIn("模型加载失败", block)
        self.assertIn("没有听到内容", block)

    def test_unreadable_audio_is_reported(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=b"")])
        outcome = engine.handle(msg)
        self.assertFalse(outcome.debug["asr"]["ok"])
        self.assertIn("没有听到内容", engine._transcript_block(msg))

    def test_quality_warning_is_passed_through(self):
        # ASR 在噪声上会输出"通顺但完全错误"的文本，光看文字看不出来。
        # 质量指标不行时必须把警告带给模型，否则等于把错误当用户原话。
        engine = self._engine(_FakeASR(text="嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯",
                                       avg_logprob=-1.9, no_speech_prob=0.9))
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        block = engine._transcript_block(msg)
        self.assertIn("注意", block, "识别质量差时应当带上警告")

    def test_transcript_does_not_leak_across_turns(self):
        # 把 A 的语音转写带进 B 的对话，等于泄漏他说过的话
        engine = self._engine()
        first = IncomingMessage(user_id="1", text="", is_private=True, user_name="a",
                                audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(first)
        self.assertIn("打游戏", engine._transcript_block(first))

        second = IncomingMessage(user_id="2", text="你好", is_private=True, user_name="b")
        engine.handle(second)
        self.assertEqual(engine._transcript_block(second), "",
                         "第二轮没有语音时不能带上上一轮的转写")


class VoiceQuotaTest(unittest.TestCase):
    def _engine(self):
        engine = make_engine()
        engine.asr_client = _FakeASR()
        return engine

    def test_voice_quota_is_counted_separately_from_video_asr(self):
        # "转写视频音轨"和"听用户说一句话"用量级完全不同：
        # 前者一天几次，后者可能几十次。共用额度会导致
        # 转两段视频就把语音对话的额度吃光。
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        period = policy._period_day()
        self.assertEqual(engine.storage.get_counter("quota_asrvoice_user", "1", period), 1)
        self.assertEqual(engine.storage.get_counter("quota_asr_user", "1", period), 0,
                         "不该串到视频转写的额度上")

    def test_failed_transcription_does_not_consume_quota(self):
        engine = make_engine()
        engine.asr_client = _FakeASR(ok=False, error="boom")
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        engine.handle(msg)
        period = policy._period_day()
        self.assertEqual(engine.storage.get_counter("quota_asrvoice_user", "1", period), 0)

    def test_blocked_feature_says_so(self):
        engine = self._engine()
        engine._feature_allowed = lambda message, feature: False
        engine._last_block_reason = "测试用：额度不足"
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
        stub = type("O", (), {"debug": {}})()
        engine._analyze_audio(msg, stub)
        self.assertTrue(stub.debug["asr"].get("blocked"))
        self.assertEqual(engine.asr_client.calls, [], "被拦下时不该真的去识别")


class PromptTemplateTest(unittest.TestCase):
    def test_templates_have_transcript_placeholder(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("alice_system.md", "assistant_system.md"):
            text = (root / "prompts" / name).read_text(encoding="utf-8")
            self.assertIn("{{transcript}}", text, "%s 缺少 {{transcript}}" % name)

    def test_templates_forbid_guessing_about_voice(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("alice_system.md", "assistant_system.md"):
            text = (root / "prompts" / name).read_text(encoding="utf-8")
            i = text.find("{{transcript}}")
            section = text[max(0, i - 400):i + 700]
            self.assertIn("语音", section)
            # 两个模板措辞不同（一个写"没听清"、一个写"没听到"），
            # 断言共同点：都要求把"没听清"这件事说出来
            self.assertTrue("没听到" in section or "没听清" in section,
                            "%s 没有要求『听不清就说出来』" % name)
            self.assertIn("不要猜", section,
                          "%s 没有禁止在听不到时猜测内容" % name)


class UnifiedInputTest(unittest.TestCase):
    """图片和语音由同一个入口处理。"""

    def test_analyze_inputs_handles_both(self):
        from aliceqq import vision
        import json

        engine = make_engine()
        engine.config.vision = VisionConfig(enabled=True, base_url="https://x/v1",
                                            api_key="k", model="m")
        engine.vision_client = vision.build_client(engine.config.vision)
        engine.asr_client = _FakeASR()

        original = vision.net.open

        class _Resp:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "一张猫的照片"}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        vision.net.open = lambda request, timeout=None: _Resp()
        try:
            msg = IncomingMessage(
                user_id="1", text="", is_private=True, user_name="x",
                images=[vision.ImageRef(blob=b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)],
                audios=[asr_mod.AudioRef(blob=FAKE_AUDIO)])
            outcome = engine.handle(msg)
        finally:
            vision.net.open = original

        self.assertTrue(outcome.debug.get("vision", {}).get("ok"))
        self.assertTrue(outcome.debug.get("asr", {}).get("ok"))
        self.assertIn("猫", engine._vision_block(msg))
        self.assertIn("打游戏", engine._transcript_block(msg))

    def test_analyze_inputs_is_the_only_call_site(self):
        # 收敛成一个入口，以后加视频输入时不用改多处调用点
        src = (Path(__file__).resolve().parent.parent /
               "aliceqq" / "engine.py").read_text(encoding="utf-8")
        self.assertIn("self._analyze_inputs(message, outcome)", src)

    def test_build_asr_client_reads_outer_config(self):
        # 实测踩过的混淆点：asr.build_client 收**外层** AppConfig，
        # 而 vision.build_client 收子配置（config.vision）。
        # 两者语义不同，写错会**静默返回 None**。
        import inspect

        src = inspect.getsource(asr_mod.build_client)
        self.assertIn('getattr(config, "asr"', src,
                      "asr.build_client 应当是取 config.asr 的形式")


if __name__ == "__main__":
    unittest.main()
