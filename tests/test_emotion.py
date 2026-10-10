"""情感色彩测试：推断、传递、以及"没生效要如实说"。

全部离线。
"""
from __future__ import annotations

import unittest

from aliceqq import emotion as emo
from aliceqq.config import AppConfig
from aliceqq.engine import IncomingMessage, MODE_ALICE
from aliceqq.generation import (BaiduTTSClient, GPTSoVITSTTSClient, SpeechResult,
                                VoiceConvertedTTS, describe_voice_pipeline,
                                parse_emotion_refs)
from tests.test_engine import make_engine, private

WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"0" * 200


class InferTest(unittest.TestCase):
    def test_explicit_request_wins(self):
        guess = emo.infer("用开心的语气说句话", "随便什么内容")
        self.assertEqual(guess.emotion, emo.HAPPY)
        self.assertTrue(guess.explicit)

    def test_reply_signals_detected(self):
        self.assertEqual(emo.infer("", "太好了！终于过了哈哈哈").emotion, emo.HAPPY)
        self.assertEqual(emo.infer("", "呜呜，这关太难了，好难过").emotion, emo.SAD)
        self.assertEqual(emo.infer("", "这个居然是真的吗").emotion, emo.SURPRISED)
        self.assertEqual(emo.infer("", "别怕，爱丽丝陪着你").emotion, emo.GENTLE)

    def test_neutral_when_no_signal(self):
        self.assertEqual(emo.infer("", "你说得对。").emotion, emo.NEUTRAL)

    def test_multiple_exclamations_mean_excited(self):
        self.assertEqual(emo.infer("", "冲啊！！！").emotion, emo.EXCITED)

    def test_explicit_tone_variants(self):
        self.assertEqual(emo.detect_explicit("用温柔的声音安慰我"), emo.GENTLE)
        self.assertEqual(emo.detect_explicit("语气开心一点"), emo.HAPPY)
        # 不带语气词的普通句子不应误判
        self.assertIsNone(emo.detect_explicit("今天天气怎么样"))

    def test_reply_priority_over_user_message(self):
        # 要念出来的是回复，所以回复的信号优先
        guess = emo.infer("我好难过", "哈哈哈太好了")
        self.assertEqual(guess.emotion, emo.HAPPY)

    def test_all_emotions_have_labels(self):
        for name in emo.ALL_EMOTIONS:
            self.assertIn(name, emo.LABELS)


class EmotionRefsTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_emotion_refs("happy=/v/h.wav,sad=/v/s.wav"),
                         {"happy": "/v/h.wav", "sad": "/v/s.wav"})

    def test_fullwidth_and_spaces(self):
        self.assertEqual(parse_emotion_refs("happy = /v/h.wav ， sad=/v/s.wav"),
                         {"happy": "/v/h.wav", "sad": "/v/s.wav"})

    def test_empty(self):
        self.assertEqual(parse_emotion_refs(""), {})

    def test_emotion_swaps_reference_audio(self):
        """情感靠**切换参考音频**实现。

        官方 api_v2 的参数表里没有 emo_ref_audio_path / emo_alpha
        （那是某些分支的扩展），所以换成带该情绪的参考音频及其文本，
        这在任何版本上都成立。
        """
        captured = {}

        def fake_urlopen(request, timeout=None):
            import json
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Resp(WAV)

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/main.wav",
                                    prompt_text="主参考文本",
                                    emotion_refs={"happy": ("/v/happy.wav", "开心地说")})
        self.assertTrue(client.supports_emotion)
        with _patch_urlopen(fake_urlopen):
            result = client.synthesize("你好", "happy")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(captured["body"]["ref_audio_path"], "/v/happy.wav")
        # 参考音频换了，它的文本也必须跟着换，否则语气会飘
        self.assertEqual(captured["body"]["prompt_text"], "开心地说")
        self.assertTrue(result.emotion_applied)
        # 不该发送官方不存在的参数
        self.assertNotIn("emo_ref_audio_path", captured["body"])

    def test_emotion_ref_without_prompt_keeps_main_prompt(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            import json
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Resp(WAV)

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/main.wav",
                                    prompt_text="主参考文本",
                                    emotion_refs={"sad": "/v/sad.wav"})
        with _patch_urlopen(fake_urlopen):
            client.synthesize("你好", "sad")
        self.assertEqual(captured["body"]["ref_audio_path"], "/v/sad.wav")
        self.assertEqual(captured["body"]["prompt_text"], "主参考文本")

    def test_gptsovits_without_that_emotion_says_so(self):
        def fake_urlopen(request, timeout=None):
            return _Resp(WAV)

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/main.wav",
                                    emotion_refs={"happy": ("/v/happy.wav", "开心地说")})
        with _patch_urlopen(fake_urlopen):
            result = client.synthesize("你好", "sad")
        self.assertTrue(result.ok)
        # 没有 sad 的情感参考 -> 必须说明情感没生效，不能默认"带了情绪"
        self.assertFalse(result.emotion_applied)
        self.assertIn("sad", result.emotion_note)


class FreeProviderHonestyTest(unittest.TestCase):
    """免费端点没有情感参数——必须如实标注，不能假装有情绪。"""

    def test_baidu_declares_no_emotion(self):
        self.assertFalse(BaiduTTSClient.supports_emotion)

    def test_baidu_marks_emotion_not_applied(self):
        def fake_urlopen(request, timeout=None):
            return _Resp(b"\xff\xf3\x88\xc4" + b"0" * 100)

        with _patch_urlopen(fake_urlopen):
            result = BaiduTTSClient().synthesize("你好", "happy")
        self.assertTrue(result.ok)
        self.assertFalse(result.emotion_applied)
        self.assertIn("不支持情感参数", result.emotion_note)

    def test_pipeline_description_admits_limitation(self):
        config = AppConfig()
        config.media.tts_enabled = True
        config.media.tts_provider = "baidu"
        config.media.tts_emotion_refs = "happy=/v/h.wav"
        text = describe_voice_pipeline(config.media)
        self.assertIn("不支持情感参数", text)


class ConversionPreservesEmotionTest(unittest.TestCase):
    def test_emotion_passed_to_inner_not_converter(self):
        captured = {}

        class _Inner:
            available = True
            supports_emotion = True

            def synthesize(self, text, emotion=""):
                captured["emotion"] = emotion
                return SpeechResult(True, data=WAV, provider="inner",
                                    emotion=emotion, emotion_applied=True)

        def fake_urlopen(request, timeout=None):
            return _Resp(WAV)

        wrapper = VoiceConvertedTTS(_Inner(), "http://127.0.0.1:7890")
        with _patch_urlopen(fake_urlopen):
            result = wrapper.synthesize("你好", "happy")
        # 语气在合成阶段就定了；转换只改音色，要透传而不是丢掉
        self.assertEqual(captured["emotion"], "happy")
        self.assertTrue(result.emotion_applied)

    def test_backward_compatible_with_old_inner(self):
        class _OldInner:
            available = True

            def synthesize(self, text):
                return SpeechResult(True, data=WAV, provider="old")

        def fake_urlopen(request, timeout=None):
            return _Resp(WAV)

        wrapper = VoiceConvertedTTS(_OldInner(), "http://127.0.0.1:7890")
        with _patch_urlopen(fake_urlopen):
            result = wrapper.synthesize("你好", "happy")
        self.assertTrue(result.ok)


class EngineEmotionTest(unittest.TestCase):
    def setUp(self):
        import shutil
        from aliceqq.media import MediaLibrary
        from tests.test_v4_media import _media_dir
        self._dir = _media_dir()
        self._shutil = shutil
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self._dir)
        self.engine.config.media.sticker_probability = 0.0
        self.engine.config.features.voice = True
        self.engine.config.media.tts_emotion_enabled = True

        class _TTS:
            available = True
            supports_emotion = True

            def __init__(self):
                self.emotions = []

            def synthesize(self, text, emotion=""):
                self.emotions.append(emotion)
                return SpeechResult(True, data=WAV, provider="fake",
                                    emotion=emotion, emotion_applied=True)

        self.tts = _TTS()
        self.engine.tts_client = self.tts

    def tearDown(self):
        self._shutil.rmtree(self._dir, ignore_errors=True)

    def test_explicit_tone_reaches_tts(self):
        _, outcome = private("用开心的语气说句话", engine=self.engine)
        self.assertEqual(self.tts.emotions[-1], emo.HAPPY)
        self.assertIn("开心", outcome.debug.get("emotion", ""))
        self.assertTrue(outcome.debug.get("emotion_applied"))

    def test_emotion_inferred_from_reply_when_not_specified(self):
        _, outcome = private("说句话吧", engine=self.engine)
        # 具体是什么情感取决于 mock 回复，但一定要推断出一个并记录
        self.assertIn("emotion", outcome.debug)

    def test_emotion_not_applied_is_recorded(self):
        class _NoEmotion:
            available = True
            supports_emotion = False

            def synthesize(self, text, emotion=""):
                return SpeechResult(True, data=WAV, provider="fake",
                                    emotion=emotion, emotion_applied=False,
                                    emotion_note="fake 端点不支持情感参数")

        self.engine.tts_client = _NoEmotion()
        _, outcome = private("用开心的语气说句话", engine=self.engine)
        # 情感没生效必须留在 debug 里，便于排查"为什么语音没情绪"
        self.assertFalse(outcome.debug.get("emotion_applied"))
        self.assertIn("不支持情感参数", outcome.debug.get("emotion_note", ""))

    def test_emotion_can_be_disabled(self):
        self.engine.config.media.tts_emotion_enabled = False
        _, outcome = private("用开心的语气说句话", engine=self.engine)
        self.assertNotIn("emotion", outcome.debug)


# ---- 辅助 ----
class _Resp:
    def __init__(self, payload):
        import io
        self._buf = io.BytesIO(payload)
        self.status = 200
        self.headers = {}

    def read(self, *args):
        return self._buf.read(*args)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(func):
    from unittest import mock
    return mock.patch("aliceqq.generation.net.open", func)


if __name__ == "__main__":
    unittest.main(verbosity=2)
