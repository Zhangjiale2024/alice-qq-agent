"""免费 TTS（百度 / Google）的测试。

全部离线（打桩 urlopen），但用例形状取自实测：
百度返回 MP3 帧头 \xff\xf3，Google 返回 \xff\xf3 —— 只判断 \xff\xfb 会漏判。
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from aliceqq.config import AppConfig
from aliceqq.generation import (BaiduTTSClient, GoogleTTSClient, OpenAITTSClient,
                                _looks_like_audio, build_tts_client)

# 实测帧头（百度/Google 都是 MPEG 帧同步 \xff\xf3）
MP3_F3 = b"\xff\xf3\x88\xc4" + b"0" * 200
MP3_ID3 = b"ID3\x03\x00\x00\x00" + b"0" * 200


class _FakeResponse(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.status = 200
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class AudioSniffTest(unittest.TestCase):
    def test_accepts_mpeg_sync_f3(self):
        # 实测踩过：只判 \xff\xfb 会漏掉 \xff\xf3 这个帧头
        self.assertTrue(_looks_like_audio(MP3_F3))

    def test_accepts_other_containers(self):
        self.assertTrue(_looks_like_audio(MP3_ID3))
        self.assertTrue(_looks_like_audio(b"RIFF\x00\x00\x00\x00WAVE"))
        self.assertTrue(_looks_like_audio(b"OggS\x00\x00"))

    def test_rejects_html_and_empty(self):
        self.assertFalse(_looks_like_audio(b""))
        self.assertFalse(_looks_like_audio(b"<html><body>error</body></html>"))
        self.assertFalse(_looks_like_audio(b"{\"error\":\"quota\"}"))


class BaiduTTSTest(unittest.TestCase):
    def test_available_without_key(self):
        self.assertTrue(BaiduTTSClient().available)

    def test_synthesize_returns_audio(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["referer"] = request.get_header("Referer")
            return _FakeResponse(MP3_F3, {"Content-Type": "audio/mpeg"})

        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            result = BaiduTTSClient(language="zh", speed=5).synthesize("你好")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.data, MP3_F3)
        self.assertEqual(result.provider, "baidu")
        self.assertIn("fanyi.baidu.com/gettts", captured["url"])
        self.assertIn("lan=zh", captured["url"])
        # 片段必须是 URL 编码的
        self.assertIn("%E4%BD%A0%E5%A5%BD", captured["url"])

    def test_empty_text_is_rejected(self):
        result = BaiduTTSClient().synthesize("   ")
        self.assertFalse(result.ok)
        self.assertIn("为空", result.error)

    def test_non_audio_response_is_failure(self):
        # 端点返回 HTML 错误页时不能当成成功
        with mock.patch("aliceqq.generation.net.open",
                        lambda request, timeout=None: _FakeResponse(
                            b"<html>blocked</html>")):
            result = BaiduTTSClient().synthesize("你好")
        self.assertFalse(result.ok)
        self.assertIn("不是音频", result.error)

    def test_speed_is_clamped(self):
        self.assertEqual(BaiduTTSClient(speed=99).speed, 9)
        self.assertEqual(BaiduTTSClient(speed=-3).speed, 1)
        # 0 视为"未设置"，回落到默认 5（而不是被夹成 1）
        self.assertEqual(BaiduTTSClient(speed=0).speed, 5)


class GoogleTTSTest(unittest.TestCase):
    def test_synthesize_returns_audio(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return _FakeResponse(MP3_F3)

        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            result = GoogleTTSClient(language="zh-CN").synthesize("你好")
        self.assertTrue(result.ok, result.error)
        self.assertIn("translate_tts", captured["url"])
        self.assertIn("tl=zh-CN", captured["url"])

    def test_long_text_is_truncated(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return _FakeResponse(MP3_F3)

        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            GoogleTTSClient().synthesize("字" * 500)
        # 该端点对长文本会截断，所以代码里先截到 200 字。
        # 注意不能按 URL 长度断言：中文一个字符 URL 编码后是 9 字节。
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(captured["url"]).query)
        self.assertEqual(len(query["q"][0]), 200, "应先截到 200 字")


class BuildTTSClientTest(unittest.TestCase):
    def _config(self, provider, enabled=True, key=""):
        config = AppConfig()
        config.media.tts_enabled = enabled
        config.media.tts_provider = provider
        config.media.tts_api_key = key
        return config

    def test_baidu_is_default_and_needs_no_key(self):
        config = AppConfig()
        config.media.tts_enabled = True
        client = build_tts_client(config)
        self.assertIsInstance(client, BaiduTTSClient)

    def test_google_alias(self):
        self.assertIsInstance(build_tts_client(self._config("gtts")), GoogleTTSClient)

    def test_openai_requires_key(self):
        self.assertIsNone(build_tts_client(self._config("openai_compatible")))
        self.assertIsInstance(build_tts_client(self._config("openai_compatible", key="sk-x")),
                              OpenAITTSClient)

    def test_disabled_returns_none(self):
        self.assertIsNone(build_tts_client(self._config("baidu", enabled=False)))

    def test_free_provider_survives_missing_key(self):
        """实测踩过：早期"没 key 就把 tts 关掉"会把免费 provider 也关掉。

        这里覆盖配置加载路径：provider=baidu 且没有 key 时必须仍然可用。
        """
        from aliceqq.config import AppConfig as Cfg
        config = Cfg()
        config.media.tts_provider = "baidu"
        config.media.tts_api_key = ""
        config.media.tts_enabled = True
        self.assertIsNotNone(build_tts_client(config))


class MediaIntentTest(unittest.TestCase):
    """媒体意图要提前告知模型，否则会出现"说语音没解锁、却发了语音"。

    实测踩过这个矛盾：文本先生成、媒体后附加，模型并不知道本次会发语音。
    """

    def setUp(self):
        import os
        import shutil
        from aliceqq.media import MediaLibrary
        from tests.test_engine import make_engine
        from tests.test_v4_media import _media_dir
        self._dir = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self._dir)
        self._shutil = shutil

    def tearDown(self):
        self._shutil.rmtree(self._dir, ignore_errors=True)

    def _intent(self, text):
        from aliceqq.engine import IncomingMessage
        return self.engine._media_intent(
            IncomingMessage(user_id="1", text=text, is_private=True, user_name="x"))

    def test_voice_request_with_tts_says_voice_will_be_sent(self):
        class _T:
            available = True
        self.engine.tts_client = _T()
        self.engine.config.features.voice = True
        intent = self._intent("用语音说句话")
        self.assertIn("语音形式发送", intent)
        self.assertIn("不要", intent)

    def test_voice_request_without_tts_says_unavailable(self):
        self.engine.tts_client = None
        self.engine.config.features.voice = True
        self.assertIn("不可用", self._intent("用语音说句话"))

    def test_plain_chat_does_not_promise_voice(self):
        intent = self._intent("今天天气不错")
        self.assertNotIn("语音形式发送", intent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
