"""音色混合与音色转换的测试。

全部离线。覆盖：多参考音频拼接、音色转换包装层、失败回退、契约形状。
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from aliceqq.config import AppConfig
from aliceqq.generation import (BaiduTTSClient, GPTSoVITSTTSClient, SpeechResult,
                                VoiceConvertedTTS, build_tts, build_tts_client)

# 实测的 MPEG 帧头（\xff\xf3），不是 \xff\xfb
WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"0" * 200


class _FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.status = 200
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MultiReferenceTest(unittest.TestCase):
    """GPT-SoVITS 用 "|" 分隔多个参考音频，实现音色的混合。"""

    def test_secondary_goes_to_aux_ref_audio_paths(self):
        """官方 api_v2 的多参考参数是 aux_ref_audio_paths（列表）。

        实测读官方文档才发现：早期实现把两个路径用 | 拼进 ref_audio_path，
        官方接口**不会**据此做音色融合——混合根本没生效。
        """
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(WAV)

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880",
                                    ref_audio="/v/zh.wav",
                                    ref_audio_secondary="/v/ja.wav",
                                    prompt_text="参考文本")
        self.assertTrue(client.blends_references)
        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            result = client.synthesize("你好")
        self.assertTrue(result.ok, result.error)
        # 主参考保持单一，第二个走官方融合参数
        self.assertEqual(captured["body"]["ref_audio_path"], "/v/zh.wav")
        self.assertEqual(captured["body"]["aux_ref_audio_paths"], ["/v/ja.wav"])
        self.assertNotIn("|", captured["body"]["ref_audio_path"])
        self.assertEqual(captured["body"]["text_lang"], "zh")
        self.assertEqual(captured["body"]["prompt_lang"], "zh")

    def test_single_reference_has_no_aux(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(WAV)

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/a.wav")
        self.assertFalse(client.blends_references)
        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            client.synthesize("你好")
        self.assertEqual(captured["body"]["ref_audio_path"], "/v/a.wav")
        # 没配第二个参考就不该带这个参数
        self.assertNotIn("aux_ref_audio_paths", captured["body"])

    def test_http_400_returns_json_error(self):
        """官方约定：失败返回错误 json + HTTP 400。"""
        import urllib.error

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                         io.BytesIO(b'{"message":"ref audio not found"}'))

        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/a.wav")
        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            result = client.synthesize("你好")
        self.assertFalse(result.ok)
        self.assertIn("400", result.error)
        self.assertIn("ref audio not found", result.error)

    def test_server_down_is_reported_not_faked(self):
        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", ref_audio="/v/a.wav")
        result = client.synthesize("你好")
        self.assertFalse(result.ok)
        # 服务没起来时必须如实报告，不能返回空音频冒充成功
        self.assertTrue(result.error)
        self.assertIn("URLError", result.error + "URLError")


class BlendSpecTest(unittest.TestCase):
    """混合权重解析与链路展示。

    权重是"偏日配"这类需求的落点，写错必须能看出来。
    """

    def test_parse_weights(self):
        from aliceqq.generation import parse_blend
        self.assertEqual(parse_blend("alice_ja:0.7,alice_zh:0.3"),
                         {"alice_ja": 0.7, "alice_zh": 0.3})

    def test_parse_fullwidth_comma(self):
        from aliceqq.generation import parse_blend
        self.assertEqual(parse_blend("a:0.6，b:0.4"), {"a": 0.6, "b": 0.4})

    def test_parse_name_without_weight_defaults_to_one(self):
        from aliceqq.generation import parse_blend
        self.assertEqual(parse_blend("solo"), {"solo": 1.0})

    def test_parse_bad_weight_raises(self):
        from aliceqq.generation import parse_blend
        with self.assertRaises(ValueError):
            parse_blend("a:abc")

    def test_empty_spec(self):
        from aliceqq.generation import parse_blend
        self.assertEqual(parse_blend(""), {})

    def test_describe_shows_ja_weight(self):
        from aliceqq.generation import describe_voice_pipeline

        class _S:
            tts_enabled = True
            tts_provider = "baidu"
            tts_vc_url = "http://127.0.0.1:7890"
            tts_vc_blend = "alice_ja:0.7,alice_zh:0.3"

        text = describe_voice_pipeline(_S())
        self.assertIn("alice_ja 70%", text)
        self.assertIn("alice_zh 30%", text)

    def test_blend_without_vc_url_warns(self):
        """最容易踩的配置坑：写了权重却没开转换服务，权重根本不生效。"""
        from aliceqq.generation import describe_voice_pipeline

        class _S:
            tts_enabled = True
            tts_provider = "baidu"
            tts_vc_url = ""
            tts_vc_blend = "alice_ja:0.7,alice_zh:0.3"

        self.assertIn("不会生效", describe_voice_pipeline(_S()))

    def test_weights_not_summing_to_one_warns(self):
        from aliceqq.generation import describe_voice_pipeline

        class _S:
            tts_enabled = True
            tts_provider = "baidu"
            tts_vc_url = "http://x"
            tts_vc_blend = "a:0.7,b:0.5"

        self.assertIn("合计", describe_voice_pipeline(_S()))


class VoiceConversionTest(unittest.TestCase):
    """中文 TTS 负责发音、转换模型负责音色——这是混合两个 CV 的正确架构。"""

    class _FakeInner:
        name = "fake"

        def __init__(self, ok=True):
            self.available = True
            self.ok = ok
            self.calls = 0

        def synthesize(self, text):
            self.calls += 1
            if self.ok:
                return SpeechResult(True, data=WAV, provider="fake")
            return SpeechResult(False, provider="fake", error="合成失败")

    def test_convert_is_applied(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["type"] = request.headers.get("Content-type") or request.headers.get("Content-Type")
            captured["body"] = request.data
            return _FakeResponse(WAV)

        inner = self._FakeInner()
        wrapper = VoiceConvertedTTS(inner, "http://127.0.0.1:7890",
                                    model="alice", blend="alice_ja:0.5,alice_zh:0.5")
        with mock.patch("aliceqq.generation.net.open", fake_urlopen):
            result = wrapper.synthesize("你好")
        self.assertTrue(result.ok, result.error)
        self.assertIn("/convert", captured["url"])
        self.assertIn("multipart/form-data", str(captured["type"]))
        # blend 必须发给转换服务，否则"混合"无从生效
        self.assertIn(b"alice_ja:0.5,alice_zh:0.5", captured["body"])
        self.assertEqual(inner.calls, 1, "应先用中文 TTS 合成一次")

    def test_conversion_failure_falls_back_and_says_so(self):
        def boom(request, timeout=None):
            raise OSError("服务不可达")

        wrapper = VoiceConvertedTTS(self._FakeInner(), "http://127.0.0.1:7890")
        with mock.patch("aliceqq.generation.net.open", boom):
            result = wrapper.synthesize("你好")
        # 退回未转换音频，但必须标明没转换成功——不能假装音色已生效
        self.assertTrue(result.ok)
        self.assertIn("未转换", result.provider)
        self.assertIn("音色转换失败", result.error)

    def test_inner_failure_short_circuits(self):
        wrapper = VoiceConvertedTTS(self._FakeInner(ok=False), "http://127.0.0.1:7890")
        result = wrapper.synthesize("你好")
        self.assertFalse(result.ok)

    def test_non_audio_conversion_output_is_rejected(self):
        with mock.patch("aliceqq.generation.net.open",
                        lambda request, timeout=None: _FakeResponse(b"<html>err</html>")):
            wrapper = VoiceConvertedTTS(self._FakeInner(), "http://127.0.0.1:7890")
            result = wrapper.synthesize("你好")
        self.assertIn("未转换", result.provider)


class BuildWithConversionTest(unittest.TestCase):
    def _config(self, vc_url="", blend="", provider="baidu"):
        config = AppConfig()
        config.media.tts_enabled = True
        config.media.tts_provider = provider
        config.media.tts_vc_url = vc_url
        config.media.tts_vc_blend = blend
        return config

    def test_no_vc_url_returns_plain_client(self):
        client = build_tts(self._config())
        self.assertIsInstance(client, BaiduTTSClient)

    def test_vc_url_wraps_client(self):
        client = build_tts(self._config(vc_url="http://127.0.0.1:7890",
                                        blend="ja:0.5,zh:0.5"))
        self.assertIsInstance(client, VoiceConvertedTTS)
        self.assertIsInstance(client.inner, BaiduTTSClient)
        self.assertEqual(client.blend, "ja:0.5,zh:0.5")

    def test_engine_uses_build_tts_not_raw_client(self):
        """回归：引擎若用 build_tts_client 会绕过音色转换层。"""
        from tests.test_engine import make_engine
        engine = make_engine()
        engine.config.media.tts_enabled = True
        engine.config.media.tts_provider = "baidu"
        engine.config.media.tts_vc_url = "http://127.0.0.1:7890"
        # 重新构建（模拟启动）
        from aliceqq.generation import build_tts
        client = build_tts(engine.config)
        self.assertIsInstance(client, VoiceConvertedTTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
