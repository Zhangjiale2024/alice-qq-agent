"""TTS 客户端构建：确保每种 provider 都真的能构建出来。

这个文件是因为一个真 bug 才加的：build_tts_client 给 GPTSoVITSTTSClient 传了
emotion_alpha 参数，而客户端 __init__ 根本不接受它。后果是 TypeError ->
语音功能静默退回 baidu，而没人发现，因为没有测试覆盖"构建"这一步。
所以这里把每种 provider 都真的构造一遍。
"""
from __future__ import annotations

import unittest

from aliceqq.config import AppConfig
from aliceqq.generation import (BaiduTTSClient, GoogleTTSClient,
                                GPTSoVITSTTSClient, OpenAITTSClient,
                                build_tts_client)


def _config(provider: str, **media_overrides):
    cfg = AppConfig.load()
    cfg.media.tts_enabled = True
    cfg.media.tts_provider = provider
    for key, value in media_overrides.items():
        setattr(cfg.media, key, value)
    return cfg


class BuildTTSClientTest(unittest.TestCase):
    def test_gptsovits_constructs(self):
        """回归：传 emotion_alpha 会 TypeError，导致静默退回 baidu。"""
        cfg = _config("gptsovits",
                      tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_ref_audio="D:/refs/ja.wav",
                      tts_clone_ref_audio_secondary="D:/refs/zh.wav",
                      tts_clone_prompt_text="")
        client = build_tts_client(cfg)
        self.assertIsInstance(client, GPTSoVITSTTSClient)
        self.assertEqual(client.ref_audio, "D:/refs/ja.wav")
        self.assertEqual(client.ref_audio_secondary, "D:/refs/zh.wav")

    def test_gptsovits_accepts_emotion_refs(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_emotion_refs="happy=D:/refs/happy.wav|今天真开心")
        client = build_tts_client(cfg)
        self.assertIsInstance(client, GPTSoVITSTTSClient)
        self.assertTrue(client.supports_emotion)

    def test_baidu_constructs(self):
        self.assertIsInstance(build_tts_client(_config("baidu")), BaiduTTSClient)

    def test_google_constructs(self):
        self.assertIsInstance(build_tts_client(_config("google")), GoogleTTSClient)

    def test_openai_requires_key(self):
        cfg = _config("openai_compatible", tts_api_key="", tts_base_url="")
        self.assertIsNone(build_tts_client(cfg))

    def test_disabled_returns_none(self):
        cfg = AppConfig.load()
        cfg.media.tts_enabled = False
        self.assertIsNone(build_tts_client(cfg))

    def test_unknown_provider_returns_none(self):
        self.assertIsNone(build_tts_client(_config("不存在的provider")))


class SpeedFactorTest(unittest.TestCase):
    """客户端支持 speed_factor，但 build_tts_client 曾漏传，导致配置语速无效。"""

    def test_speed_wired_from_config(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_speed=1.15)
        client = build_tts_client(cfg)
        self.assertAlmostEqual(client.speed_factor, 1.15)

    def test_speed_value_is_passed_through(self):
        # 显式给值，不依赖 .env —— .env 会随用户调音变化，
        # 测试断言"默认是 1.0"就会因为用户改成 0.92 而莫名失败（踩过）。
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_speed=1.0)
        self.assertAlmostEqual(build_tts_client(cfg).speed_factor, 1.0)

    def test_speed_one_point_zero_eight(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_speed=1.08)
        self.assertAlmostEqual(build_tts_client(cfg).speed_factor, 1.08)

    def test_zero_speed_falls_back(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_speed=0)
        self.assertAlmostEqual(build_tts_client(cfg).speed_factor, 1.0)


class SeedTest(unittest.TestCase):
    """种子必须显式传给客户端。

    客户端默认是 -1（随机），而配置里长期没有这个键，等于每次都在随机采样：
    实测时长在 4.54~5.72 秒间摆动（±13%）、音调 ±15 Hz，
    听感就是"语速忽快忽慢、前后像两个人"。固定种子后字节级可复现。
    """

    def test_seed_is_wired(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_seed=42)
        self.assertEqual(build_tts_client(cfg).seed, 42)

    def test_seed_accepts_random(self):
        cfg = _config("gptsovits", tts_clone_url="http://127.0.0.1:9880",
                      tts_clone_seed=-1)
        self.assertEqual(build_tts_client(cfg).seed, -1)

    def test_media_config_default_seed_is_fixed(self):
        """数据类默认值要是固定数，不能是 -1（随机）。"""
        from aliceqq.config import MediaConfig
        self.assertNotEqual(MediaConfig().tts_clone_seed, -1)


class MediaTypeFallbackTest(unittest.TestCase):
    """api_v2 只支持 wav/raw/ogg/aac，不支持 mp3——不能把 mp3 透传过去。"""

    def test_mp3_falls_back_to_wav(self):
        client = GPTSoVITSTTSClient("http://127.0.0.1:9880", media_type="mp3")
        self.assertEqual(client.media_type, "wav")

    def test_supported_types_pass_through(self):
        for fmt in ("wav", "ogg", "aac", "raw", "WAV"):
            client = GPTSoVITSTTSClient("http://127.0.0.1:9880", media_type=fmt)
            self.assertEqual(client.media_type, fmt.lower())

    def test_empty_defaults_to_wav(self):
        self.assertEqual(GPTSoVITSTTSClient("http://x", media_type="").media_type, "wav")


class AuxRefSplittingTest(unittest.TestCase):
    """辅助参考支持逗号/竖线分隔的重复写法（用于粗略调整音色倾向）。"""

    def test_secondary_split_on_comma(self):
        import re
        spec = "D:/a.wav,D:/a.wav"
        self.assertEqual([p for p in re.split(r"[,|]", spec) if p.strip()],
                         ["D:/a.wav", "D:/a.wav"])

    def test_secondary_split_on_pipe(self):
        import re
        spec = "D:/a.wav|D:/b.wav"
        self.assertEqual([p for p in re.split(r"[,|]", spec) if p.strip()],
                         ["D:/a.wav", "D:/b.wav"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
