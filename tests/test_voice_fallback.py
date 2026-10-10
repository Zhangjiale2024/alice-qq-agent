"""语音降级提示的测试。

背景：FallbackTTS 在克隆服务不通时会回退到免费 TTS，此时 ok=True、error 为空，
用户完全看不出声音换了人。按项目的诚实原则必须如实说明。
"""
from __future__ import annotations

import os
import shutil
import time
import unittest
import uuid

from aliceqq.generation import SpeechResult
from aliceqq.media import MediaLibrary
from tests.test_engine import make_engine, private

TRIGGER = "说句话"          # 命中 _VOICE_PATTERNS，才会走语音路径


def _media_dir() -> str:
    """在工作区内建临时素材目录（系统临时目录在沙箱下不可写）。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "test-media")
    path = os.path.join(base, "case-" + uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    return path


class _StubTTS:
    """可控的 TTS 桩：provider 里带不带「(备用)」决定是否触发降级提示。"""

    name = "stub"
    available = True
    supports_emotion = False

    def __init__(self, provider="stub", ok=True):
        self.provider = provider
        self.ok = ok
        self.calls = 0

    def synthesize(self, text, emotion=""):
        self.calls += 1
        return SpeechResult(self.ok, data=b"RIFF____WAVEfmt ", mime="audio/wav",
                            provider=self.provider, error="" if self.ok else "服务不可用")


class FallbackNoteTest(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        # 注意：_attach_media 开头是 "if self.media_library is None: return"，
        # 不建素材库的话整条媒体路径会被提前返回，语音永远不会被尝试。
        self.root = _media_dir()
        self.engine.media_library = MediaLibrary(self.root)
        self.engine.config.bot.owner_ids = ["10001"]
        # 两个开关都要开，别搞混：
        #   config.features.voice —— 功能总开关，_media_allowed 检查的是这个
        #   config.media.voice    —— 媒体子配置里的对应项
        self.engine.config.features.voice = True
        self.engine.config.media.voice = True
        self.engine.config.media.tts_max_chars = 120

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_fallback_provider_adds_note(self):
        self.engine.tts_client = _StubTTS(provider="baidu(备用)")
        _, outcome = private(TRIGGER, user="10001", engine=self.engine)
        self.assertIn("备用声音", outcome.reply or "")
        self.assertEqual(outcome.debug.get("voice_fallback"), "baidu(备用)")

    def test_normal_provider_has_no_note(self):
        self.engine.tts_client = _StubTTS(provider="gptsovits")
        _, outcome = private(TRIGGER, user="10001", engine=self.engine)
        self.assertNotIn("备用声音", outcome.reply or "")
        self.assertNotIn("voice_fallback", outcome.debug)

    def test_note_is_rate_limited(self):
        """同一用户短时间内只提醒一次，避免每条消息都刷同一句。"""
        engine = self.engine
        engine.tts_client = _StubTTS(provider="baidu(备用)")
        _, first = private(TRIGGER, user="10001", engine=engine)
        self.assertIn("备用声音", first.reply or "")
        _, second = private(TRIGGER, user="10001", engine=engine)
        self.assertNotIn("备用声音", second.reply or "")
        # debug 每次都记，只是不再打扰用户
        self.assertEqual(second.debug.get("voice_fallback"), "baidu(备用)")

    def test_rate_limit_window_expires(self):
        engine = self.engine
        engine.tts_client = _StubTTS(provider="baidu(备用)")
        private(TRIGGER, user="10001", engine=engine)
        self.assertFalse(engine._should_note_fallback("10001", time.time()))
        engine.storage.kv_set("voice_fallback_noted:10001", time.time() - 99999)
        self.assertTrue(engine._should_note_fallback("10001", time.time()))

    def test_different_users_not_shared(self):
        engine = self.engine
        engine.tts_client = _StubTTS(provider="baidu(备用)")
        private(TRIGGER, user="10001", engine=engine)
        _, other = private(TRIGGER, user="10002", engine=engine)
        self.assertIn("备用声音", other.reply or "")

    def test_tts_actually_called(self):
        """确保测试真的走到了语音路径，否则上面的断言都是空转。"""
        stub = _StubTTS(provider="gptsovits")
        self.engine.tts_client = stub
        private(TRIGGER, user="10001", engine=self.engine)
        self.assertGreater(stub.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
