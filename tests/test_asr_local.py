"""本地 ASR 后端（faster-whisper）的测试。

全部离线：不加载真模型（加载一次要 20 秒、占 500MB），只验证
「选择逻辑、依赖缺失时的降级、错误如实返回」这些容易出错的部分。
"""
from __future__ import annotations

import unittest
from unittest import mock

from aliceqq import asr as asr_mod
from aliceqq.asr import (ASRClient, LocalWhisperASR, Transcription, build_client,
                         quality_warning)
from tests.test_engine import make_engine


class _FakeInfo:
    duration = 8.0


class _FakeSegment:
    def __init__(self, text, avg_logprob=None, no_speech_prob=None):
        self.text = text
        # 真实 faster-whisper 的 segment 带这两个字段，
        # LocalWhisperASR 会收集它们做质量自检（见 asr.quality_warning）。
        # 这里默认给 None，测"模型不给指标时也不能崩"。
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class _FakeModel:
    """假模型：只验证调用与结果拼装，不做真实推理。"""

    def __init__(self, texts=("你好",), raise_exc=None, avg_logprob=None,
                 no_speech_prob=None):
        self.texts = texts
        self.raise_exc = raise_exc
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.calls = []

    def transcribe(self, audio, language=None, beam_size=5, word_timestamps=False):
        self.calls.append({"language": language, "beam_size": beam_size,
                           "word_timestamps": word_timestamps})
        if self.raise_exc:
            raise self.raise_exc
        return ([_FakeSegment(t, self.avg_logprob, self.no_speech_prob)
                 for t in self.texts], _FakeInfo())


class BuildClientTest(unittest.TestCase):
    def _config(self, **kw):
        cfg = make_engine().config
        for key, value in kw.items():
            setattr(cfg.asr, key, value)
        return cfg

    def test_disabled_returns_none(self):
        cfg = self._config(enabled=False, provider="local")
        self.assertIsNone(build_client(cfg))

    def test_local_provider_returns_local_client(self):
        cfg = self._config(enabled=True, provider="local", local_model="small",
                           local_device="cpu", local_cache="/tmp/x")
        client = build_client(cfg)
        self.assertIsInstance(client, LocalWhisperASR)
        self.assertEqual(client.model_size, "small")

    def test_local_needs_no_api_key(self):
        """本地后端没 Key 也要能构建出来。

        踩过的坑：早期"没 key 就把 ASR 关掉"的校验会把本地后端一起关掉，
        结果"开了 ASR 却永远不可用"（TTS 那边犯过同样的错）。
        """
        cfg = self._config(enabled=True, provider="local", api_key="")
        self.assertIsInstance(build_client(cfg), LocalWhisperASR)

    def test_cloud_provider_returns_cloud_client(self):
        cfg = self._config(enabled=True, provider="openai_compatible",
                           api_key="sk-x", base_url="https://api.openai.com/v1")
        self.assertIsInstance(build_client(cfg), ASRClient)

    def test_cloud_without_key_returns_none(self):
        cfg = self._config(enabled=True, provider="openai_compatible", api_key="")
        self.assertIsNone(build_client(cfg))

    def test_local_provider_aliases(self):
        for alias in ("local", "faster_whisper", "faster-whisper", "whisper_local",
                      "LOCAL"):
            cfg = self._config(enabled=True, provider=alias)
            self.assertIsInstance(build_client(cfg), LocalWhisperASR, alias)


class LocalWhisperTest(unittest.TestCase):
    def setUp(self):
        """把 faster_whisper 打进 sys.modules。

        测试用的解释器不一定装了它——本套件的其余部分是纯标准库的，
        不能因为一个可选依赖没装就整套失败。真正测"没装会怎样"的用例
        会在自己内部覆盖这个打桩。
        """
        self._fake_module = mock.MagicMock()
        patcher = mock.patch.dict("sys.modules", {"faster_whisper": self._fake_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_dependency_degrades_gracefully(self):
        """没装 faster-whisper 时 available 为 False，且不抛异常。"""
        client = LocalWhisperASR()
        with mock.patch.dict("sys.modules", {"faster_whisper": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("no module")):
                self.assertFalse(client.available)
        result = client.transcribe(b"RIFF....")
        self.assertFalse(result.ok)
        self.assertEqual(result.provider, "local_whisper")
        self.assertTrue(result.error)

    def test_empty_audio_rejected(self):
        client = LocalWhisperASR()
        result = client.transcribe(b"")
        self.assertFalse(result.ok)
        self.assertIn("空", result.error)

    def test_successful_transcription_joins_segments(self):
        client = LocalWhisperASR(language="ja")
        client._model = _FakeModel(texts=("你好，", "世界。"))
        result = client.transcribe(b"RIFF....")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "你好，世界。")
        self.assertEqual(result.provider, "local_whisper")

    def test_language_is_passed_through(self):
        client = LocalWhisperASR(language="ja")
        fake = _FakeModel()
        client._model = fake
        client.transcribe(b"RIFF....")
        self.assertEqual(fake.calls[0]["language"], "ja")

    def test_empty_language_becomes_none(self):
        """language 为空要传 None（让 whisper 自动判断），不能传空字符串。"""
        client = LocalWhisperASR(language="")
        fake = _FakeModel()
        client._model = fake
        client.transcribe(b"RIFF....")
        self.assertIsNone(fake.calls[0]["language"])

    def test_model_is_built_only_once(self):
        """模型只能构造一次，否则每条语音都要等一次加载（实测约 20 秒）。

        注意断言的是**底层模型被构造的次数**，而不是"_load 没被调用"——
        transcribe 每次都会调 _load，由它内部判断缓存命中，
        _load 本身被调用是正常且必要的（首次失败后还要能重试）。
        """
        build_count = {"n": 0}

        def factory(size, **kwargs):
            build_count["n"] += 1
            return _FakeModel()

        fake_module = mock.MagicMock(WhisperModel=factory)
        with mock.patch.dict("sys.modules", {"faster_whisper": fake_module}):
            client = LocalWhisperASR()
            client.transcribe(b"RIFF....")
            client.transcribe(b"RIFF....")
            client.transcribe(b"RIFF....")
        self.assertEqual(build_count["n"], 1, "模型被重复构造了")

    def test_load_retried_after_failure(self):
        """首次加载失败后要能重试（_model 保持 None，不会被错误地缓存成负结果）。"""
        attempts = {"n": 0}

        def flaky(size, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("临时故障")
            return _FakeModel(texts=("好了",))

        fake_module = mock.MagicMock(WhisperModel=flaky)
        with mock.patch.dict("sys.modules", {"faster_whisper": fake_module}):
            client = LocalWhisperASR()
            first = client.transcribe(b"RIFF....")
            second = client.transcribe(b"RIFF....")
        self.assertFalse(first.ok)
        self.assertTrue(second.ok, "首次失败后应能重试成功")
        self.assertEqual(second.text, "好了")

    def test_model_error_returned_honestly(self):
        """识别失败要如实返回错误，不能编造文字。"""
        client = LocalWhisperASR()
        client._model = _FakeModel(raise_exc=RuntimeError("显存不足"))
        result = client.transcribe(b"RIFF....")
        self.assertFalse(result.ok)
        self.assertIn("显存不足", result.error)
        self.assertEqual(result.text, "")

    def test_empty_result_is_failure(self):
        client = LocalWhisperASR()
        client._model = _FakeModel(texts=("   ",))
        result = client.transcribe(b"RIFF....")
        self.assertFalse(result.ok)
        self.assertIn("空", result.error)

    def test_load_failure_is_reported(self):
        client = LocalWhisperASR()
        with mock.patch.object(client, "_load", side_effect=OSError("模型文件损坏")):
            result = client.transcribe(b"RIFF....")
        self.assertFalse(result.ok)
        self.assertIn("模型文件损坏", result.error)

    def test_cpu_uses_int8_gpu_uses_float16(self):
        """compute_type 要按设备选，选错会白白慢几倍或直接报错。"""
        captured = {}

        class _Recorder:
            def __init__(self, size, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict("sys.modules",
                             {"faster_whisper": mock.MagicMock(WhisperModel=_Recorder)}):
            LocalWhisperASR(device="cpu", cache_dir="/tmp/c")._load()
            self.assertEqual(captured.get("compute_type"), "int8")
            self.assertEqual(captured.get("download_root"), "/tmp/c")

            captured.clear()
            LocalWhisperASR(device="cuda")._load()
            self.assertEqual(captured.get("compute_type"), "float16")

    def test_cuda_load_prepares_torch_dlls(self):
        """cuda 设备要先准备 CUDA 运行库路径。

        踩过的坑：ctranslate2 报 "Library cublas64_12.dll is not found"，
        而那个 DLL 其实**已经在 torch\lib 里**（torch 自带的），
        只是 Windows 上 Python 3.8+ 不再从 PATH 找扩展模块依赖，
        必须显式 add_dll_directory。
        """
        class _Recorder:
            def __init__(self, size, **kwargs):
                pass

        with mock.patch.dict("sys.modules",
                             {"faster_whisper": mock.MagicMock(WhisperModel=_Recorder)}):
            with mock.patch.object(asr_mod, "_add_torch_cuda_dlls") as prep:
                LocalWhisperASR(device="cuda")._load()
                prep.assert_called_once()

    def test_cpu_load_does_not_touch_dlls(self):
        class _Recorder:
            def __init__(self, size, **kwargs):
                pass

        with mock.patch.dict("sys.modules",
                             {"faster_whisper": mock.MagicMock(WhisperModel=_Recorder)}):
            with mock.patch.object(asr_mod, "_add_torch_cuda_dlls") as prep:
                LocalWhisperASR(device="cpu")._load()
                prep.assert_not_called()

    def test_dll_helper_is_safe_without_torch(self):
        """torch 没装时要安全返回，不能抛异常。"""
        with mock.patch.dict("sys.modules", {"torch": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("no torch")):
                self.assertEqual(asr_mod._add_torch_cuda_dlls(), "")


class QualityWarningTest(unittest.TestCase):
    """识别质量自检（asr.quality_warning）。

    为什么要有这些用例：ASR 在噪声/背景音乐上会输出**语法通顺但完全无关**的
    文本（本项目实测过），这比"识别失败"更危险，因为看起来像真的。
    这里逐条验证每个告警条件都能触发，且正常结果不会被误报。
    """

    def test_clean_result_has_no_warning(self):
        text = "今天讲了入坑需要准备什么，先看主线剧情再考虑抽卡"
        self.assertEqual(quality_warning(text, seconds=10.0,
                                         avg_logprob=-0.3, no_speech_prob=0.05), "")

    def test_empty_text_has_no_warning(self):
        self.assertEqual(quality_warning("", seconds=10.0), "")

    def test_too_few_characters_for_the_audio(self):
        """40 秒音频只识别出几个字，基本可以断定没识别到人声。"""
        warning = quality_warning("啊啊啊", seconds=40.0)
        self.assertIn("不可靠", warning)
        self.assertIn("3 个字", warning)

    def test_short_audio_is_not_flagged_for_length(self):
        """短音频里字数少是正常的，不能误报。"""
        self.assertEqual(quality_warning("啊啊啊", seconds=5.0), "")

    def test_repetition_detected(self):
        warning = quality_warning("哈哈哈哈哈哈哈哈" * 8, seconds=60.0)
        self.assertIn("重复", warning)

    def test_moderate_repetition_is_allowed(self):
        """正常文本里某个字出现偏多（比如"的"）不该被判成复读。"""
        text = ("这个游戏的内容是这样的，玩家需要先完成这样的任务，"
                "然后才能得到这样的奖励，这样的设计是为了让玩家慢慢熟悉")
        self.assertEqual(quality_warning(text, seconds=30.0), "")

    def test_high_no_speech_probability(self):
        warning = quality_warning("嗯……", seconds=5.0, no_speech_prob=0.85)
        self.assertIn("没有人声", warning)

    def test_low_average_logprob(self):
        warning = quality_warning("一些听不清的内容", seconds=5.0, avg_logprob=-1.4)
        self.assertIn("置信度偏低", warning)

    def test_several_reasons_are_all_listed(self):
        warning = quality_warning("哈" * 60, seconds=120.0, avg_logprob=-1.5,
                                  no_speech_prob=0.9)
        self.assertIn("重复", warning)
        self.assertIn("置信度偏低", warning)
        self.assertIn("没有人声", warning)

    def test_missing_metrics_are_tolerated(self):
        """后端不给指标时（None）不能崩，也不能瞎报警。"""
        self.assertEqual(quality_warning("正常的一句话内容", seconds=5.0), "")
        self.assertEqual(
            quality_warning("正常的一句话内容", seconds=5.0, avg_logprob=None,
                            no_speech_prob=None), "")


class LocalWhisperMetricsTest(unittest.TestCase):
    """本地后端要把置信度指标收集出来，交给质量自检。"""

    def setUp(self):
        patcher = mock.patch.dict("sys.modules", {"faster_whisper": mock.MagicMock()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_metrics_are_averaged_and_maxed(self):
        client = LocalWhisperASR()
        client._model = _FakeModel(texts=("你好", "世界"), avg_logprob=-0.4,
                                   no_speech_prob=0.3)
        result = client.transcribe(b"RIFF....")
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.avg_logprob, -0.4, places=6)
        self.assertAlmostEqual(result.no_speech_prob, 0.3, places=6)
        self.assertEqual(result.segments, 2)

    def test_missing_metrics_stay_none(self):
        client = LocalWhisperASR()
        client._model = _FakeModel(texts=("你好",))
        result = client.transcribe(b"RIFF....")
        self.assertTrue(result.ok)
        self.assertIsNone(result.avg_logprob)
        self.assertIsNone(result.no_speech_prob)


class ConfigTest(unittest.TestCase):
    """分段配置要有默认值，并且能从环境变量读出来。"""

    def test_defaults(self):
        from aliceqq.config import AsrConfig
        cfg = AsrConfig()
        self.assertEqual(cfg.segment_seconds, 300)
        self.assertEqual(cfg.max_segments, 24)

    def test_env_values_are_read(self):
        import tempfile
        from pathlib import Path
        from aliceqq.config import AppConfig, load_env_file
        # 直接验证解析层：这两个键名要能对上
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ASR_SEGMENT_SECONDS=120\nASR_MAX_SEGMENTS=5\n",
                            encoding="utf-8")
            raw = load_env_file(path)
        self.assertEqual(raw["ASR_SEGMENT_SECONDS"], "120")
        self.assertEqual(raw["ASR_MAX_SEGMENTS"], "5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
