# -*- coding: utf-8 -*-
"""GPU 不可用时自动退回 CPU 的测试（全部离线，用假模型）。

**这是真机验证换来的修复。**

真机上语音完全不可用，报的原因是环境问题：

    机器人解释器（Python311）没装 faster-whisper
    → available=False → '没有可用的语音识别'

装完之后**还是不行**，因为 `.env` 配的是 `ASR_LOCAL_DEVICE=cuda`，
而那个解释器没有 torch，ctranslate2 拿不到 CUDA 运行库：

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

同一个模型在 CPU 上实测 4.6 秒转写完全可用，所以让代码自动降级——
否则表现是"语音识别又不可用"，而真正的原因（缺一个 cublas DLL）
完全看不出来。

**踩到的第二个坑**：`WhisperModel(device="cuda")` 构造时**不会**
加载 cuBLAS，错误要等到第一次 transcribe 才爆。所以只把构造包在
try 里是不够的（第一版就是这么写的，降级从未触发）。现在加载时
用一小段静音跑一次探针，把问题在那一步就暴露出来。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aliceqq.asr import LocalWhisperASR


class _FakeCudaModel:
    """模拟"构造成功、一推理就报 cuBLAS 缺失"的真实行为。"""

    def __init__(self, size, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, *a, **k):
        def gen():
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield  # pragma: no cover
        return gen(), None


class _FakeCpuModel:
    def __init__(self, size, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, *a, **k):
        class _Seg:
            text = "爱丽丝今天想去打游戏"
            avg_logprob = -0.2
            no_speech_prob = 0.05

        return iter([_Seg()]), None


class _Factory:
    """第一次返回会失败的 CUDA 模型，之后返回可用的 CPU 模型。"""

    def __init__(self):
        self.calls = []

    def __call__(self, size, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("device") == "cuda":
            return _FakeCudaModel(size, **kwargs)
        return _FakeCpuModel(size, **kwargs)


def _with_fake(factory):
    fake_module = mock.MagicMock()
    fake_module.WhisperModel = factory
    return mock.patch.dict("sys.modules", {"faster_whisper": fake_module})


class GpuFallbackTest(unittest.TestCase):
    def test_does_not_fall_back_when_cpu_requested(self):
        factory = _Factory()
        client = LocalWhisperASR(device="cpu")
        with _with_fake(factory):
            client._load()
        self.assertEqual(client.device, "cpu")
        self.assertEqual(len(factory.calls), 1, "CPU 路径不该多构造一次")

    def test_falls_back_to_cpu_when_gpu_inference_fails(self):
        factory = _Factory()
        client = LocalWhisperASR(device="cuda")
        with _with_fake(factory):
            client.transcribe(b"RIFF....")
        self.assertEqual(client.device, "cpu", "GPU 推理失败后应当退回 CPU")
        devices = [c.get("device") for c in factory.calls]
        self.assertEqual(devices, ["cuda", "cpu"])

    def test_fallback_uses_int8_on_cpu(self):
        # CPU 上用 int8（比 float32 快很多）；降级后必须换成它
        factory = _Factory()
        client = LocalWhisperASR(device="cuda")
        with _with_fake(factory):
            client.transcribe(b"RIFF....")
        self.assertEqual(factory.calls[-1].get("compute_type"), "int8")

    def test_fallback_reason_is_recorded(self):
        # 必须留下原因，否则表现只是"变慢了"，看不出发生过降级
        factory = _Factory()
        client = LocalWhisperASR(device="cuda")
        with _with_fake(factory):
            client.transcribe(b"RIFF....")
        reason = getattr(client, "_gpu_fallback_reason", "")
        self.assertIn("cublas", reason)
        self.assertIn("CPU", reason)

    def test_transcription_still_succeeds_after_fallback(self):
        factory = _Factory()
        client = LocalWhisperASR(device="cuda")
        with _with_fake(factory):
            result = client.transcribe(b"RIFF....")
        self.assertTrue(result.ok)
        self.assertIn("爱丽丝", result.text)

    def test_cache_dir_is_preserved_on_fallback(self):
        factory = _Factory()
        client = LocalWhisperASR(device="cuda", cache_dir="/tmp/cache")
        with _with_fake(factory):
            client.transcribe(b"RIFF....")
        self.assertEqual(factory.calls[-1].get("download_root"), "/tmp/cache")

    def test_cpu_failure_is_not_swallowed(self):
        """CPU 也失败说明是模型/缓存问题，降级救不了，必须如实抛出去。"""

        class _Boom:
            def __init__(self, size, **kwargs):
                raise OSError("模型文件损坏")

        fake_module = mock.MagicMock()
        fake_module.WhisperModel = _Boom
        client = LocalWhisperASR(device="cpu")
        with mock.patch.dict("sys.modules", {"faster_whisper": fake_module}):
            with self.assertRaises(OSError):
                client._load()


class ProbeGuardTest(unittest.TestCase):
    """探针不能把假模型判成"GPU 不可用"。

    测试会用**只记录参数**的假模型替换 WhisperModel，那种对象没有
    transcribe。第一版无条件调探针，于是 `compute_type` 从 float16
    变成 int8，把"按设备选 compute_type"那条既有测试弄红了。
    """

    def test_probe_skips_models_without_transcribe(self):
        captured = {}

        class _Recorder:
            def __init__(self, size, **kwargs):
                captured.update(kwargs)

        fake_module = mock.MagicMock()
        fake_module.WhisperModel = _Recorder
        client = LocalWhisperASR(device="cuda")
        with mock.patch.dict("sys.modules", {"faster_whisper": fake_module}):
            client._load()
        self.assertEqual(captured.get("compute_type"), "float16")
        self.assertEqual(client.device, "cuda", "假模型不该触发降级")

    def test_probe_guard_checks_callable(self):
        import inspect

        src = inspect.getsource(LocalWhisperASR._probe_gpu)
        self.assertIn("callable", src)


if __name__ == "__main__":
    unittest.main()
