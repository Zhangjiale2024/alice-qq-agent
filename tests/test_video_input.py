# -*- coding: utf-8 -*-
"""视频输入测试。

**全部离线**：ffmpeg 调用、视觉接口、ASR 全部用假的。

这是"多模态"的第三块，也是最晚补上的一块——在此之前项目
完全不认 `video` 段（适配层只取 `get_plaintext()`）。

视频的设计是**复用已有两条链路**而不是另起一套：

    视频 --抽帧--> JPEG --+--> VisionClient（图片那条）
         --抽音轨--> WAV --+--> ASR（语音那条）

所以测试的重点是：
  1. 拆分（帧/音轨）是否可靠，边界情况怎么处理
  2. 两部分能不能各自独立成功或失败
  3. 额度是否走到已有的 vision / asr_voice 上，不新增类型
  4. 看不到时是否如实说明（继承护栏）
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq import ffmpeg as ffmpeg_mod
from aliceqq import media_input, policy, video, vision
from aliceqq.engine import IncomingMessage
from tests.test_engine import make_engine

FAKE_VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 400
FAKE_FRAME = b"\xff\xd8\xff" + b"\x00" * 200          # JPEG 魔数
FAKE_WAV = b"RIFF" + b"\x00" * 100


class VideoRefTest(unittest.TestCase):
    def test_blob_is_used_directly(self):
        self.assertEqual(video.VideoRef(blob=FAKE_VIDEO).read_bytes(), FAKE_VIDEO)

    def test_oversized_video_is_rejected(self):
        big = FAKE_VIDEO + b"\x00" * video.VIDEO_MESSAGE_MAX_BYTES
        self.assertEqual(video.VideoRef(blob=big).read_bytes(), b"")

    def test_missing_source_yields_empty(self):
        self.assertEqual(video.VideoRef().read_bytes(), b"")

    def test_nonexistent_file_yields_empty(self):
        self.assertEqual(video.VideoRef(path=r"D:\nope\v.mp4").read_bytes(), b"")

    def test_local_file_is_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.mp4"
            p.write_bytes(FAKE_VIDEO)
            self.assertEqual(video.VideoRef(path=str(p)).read_bytes(), FAKE_VIDEO)


class AnalyzeTest(unittest.TestCase):
    """拆分逻辑。用假的 ffmpeg 函数，不真的调二进制。"""

    def setUp(self):
        self._orig = (ffmpeg_mod.probe_duration, ffmpeg_mod.extract_video_frames,
                      ffmpeg_mod.extract_video_audio)
        self.calls = []

    def tearDown(self):
        (ffmpeg_mod.probe_duration, ffmpeg_mod.extract_video_frames,
         ffmpeg_mod.extract_video_audio) = self._orig

    def _patch(self, frames=(FAKE_FRAME, FAKE_FRAME), audio=FAKE_WAV,
               duration=5.0, frame_err="", audio_err=""):
        def probe(data, exe, timeout=120):
            self.calls.append("probe")
            return duration, ""

        def ext_frames(data, exe, count=3, width=640, start=0.0,
                       duration=0.0, quality=4, timeout=180):
            self.calls.append(("frames", count, width))
            return list(frames), frame_err

        def ext_audio(data, exe, sample_rate=16000, start=0.0,
                      duration=0.0, timeout=180):
            self.calls.append(("audio", duration))
            return audio, audio_err

        ffmpeg_mod.probe_duration = probe
        ffmpeg_mod.extract_video_frames = ext_frames
        ffmpeg_mod.extract_video_audio = ext_audio

    def test_ok_when_both_parts_extracted(self):
        self._patch()
        a = video.analyze(FAKE_VIDEO, "ffmpeg")
        self.assertTrue(a.ok)
        self.assertEqual(len(a.frames), 2)
        self.assertTrue(a.has_audio)

    def test_silent_video_is_still_ok(self):
        # 纯画面视频很常见，没有音轨不该算失败
        self._patch(audio=None)
        a = video.analyze(FAKE_VIDEO, "ffmpeg")
        self.assertTrue(a.ok, "没有音轨的视频不该被判为解析失败")
        self.assertFalse(a.has_audio)
        self.assertIn("没有音轨", a.note)

    def test_frame_failure_but_audio_ok_is_still_ok(self):
        # 画面抽不出来但有声音，仍然值得听
        self._patch(frames=(), frame_err="抽帧失败")
        a = video.analyze(FAKE_VIDEO, "ffmpeg")
        self.assertTrue(a.ok)
        self.assertTrue(a.has_audio)

    def test_total_failure_is_reported(self):
        self._patch(frames=(), audio=None)
        a = video.analyze(FAKE_VIDEO, "ffmpeg")
        self.assertFalse(a.ok)
        self.assertTrue(a.error)

    def test_empty_data_fails_fast(self):
        self._patch()
        a = video.analyze(b"", "ffmpeg")
        self.assertFalse(a.ok)
        self.assertEqual(self.calls, [], "空数据不该去调 ffmpeg")

    def test_no_ffmpeg_fails_with_clear_reason(self):
        self._patch()
        a = video.analyze(FAKE_VIDEO, "")
        self.assertFalse(a.ok)
        self.assertIn("ffmpeg", a.error)
        self.assertEqual(self.calls, [])

    def test_duration_hint_skips_probing(self):
        # QQ 的 video 段通常自带 duration，省一次完整解码
        self._patch()
        video.analyze(FAKE_VIDEO, "ffmpeg", duration_hint=8.0)
        self.assertNotIn("probe", self.calls, "给了时长就不该再去探测")

    def test_long_video_audio_is_clipped(self):
        # 长视频只转写前一段：又慢又贵，而且开场通常已经包含主要信息
        self._patch(duration=600.0)
        a = video.analyze(FAKE_VIDEO, "ffmpeg", max_audio_seconds=120.0)
        audio_calls = [c for c in self.calls if isinstance(c, tuple) and c[0] == "audio"]
        self.assertEqual(audio_calls[0][1], 120.0, "长视频的音轨应当被截断")
        self.assertIn("只转写", a.note)

    def test_frame_count_is_passed_through(self):
        self._patch(frames=tuple([FAKE_FRAME] * 5))
        video.analyze(FAKE_VIDEO, "ffmpeg", frame_count=5, frame_width=480)
        frame_calls = [c for c in self.calls if isinstance(c, tuple) and c[0] == "frames"]
        self.assertEqual(frame_calls[0][1], 5)
        self.assertEqual(frame_calls[0][2], 480)


class RenderTest(unittest.TestCase):
    def test_failure_tells_model_it_cannot_see(self):
        text = video.render_for_model(video.VideoAnalysis(False, error="没有 ffmpeg"))
        self.assertIn("没有拿到内容", text)
        self.assertIn("不要猜测", text)

    def test_both_parts_are_labelled(self):
        a = video.VideoAnalysis(True, frames=[FAKE_FRAME], audio=FAKE_WAV, has_audio=True)
        text = video.render_for_model(
            a, frame_result=vision.VisionResult(True, description="一只猫在跑"),
            transcript="你好啊")
        self.assertIn("【画面】", text)
        self.assertIn("【声音】", text)
        self.assertIn("一只猫在跑", text)
        self.assertIn("你好啊", text)

    def test_states_it_did_not_watch_the_whole_video(self):
        # 这条很重要：模型不该以为自己在"看视频"，否则会编出中间过程
        a = video.VideoAnalysis(True, frames=[FAKE_FRAME], has_audio=False)
        text = video.render_for_model(a)
        self.assertIn("没有从头到尾看过", text)

    def test_no_audio_is_stated(self):
        a = video.VideoAnalysis(True, frames=[FAKE_FRAME], has_audio=False)
        text = video.render_for_model(a)
        self.assertIn("没有音轨", text)

    def test_frames_extracted_but_vision_failed_is_stated(self):
        a = video.VideoAnalysis(True, frames=[FAKE_FRAME], has_audio=False)
        text = video.render_for_model(a, frame_result=vision.VisionResult(False, error="超时"))
        self.assertIn("识别没有成功", text)

    def test_audio_present_but_transcript_failed_is_stated(self):
        a = video.VideoAnalysis(True, frames=[], audio=FAKE_WAV, has_audio=True)
        text = video.render_for_model(a, transcript="")
        self.assertIn("转写没有成功", text)

    def test_transcript_quality_note_is_passed_through(self):
        a = video.VideoAnalysis(True, frames=[FAKE_FRAME], audio=FAKE_WAV, has_audio=True)
        text = video.render_for_model(a, transcript="嗯嗯嗯", transcript_note="识别质量低")
        self.assertIn("识别质量低", text)


class SegmentsToVideosTest(unittest.TestCase):
    def test_video_segment_becomes_videoref(self):
        class Seg:
            type = "video"
            data = {"url": "https://cdn/v.mp4", "file": "v.mp4", "duration": "8"}

        refs = media_input.segments_to_videos([Seg()])
        self.assertEqual(len(refs), 1)
        self.assertIsInstance(refs[0], video.VideoRef)
        self.assertEqual(refs[0].duration, 8.0)

    def test_image_segment_is_not_video(self):
        class Seg:
            type = "image"
            data = {"url": "https://cdn/a.jpg"}

        self.assertEqual(media_input.segments_to_videos([Seg()]), [])

    def test_bad_duration_does_not_drop_the_video(self):
        for bad in ("abc", None, ""):
            class Seg:
                type = "video"
                data = {"url": "https://cdn/v.mp4", "duration": bad}

            refs = media_input.segments_to_videos([Seg()])
            self.assertEqual(len(refs), 1, "duration=%r 不该让整段视频被丢掉" % (bad,))
            self.assertEqual(refs[0].duration, 0.0)

    def test_broken_segment_yields_empty(self):
        class Broken:
            @property
            def type(self):
                raise RuntimeError("坏段")

        self.assertEqual(media_input.segments_to_videos([Broken()]), [])


class EngineVideoTest(unittest.TestCase):
    """引擎侧接线。ffmpeg 与两条模型链路都用假的。"""

    def setUp(self):
        self._orig = (ffmpeg_mod.find_ffmpeg, ffmpeg_mod.probe_duration,
                      ffmpeg_mod.extract_video_frames, ffmpeg_mod.extract_video_audio,
                      vision.net.open)

    def tearDown(self):
        (ffmpeg_mod.find_ffmpeg, ffmpeg_mod.probe_duration,
         ffmpeg_mod.extract_video_frames, ffmpeg_mod.extract_video_audio,
         vision.net.open) = self._orig

    def _patch_all(self, frames=(FAKE_FRAME,), audio=FAKE_WAV, vision_ok=True):
        ffmpeg_mod.find_ffmpeg = lambda config=None, explicit="": "ffmpeg"
        ffmpeg_mod.probe_duration = lambda data, exe, timeout=120: (5.0, "")
        ffmpeg_mod.extract_video_frames = (
            lambda data, exe, count=3, width=640, start=0.0, duration=0.0,
            quality=4, timeout=180: (list(frames), ""))
        ffmpeg_mod.extract_video_audio = (
            lambda data, exe, sample_rate=16000, start=0.0, duration=0.0,
            timeout=180: (audio, ""))

        if vision_ok:
            import json

            class _Resp:
                def read(self):
                    return json.dumps({"choices": [
                        {"message": {"content": "深蓝色画面，写着 ALICE VIDEO"}}]}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            vision.net.open = lambda request, timeout=None: _Resp()

    def _engine(self, **kw):
        from aliceqq.config import VisionConfig

        engine = make_engine()
        engine.config.vision = VisionConfig(enabled=True, base_url="https://x/v1",
                                           api_key="k", model="m")
        engine.vision_client = vision.build_client(engine.config.vision)
        engine.asr_client = _FakeASR()
        self._patch_all(**kw)
        return engine

    def test_video_only_message_is_not_dropped(self):
        # 和纯图片/纯语音一样：没有文字也不能被当空消息丢掉
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        outcome = engine.handle(msg)
        self.assertNotEqual(outcome.reason, "空消息")
        self.assertTrue(outcome.debug["video"]["ok"])

    def test_video_debug_reports_both_parts(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="看这个", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        outcome = engine.handle(msg)
        info = outcome.debug["video"]
        self.assertEqual(info["frames"], 1)
        self.assertTrue(info["frames_ok"])
        self.assertTrue(info["has_audio"])
        self.assertGreater(info["transcript_chars"], 0)

    def test_video_block_contains_frames_and_speech(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        engine.handle(msg)
        block = engine._video_block(msg)
        self.assertIn("ALICE VIDEO", block)
        self.assertIn("打游戏", block)

    def test_no_ffmpeg_says_so_instead_of_guessing(self):
        engine = self._engine()
        ffmpeg_mod.find_ffmpeg = lambda config=None, explicit="": ""
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        outcome = engine.handle(msg)
        self.assertFalse(outcome.debug["video"]["ok"])
        self.assertIn("没有拿到内容", engine._video_block(msg))

    def test_unreadable_video_is_reported(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=b"")])
        outcome = engine.handle(msg)
        self.assertFalse(outcome.debug["video"]["ok"])
        self.assertIn("没有拿到内容", engine._video_block(msg))

    def test_video_consumes_vision_and_voice_quota(self):
        # 视频**不新增额度类型**：抽帧算 vision，音轨算 asr_voice
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        engine.handle(msg)
        period = policy._period_day()
        self.assertEqual(engine.storage.get_counter("quota_vision_user", "1", period), 1)
        self.assertEqual(engine.storage.get_counter("quota_asrvoice_user", "1", period), 1)
        self.assertEqual(engine.storage.get_counter("quota_video_user", "1", period), 0,
                         "不该凭空多出一个 video 额度类型")

    def test_video_does_not_leak_across_turns(self):
        engine = self._engine()
        first = IncomingMessage(user_id="1", text="", is_private=True, user_name="a",
                                videos=[video.VideoRef(blob=FAKE_VIDEO)])
        engine.handle(first)
        self.assertIn("ALICE VIDEO", engine._video_block(first))

        second = IncomingMessage(user_id="2", text="你好", is_private=True, user_name="b")
        engine.handle(second)
        self.assertEqual(engine._video_block(second), "",
                         "第二轮没有视频时不能带上上一轮的解析结果")

    def test_no_video_means_no_block(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="你好", is_private=True, user_name="x")
        self.assertEqual(engine._video_block(msg), "")

    def test_video_failure_does_not_break_the_reply(self):
        engine = self._engine(frames=(), audio=None)
        msg = IncomingMessage(user_id="1", text="看看", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=FAKE_VIDEO)])
        outcome = engine.handle(msg)          # 不该抛异常
        self.assertTrue(outcome.reply)


class _FakeASR:
    available = True

    def transcribe(self, audio, filename="audio.m4a"):
        from aliceqq import asr as asr_mod

        return asr_mod.Transcription(True, text="艾利斯今天想去打游戏",
                                     provider="fake", seconds=2.0)


class UnifiedInputTest(unittest.TestCase):
    def test_analyze_inputs_covers_all_three_modalities(self):
        src = (Path(__file__).resolve().parent.parent /
               "aliceqq" / "engine.py").read_text(encoding="utf-8")
        block_start = src.index("def _analyze_inputs")
        block = src[block_start:block_start + 700]
        for call in ("self._analyze_images(message, outcome)",
                     "self._analyze_audio(message, outcome)",
                     "self._analyze_video(message, outcome)"):
            self.assertIn(call, block, "统一入口漏了 %s" % call)

    def test_message_has_all_three_fields(self):
        from aliceqq import asr as asr_mod

        msg = IncomingMessage(
            user_id="1", text="",
            images=[vision.ImageRef(blob=b"\x89PNG\r\n\x1a\n")],
            audios=[asr_mod.AudioRef(blob=b"OggS")],
            videos=[video.VideoRef(blob=FAKE_VIDEO)])
        self.assertTrue(msg.has_images)
        self.assertTrue(msg.has_audios)
        self.assertTrue(msg.has_videos)
        self.assertTrue(msg.has_media_input)


class PromptTemplateTest(unittest.TestCase):
    def test_templates_have_video_placeholder(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("alice_system.md", "assistant_system.md"):
            text = (root / "prompts" / name).read_text(encoding="utf-8")
            self.assertIn("{{video_result}}", text, "%s 缺少 {{video_result}}" % name)

    def test_templates_say_not_watched_fully(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("alice_system.md", "assistant_system.md"):
            text = (root / "prompts" / name).read_text(encoding="utf-8")
            i = text.find("{{video_result}}")
            section = text[max(0, i - 300):i + 800]
            self.assertIn("视频", section)
            self.assertIn("不要猜", section,
                          "%s 没有禁止在拿不到内容时猜测" % name)


class AdapterWiringTest(unittest.TestCase):
    def test_adapter_extracts_videos(self):
        src = (Path(__file__).resolve().parent.parent / "aliceqq" /
               "adapters" / "nonebot_plugin_alice.py").read_text(encoding="utf-8")
        self.assertIn("videos = _extract_videos(event)", src)
        self.assertIn("videos=videos,", src)
        self.assertIn("media_input_mod.segments_to_videos", src)


if __name__ == "__main__":
    unittest.main()
