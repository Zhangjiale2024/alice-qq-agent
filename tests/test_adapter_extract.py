# -*- coding: utf-8 -*-
"""从 OneBot 消息段抽取媒体输入的测试。

**这个文件补的是项目里最大的测试空白。**
在此之前没有任何测试碰过这段逻辑，后果是：

  * 图片段被 `get_plaintext()` 丢掉（所以"理解图片"一直无从谈起）
  * 语音段同样被丢掉（所以"发语音它不回"）
  * 两个问题都只在真机上暴露，而真机验证最难做

测的是 `aliceqq/media_input.py` 而不是适配层里的包装函数——
**这正是这次重构的目的**：适配层整个包在 `if HAS_NONEBOT:` 里，
没有 NoneBot 的环境（包括 CI）根本执行不到，那层逻辑一行都测不了。
移到核心包之后，只用假 segment 对象（duck typing）就能覆盖。

`HandlerWiringTest` 仍然对适配层做**静态检查**，
因为"抽了但忘了接线"是这里最容易犯的错。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq import asr as asr_mod
from aliceqq import media_input, vision

ADAPTER = (Path(__file__).resolve().parent.parent /
           "aliceqq" / "adapters" / "nonebot_plugin_alice.py")


class FakeSegment:
    def __init__(self, type_, **data):
        self.type = type_
        self.data = data


class BrokenSegment:
    """type 属性访问就抛异常，用来验证整段不会因此崩掉。"""

    @property
    def type(self):
        raise RuntimeError("模拟坏段")

    @property
    def data(self):
        raise RuntimeError("模拟坏段")


class RawSegment:
    """没有 type/data 属性的对象（NoneBot 之外的输入）。"""

    pass


class SegmentsToImagesTest(unittest.TestCase):
    def test_image_segment_becomes_imageref(self):
        refs = media_input.segments_to_images(
            [FakeSegment("image", url="https://cdn/a.jpg", file="a.jpg")])
        self.assertEqual(len(refs), 1)
        self.assertIsInstance(refs[0], vision.ImageRef)
        self.assertEqual(refs[0].url, "https://cdn/a.jpg")

    def test_local_file_without_url_is_kept(self):
        # NapCat 有时给的是本地缓存路径而不是 URL
        refs = media_input.segments_to_images([FakeSegment("image", file="C:/cache/b.png")])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].path, "C:/cache/b.png")

    def test_segment_without_source_is_skipped(self):
        self.assertEqual(media_input.segments_to_images([FakeSegment("image")]), [])

    def test_text_segments_are_ignored(self):
        refs = media_input.segments_to_images([
            FakeSegment("text", text="你好"),
            FakeSegment("image", url="https://cdn/a.jpg")])
        self.assertEqual(len(refs), 1)

    def test_face_segment_is_not_an_image(self):
        # QQ 内置表情拿不到图片数据，不该当成图片送去做视觉理解
        self.assertEqual(media_input.segments_to_images([FakeSegment("face", faceIndex="1")]), [])

    def test_multiple_images_are_all_collected(self):
        refs = media_input.segments_to_images([
            FakeSegment("image", url="https://cdn/a.jpg"),
            FakeSegment("image", url="https://cdn/b.jpg")])
        self.assertEqual(len(refs), 2)

    def test_empty_input_yields_empty(self):
        self.assertEqual(media_input.segments_to_images([]), [])

    def test_broken_segment_yields_empty_instead_of_raising(self):
        self.assertEqual(media_input.segments_to_images([BrokenSegment()]), [])

    def test_object_without_attributes_is_skipped(self):
        self.assertEqual(media_input.segments_to_images([RawSegment()]), [])

    def test_debug_logger_is_called_when_images_found(self):
        # 静默返回空会让"为什么它看不见我发的图"变成没有线索的问题
        logged = []
        media_input.segments_to_images(
            [FakeSegment("image", url="https://cdn/a.jpg")], logged.append)
        self.assertTrue(logged, "找到图片时应当留下日志")

    def test_debug_logger_is_called_on_failure(self):
        logged = []
        media_input.segments_to_images([BrokenSegment()], logged.append)
        self.assertTrue(logged, "解析失败时更要留下线索")


class SegmentsToAudiosTest(unittest.TestCase):
    def test_record_segment_becomes_audioref(self):
        refs = media_input.segments_to_audios(
            [FakeSegment("record", url="https://cdn/v.amr", file="v.amr")])
        self.assertEqual(len(refs), 1)
        self.assertIsInstance(refs[0], asr_mod.AudioRef)
        self.assertEqual(refs[0].url, "https://cdn/v.amr")

    def test_duration_is_parsed(self):
        refs = media_input.segments_to_audios(
            [FakeSegment("record", url="https://cdn/v.amr", duration="12")])
        self.assertEqual(refs[0].duration, 12.0)

    def test_bad_duration_does_not_break_extraction(self):
        # QQ 的 duration 有时是字符串、有时缺失、格式未必规范；
        # **一个坏字段不该让整条语音被丢掉**
        for bad in ("abc", "", None, [], {}):
            refs = media_input.segments_to_audios(
                [FakeSegment("record", url="https://cdn/v.amr", duration=bad)])
            self.assertEqual(len(refs), 1, "duration=%r 不该导致整段被丢掉" % (bad,))
            self.assertEqual(refs[0].duration, 0.0)

    def test_local_file_is_kept(self):
        refs = media_input.segments_to_audios([FakeSegment("record", file="C:/cache/v.amr")])
        self.assertEqual(refs[0].path, "C:/cache/v.amr")

    def test_segment_without_source_is_skipped(self):
        self.assertEqual(media_input.segments_to_audios([FakeSegment("record")]), [])

    def test_image_segment_is_not_audio(self):
        # 两个抽取器各管各的，不能互相串
        self.assertEqual(media_input.segments_to_audios(
            [FakeSegment("image", url="https://cdn/a.jpg")]), [])

    def test_broken_segment_yields_empty_instead_of_raising(self):
        self.assertEqual(media_input.segments_to_audios([BrokenSegment()]), [])

    def test_filename_does_not_contain_path_separators(self):
        # 文件名要能当扩展名判断依据，所以不能把整条路径塞进去
        for raw in ("https://cdn/x/y/voice.amr", r"C:\cache\audio\v.amr"):
            refs = media_input.segments_to_audios([FakeSegment("record", url=raw)])
            self.assertNotIn("/", refs[0].filename)
            self.assertNotIn("\\", refs[0].filename)
            self.assertLessEqual(len(refs[0].filename), 40)

    def test_debug_logger_is_called(self):
        logged = []
        media_input.segments_to_audios(
            [FakeSegment("record", url="https://cdn/v.amr")], logged.append)
        self.assertTrue(logged)


class MixedMessageTest(unittest.TestCase):
    """一条消息同时带图和语音（QQ 上可以）。"""

    def test_both_are_extracted(self):
        segs = [
            FakeSegment("text", text="看看这个"),
            FakeSegment("image", url="https://cdn/a.jpg"),
            FakeSegment("record", url="https://cdn/v.amr"),
        ]
        self.assertEqual(len(media_input.segments_to_images(segs)), 1)
        self.assertEqual(len(media_input.segments_to_audios(segs)), 1)

    def test_order_does_not_matter(self):
        segs = [FakeSegment("record", url="https://cdn/v.amr"),
                FakeSegment("image", url="https://cdn/a.jpg")]
        self.assertEqual(len(media_input.segments_to_images(segs)), 1)
        self.assertEqual(len(media_input.segments_to_audios(segs)), 1)


class HandlerWiringTest(unittest.TestCase):
    """静态检查：`_handle` 真的把抽取结果放进了 IncomingMessage。

    跑 `_handle` 需要完整的 NoneBot 运行时（Bot 对象、matcher、事件循环），
    成本远高于收益。而这里真正容易出错的只是"抽了但忘了接线"，
    静态检查正好覆盖它——**图片那次的 bug 就是漏了这一环**。
    """

    def setUp(self):
        self.src = ADAPTER.read_text(encoding="utf-8")

    def test_images_are_passed_into_incoming_message(self):
        self.assertIn("images = _extract_images(event)", self.src)
        self.assertIn("images=images,", self.src)

    def test_audios_are_passed_into_incoming_message(self):
        self.assertIn("audios = _extract_audios(event)", self.src)
        self.assertIn("audios=audios,", self.src)


class AdapterStaysThinTest(unittest.TestCase):
    """适配层只做翻译，业务逻辑不该回到这里。

    这条规则是踩出来的：`test_media_capture.py` 的模块说明里早就写过
    "逻辑放在 aliceqq.media 而不是适配层，就是为了能被测试到"，
    但后来写图片/语音抽取时我又把逻辑写进了适配层，于是重蹈覆辙——
    那个 `if HAS_NONEBOT` 让整块逻辑在 CI 里永远执行不到。
    """

    def setUp(self):
        self.src = ADAPTER.read_text(encoding="utf-8")

    def test_extract_functions_are_thin_forwarders(self):
        # 适配层里的抽取函数不该再出现解析循环
        self.assertIn("media_input_mod.segments_to_images", self.src)
        self.assertIn("media_input_mod.segments_to_audios", self.src)
        self.assertNotIn('!= "image"', self.src,
                         "图片解析逻辑不该留在适配层")
        self.assertNotIn('!= "record"', self.src,
                         "语音解析逻辑不该留在适配层")

    def test_media_logic_lives_in_core(self):
        root = Path(__file__).resolve().parent.parent
        core = (root / "aliceqq" / "media.py").read_text(encoding="utf-8")
        self.assertIn("def cq_code_for", core,
                      "CQ 码构造属于核心逻辑，应当能脱离 NoneBot 测试")

    def test_vision_logic_lives_in_core(self):
        root = Path(__file__).resolve().parent.parent
        core = (root / "aliceqq" / "vision.py").read_text(encoding="utf-8")
        self.assertIn("def render_for_model", core)
        self.assertIn("def build_client", core)

    def test_asr_logic_lives_in_core(self):
        root = Path(__file__).resolve().parent.parent
        core = (root / "aliceqq" / "asr.py").read_text(encoding="utf-8")
        self.assertIn("class AudioRef", core)
        self.assertIn("def quality_warning", core)


if __name__ == "__main__":
    unittest.main()
