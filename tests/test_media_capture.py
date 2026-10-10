"""素材抓取（从 OneBot 消息段入库）的测试。

这些用例对应三个**真机上才暴露出来的** bug：
  1. 照片给的是 http URL，早期只认本地文件 -> 全被跳过；
  2. QQ 表情是 face 段，早期只认 image/record -> 完全不进分支；
  3. 失败时不给反馈 -> 用户发完图什么都看不到。

逻辑放在 aliceqq.media 而不是适配层，就是为了能被这里测到
（适配层在 if HAS_NONEBOT 块内，没装 NoneBot 时根本不存在）。
"""
from __future__ import annotations

import io
import os
import shutil
import unittest
import uuid
from unittest import mock

from aliceqq.media import (FACE_REASON, MediaLibrary, classify_segment,
                           guess_media_name, read_media_bytes, reason_text,
                           segment_hint_name, sniff_extension, store_media)

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _tmp_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "test-media")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "case-" + uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    return path


class ClassifySegmentTest(unittest.TestCase):
    def test_face_is_rejected_with_clear_reason(self):
        """QQ 内置表情拿不到图片数据——必须给出明确原因，不能静默跳过。"""
        kind, source, reason = classify_segment("face", {"faceIndex": 312,
                                                         "faceText": "/变形"})
        self.assertIsNone(kind)
        self.assertEqual(reason, FACE_REASON)
        self.assertIn("内置表情", reason_text(reason))

    def test_image_with_url_is_accepted(self):
        """真机日志里 NapCat 给照片的就是 http URL。"""
        kind, source, reason = classify_segment("image", {"file": "http://x/a.jpg"})
        self.assertEqual(kind, "sticker")
        self.assertEqual(source, "http://x/a.jpg")
        self.assertEqual(reason, "")

    def test_url_field_takes_priority_over_file(self):
        kind, source, _ = classify_segment("image", {"file": "a.jpg",
                                                     "url": "http://cdn/b.png"})
        self.assertEqual(source, "http://cdn/b.png")

    def test_image_with_local_file(self):
        kind, source, reason = classify_segment("image", {"file": "D:/x/a.png"})
        self.assertEqual((kind, source, reason), ("sticker", "D:/x/a.png", ""))

    def test_record_becomes_voice(self):
        kind, _, reason = classify_segment("record", {"file": "a.silk"})
        self.assertEqual(kind, "voice")
        self.assertEqual(reason, "")

    def test_image_without_source(self):
        kind, source, reason = classify_segment("image", {})
        self.assertIsNone(kind)
        self.assertEqual(reason, "no_source")

    def test_unsupported_segment(self):
        kind, _, reason = classify_segment("text", {"text": "hi"})
        self.assertIsNone(kind)
        self.assertEqual(reason, "unsupported")

    def test_market_face_still_face(self):
        # 表情包（market face）带 packId，但段类型依然是 face
        kind, _, reason = classify_segment("face", {"packId": "1", "faceType": 2})
        self.assertIsNone(kind)
        self.assertEqual(reason, FACE_REASON)


class ReadMediaBytesTest(unittest.TestCase):
    def setUp(self):
        self.root = _tmp_dir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_local_path(self):
        path = os.path.join(self.root, "a.png")
        with open(path, "wb") as fh:
            fh.write(PNG)
        self.assertEqual(read_media_bytes(path), PNG)

    def test_file_uri(self):
        path = os.path.join(self.root, "b.png")
        with open(path, "wb") as fh:
            fh.write(PNG)
        self.assertEqual(read_media_bytes("file:///" + path.replace("\\", "/")), PNG)

    def test_http_url_is_downloaded(self):
        """这里是关键回归：早期 os.path.isfile 判断会把 URL 全部拒掉。"""
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                return PNG

        with mock.patch("aliceqq.media.urllib.request.build_opener") as build:
            build.return_value.open.return_value = _Resp()
            data = read_media_bytes("http://cdn.example/a.png")
        self.assertEqual(data, PNG)

    def test_http_failure_returns_empty_not_exception(self):
        with mock.patch("aliceqq.media.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = OSError("boom")
            with mock.patch("aliceqq.media.time.sleep", lambda *_: None):
                self.assertEqual(read_media_bytes("http://cdn.example/a.png"), b"")

    def test_missing_file_and_empty_source(self):
        self.assertEqual(read_media_bytes(""), b"")
        self.assertEqual(read_media_bytes("D:/definitely/not/here.png"), b"")

    def test_non_http_scheme_is_rejected(self):
        self.assertEqual(read_media_bytes("ftp://x/a.png"), b"")

    def test_proxy_is_bypassed(self):
        """这台机器的系统代理指向一个没在运行的端口，必须显式禁用。"""
        with mock.patch("aliceqq.media.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = OSError("no")
            with mock.patch("aliceqq.media.time.sleep", lambda *_: None):
                read_media_bytes("http://cdn/a.png")
        # build_opener 必须收到一个 ProxyHandler（且是"绕过所有代理"的那种）
        import urllib.request
        args = build.call_args[0]
        self.assertTrue(args, "应该显式传入 ProxyHandler，否则会走失效的系统代理")
        handler = args[0]
        self.assertIsInstance(handler, urllib.request.ProxyHandler)
        self.assertEqual(handler.proxies, {}, "ProxyHandler 必须是空字典（=不用任何代理）")


class FormatSniffTest(unittest.TestCase):
    """URL 不带扩展名时必须能从字节认出格式。

    真机踩过：QQ 图片 URL 是 multimedia.nt.qq.com.cn/download?appid=...
    路径里没有扩展名，按文件名推出来是 "download"，
    于是 library.add 抛 ValueError: 不支持的格式，素材永远入不了库。
    """

    def test_sniff_common_formats(self):
        cases = [
            (b"\x89PNG\r\n\x1a\n" + b"0" * 40, ".png"),
            (b"\xff\xd8\xff\xe0" + b"0" * 40, ".jpg"),
            (b"GIF89a" + b"0" * 40, ".gif"),
            (b"GIF87a" + b"0" * 40, ".gif"),
            (b"BM" + b"0" * 40, ".bmp"),
        ]
        for blob, expected in cases:
            self.assertEqual(sniff_extension(blob, "sticker"), expected)

    def test_riff_disambiguates_webp_and_wav(self):
        webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"0" * 20
        wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"0" * 20
        self.assertEqual(sniff_extension(webp, "sticker"), ".webp")
        self.assertEqual(sniff_extension(wav, "voice"), ".wav")

    def test_unknown_bytes_return_empty(self):
        self.assertEqual(sniff_extension(b"\x00" * 100), "")
        self.assertEqual(sniff_extension(b""), "")

    def test_guess_prefers_real_filename(self):
        # source 是没扩展名的 URL，但 hint_name 是真文件名
        name = guess_media_name("https://multimedia.nt.qq.com.cn/download?appid=1",
                                "sticker", b"\xff\xd8\xff\xe0" + b"0" * 40,
                                hint_name="D38114B8.jpg")
        self.assertTrue(name.endswith(".jpg"), name)

    def test_guess_falls_back_to_sniffing(self):
        name = guess_media_name("https://x/download?f=1", "sticker",
                                b"\x89PNG\r\n\x1a\n" + b"0" * 40)
        self.assertTrue(name.endswith(".png"), name)

    def test_guess_raises_with_actionable_message(self):
        with self.assertRaises(ValueError) as ctx:
            guess_media_name("https://x/download?f=1", "sticker", b"\x00" * 100)
        # 错误信息要能指导用户怎么办，不能只说"失败"
        self.assertIn("换一张", str(ctx.exception))

    def test_segment_hint_name(self):
        self.assertEqual(segment_hint_name({"file": "a.jpg", "url": "http://x/1"}),
                         "a.jpg")
        self.assertEqual(segment_hint_name({"url": "http://x/1"}), "")

    def test_real_qq_sticker_payload_is_accepted(self):
        """真机日志里的完整 image 段。"""
        data = {
            "summary": "[动画表情]",
            "file": "D38114B8C89E6EA98DA1F087B6B20AE8.jpg",
            "sub_type": "1",
            "url": ("https://multimedia.nt.qq.com.cn/download?appid=1406"
                    "&fileid=EhRbZjel&spec=0&rkey=abc"),
            "file_size": "178238",
        }
        kind, source, reason = classify_segment("image", data)
        self.assertEqual(kind, "sticker")
        self.assertEqual(reason, "")
        self.assertIn("multimedia.nt.qq.com.cn", source)
        self.assertTrue(segment_hint_name(data).endswith(".jpg"))


class StoreMediaTest(unittest.TestCase):
    def setUp(self):
        self.root = _tmp_dir()
        self.library = MediaLibrary(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_store_from_url_derives_filename(self):
        item = store_media(self.library, "http://cdn.example/pic.png?x=1",
                           "sticker", PNG, "困,累", "打哈欠")
        self.assertEqual(item.kind, "sticker")
        self.assertEqual(item.tags, ["困", "累"])
        self.assertEqual(item.desc, "打哈欠")
        self.assertTrue(self.library.exists(item))

    def test_store_voice(self):
        item = store_media(self.library, "http://cdn/v.silk", "voice",
                           b"ID3" + b"0" * 32, "问候", "打招呼")
        self.assertEqual(item.kind, "voice")
        self.assertTrue(self.library.exists(item))

    def test_stored_item_is_pickable_by_tag(self):
        store_media(self.library, "http://cdn/a.png", "sticker", PNG, "困", "困了")
        self.assertIsNotNone(self.library.pick("好困啊", kind="sticker"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
