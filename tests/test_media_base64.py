# -*- coding: utf-8 -*-
"""`base64://` 媒体来源测试。

**这一组守着一个实测发现的静默失效。**

OneBot v11 规范里 image / record / video 三段的 `file` 字段都允许
`base64://<数据>` 形式（见规范"发送时 file 还支持 base64://…"那段）。
NapCat 配置里 `enableLocalFile2Url=false` 时更可能直接给 base64。

而项目原来完全不认这个形式，表现是：

    read_media_bytes("base64://...")  ->  b""（0 字节）
    ImageRef.to_data_uri()            ->  ""（空串）

**没有任何报错**，只有"它看不见我发的东西"。三种模态一起失效。

值得注意的是修的过程里出现过一次**只修好两种**的情况：
语音和视频走 `read_media_bytes`（补上 base64 后就好了），
而图片走 `ImageRef.to_data_uri`（它只认本地 path），
于是"三种模态里坏一种"。所以这里的用例是**逐模态**的。
"""
from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq import asr as asr_mod
from aliceqq import media_input, video, vision
from aliceqq.media import read_media_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff" + b"\x00" * 64
WAV = b"RIFF" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def b64(data: bytes) -> str:
    return "base64://" + base64.b64encode(data).decode("ascii")


class Segment:
    def __init__(self, type_, **data):
        self.type = type_
        self.data = data


class ReadMediaBytesTest(unittest.TestCase):
    def test_base64_uri_is_decoded(self):
        self.assertEqual(read_media_bytes(b64(PNG)), PNG)

    def test_base64_with_whitespace_is_tolerated(self):
        # 有些实现会把长数据按行折开，解码前必须去掉空白
        payload = base64.b64encode(PNG).decode()
        folded = "\n".join(payload[i:i + 20] for i in range(0, len(payload), 20))
        self.assertEqual(read_media_bytes("base64://" + folded), PNG)

    def test_broken_base64_yields_empty_not_exception(self):
        self.assertEqual(read_media_bytes("base64://!!!not-base64!!!"), b"")

    def test_empty_base64_yields_empty(self):
        self.assertEqual(read_media_bytes("base64://"), b"")

    def test_unknown_scheme_still_yields_empty(self):
        # 不能因为支持 base64 就把"无法识别"变成"读到了东西"
        self.assertEqual(read_media_bytes("ftp://example.com/a.jpg"), b"")

    def test_oversized_base64_is_truncated_or_empty(self):
        big = b"\x00" * (20 * 1024 * 1024)
        got = read_media_bytes(b64(big))
        self.assertLessEqual(len(got), 20 * 1024 * 1024)


class ImageBase64Test(unittest.TestCase):
    def test_data_uri_from_file_field(self):
        # 规范里 base64 来自 `file`，适配层把它放进 path
        ref = vision.ImageRef(path=b64(PNG))
        uri = ref.to_data_uri()
        self.assertTrue(uri.startswith("data:image/png;base64,"), uri[:40])
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), PNG)

    def test_data_uri_from_url_field(self):
        # 也有的实现放在 url；两个字段都要认
        ref = vision.ImageRef(url=b64(PNG))
        self.assertTrue(ref.to_data_uri().startswith("data:image/png;base64,"))

    def test_mime_is_detected_from_decoded_bytes(self):
        self.assertTrue(vision.ImageRef(path=b64(JPG)).to_data_uri()
                        .startswith("data:image/jpeg;base64,"))

    def test_segment_to_imageref_base64_round_trip(self):
        # 端到端：段 -> ImageRef -> data URI
        refs = media_input.segments_to_images([Segment("image", file=b64(PNG))])
        self.assertEqual(len(refs), 1)
        uri = refs[0].to_data_uri()
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), PNG)

    def test_broken_base64_yields_empty_uri(self):
        refs = media_input.segments_to_images([Segment("image", file="base64://!!bad!!")])
        self.assertEqual(len(refs), 1, "段本身应当被接受")
        self.assertEqual(refs[0].to_data_uri(), "", "坏数据要返回空而不是崩")


class AudioBase64Test(unittest.TestCase):
    def test_read_bytes_decodes_base64(self):
        ref = asr_mod.AudioRef(path=b64(WAV))
        self.assertEqual(ref.read_bytes(), WAV)

    def test_segment_round_trip(self):
        refs = media_input.segments_to_audios([Segment("record", file=b64(WAV))])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].read_bytes(), WAV)


class VideoBase64Test(unittest.TestCase):
    def test_read_bytes_decodes_base64(self):
        ref = video.VideoRef(path=b64(MP4))
        self.assertEqual(ref.read_bytes(), MP4)

    def test_segment_round_trip(self):
        refs = media_input.segments_to_videos([Segment("video", file=b64(MP4))])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].read_bytes(), MP4)


class AllModalitiesTest(unittest.TestCase):
    """三种模态必须**同时**能用。

    修这个 bug 的过程中出现过"只修好两种"：语音/视频走
    `read_media_bytes`（补上 base64 就好了），图片走
    `ImageRef.to_data_uri`（只认本地 path），于是只有图片坏。
    这条用例一次性覆盖三种，避免以后再出现这种偏差。
    """

    def test_one_segment_of_each_kind(self):
        segs = [Segment("image", file=b64(PNG)),
                Segment("record", file=b64(WAV)),
                Segment("video", file=b64(MP4))]
        images = media_input.segments_to_images(segs)
        audios = media_input.segments_to_audios(segs)
        videos = media_input.segments_to_videos(segs)

        self.assertTrue(images[0].to_data_uri(), "图片模态的 base64 没读出来")
        self.assertEqual(audios[0].read_bytes(), WAV, "语音模态的 base64 没读出来")
        self.assertEqual(videos[0].read_bytes(), MP4, "视频模态的 base64 没读出来")


class HttpStillPreferredTest(unittest.TestCase):
    """加 base64 支持不能改变 http(s) 的既有行为。"""

    def test_http_url_is_still_passed_through(self):
        ref = vision.ImageRef(url="https://cdn/a.jpg")
        self.assertEqual(ref.to_url_or_uri(), "https://cdn/a.jpg")

    def test_data_uri_is_still_passed_through(self):
        given = "data:image/webp;base64,AAAA"
        self.assertEqual(vision.ImageRef(url=given).to_data_uri(), given)

    def test_base64_is_not_mistaken_for_http(self):
        ref = vision.ImageRef(url=b64(PNG))
        # 不能被当成远程地址直接透传，否则接口收到一个不认识的 scheme
        self.assertFalse(ref.to_url_or_uri().startswith("base64://"))


if __name__ == "__main__":
    unittest.main()
