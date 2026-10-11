# -*- coding: utf-8 -*-
"""给"纯媒体消息的 user 占位文本"写测试。

**这是真机验证发现的 bug 的回归守卫。**

真机日志里模型多次回"你只发了一个空消息""爱丽丝没收到内容哦"，
而同一轮的 debug 里 `vision: {'ok': True}`。查下来的根因：

用户只发图/发语音时 `message.text` 是空的，原来的代码把空字符串
原样塞进 user 消息：

    messages.append({"role": "user", "content": current_text or message.text})

图片描述虽然在 system 里，但**连续的「空 user 消息」是退化输入模式**，
模型会否认收到了东西。本地稳定复现过（同一用例跑多次，
有时承认、有时否认）。

修法是给一句固定占位文本，让模型有具体东西可回应。
实测修复后同一用例 5/5 全部承认，0 次否认。

这里只测**确定性部分**（占位文本有没有正确生成并进入 messages）；
"模型因此不再否认"是模型行为，靠上面的实测数字说明，
不放进单元测试（那需要真实 API）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aliceqq import asr as asr_mod
from aliceqq import video, vision
from aliceqq.engine import IncomingMessage
from tests.test_engine import make_engine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _SpyLLM:
    name = "spy"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, config, tools=None):
        self.calls.append([dict(m) for m in messages])
        from aliceqq.llm import Reply
        return Reply("（spy）", provider="spy", model="spy")

    def user_contents(self, i=0):
        return [m.get("content") or "" for m in self.calls[i]
                if m.get("role") == "user"]


class PlaceholderTextTest(unittest.TestCase):
    def _engine(self):
        engine = make_engine()
        engine.config.search.enabled = False
        engine.config.features.web_search = False
        engine.llm = _SpyLLM()
        return engine

    def test_placeholder_for_image_only(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        text = engine._media_placeholder(msg)
        self.assertIn("图片", text)
        self.assertIn("没有打字", text)

    def test_placeholder_for_audio_only(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              audios=[asr_mod.AudioRef(blob=b"OggS")])
        self.assertIn("语音", engine._media_placeholder(msg))

    def test_placeholder_for_video_only(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              videos=[video.VideoRef(blob=b"\x00\x00\x00\x18ftypmp42")])
        self.assertIn("视频", engine._media_placeholder(msg))

    def test_placeholder_lists_all_kinds(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)],
                              audios=[asr_mod.AudioRef(blob=b"OggS")])
        text = engine._media_placeholder(msg)
        self.assertIn("图片", text)
        self.assertIn("语音", text)

    def test_no_placeholder_without_media(self):
        # 没媒体、没文字时不该凭空造一句
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x")
        self.assertEqual(engine._media_placeholder(msg), "")

    def test_placeholder_does_not_describe_content(self):
        """占位文本只说明"发了什么类型"，不描述内容。

        真正的内容描述由 system 里的 vision/transcript/video 区块提供，
        那是**唯一**的信息来源。如果占位文本也描述内容，模型可能
        把两处混起来编（项目里已有"两处都写 = 静默冲突"的教训）。
        """
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        text = engine._media_placeholder(msg)
        for word in ("猫", "文字", "画面", "颜色", "内容为"):
            self.assertNotIn(word, text, "占位文本不该描述内容：%r" % word)


class UserMessageTest(unittest.TestCase):
    """真正发出去的 user 消息不能是空字符串。"""

    def _engine_with_spy(self):
        engine = make_engine()
        engine.config.search.enabled = False
        engine.config.features.web_search = False
        spy = _SpyLLM()
        engine.llm = spy
        return engine, spy

    def test_image_only_message_has_non_empty_user_content(self):
        engine, spy = self._engine_with_spy()
        engine.handle(IncomingMessage(user_id="1", text="", is_private=True,
                                      user_name="x",
                                      images=[vision.ImageRef(blob=PNG)]))
        contents = spy.user_contents(0)
        self.assertTrue(contents, "没有抓到 user 消息")
        self.assertTrue(contents[-1].strip(),
                        "user 消息是空的——这正是真机上模型否认收到图的根因")

    def test_audio_only_message_has_non_empty_user_content(self):
        engine, spy = self._engine_with_spy()
        engine.asr_client = None
        engine.handle(IncomingMessage(user_id="1", text="", is_private=True,
                                      user_name="x",
                                      audios=[asr_mod.AudioRef(blob=b"OggS" + b"\x00" * 40)]))
        self.assertTrue(spy.user_contents(0)[-1].strip())

    def test_text_message_is_unchanged(self):
        # 有文字时不该被占位文本替换掉
        engine, spy = self._engine_with_spy()
        engine.handle(IncomingMessage(user_id="1", text="你好呀", is_private=True,
                                      user_name="x"))
        self.assertEqual(spy.user_contents(0)[-1], "你好呀")

    def test_image_with_text_keeps_the_text(self):
        engine, spy = self._engine_with_spy()
        engine.handle(IncomingMessage(user_id="1", text="这是什么", is_private=True,
                                      user_name="x",
                                      images=[vision.ImageRef(blob=PNG)]))
        self.assertEqual(spy.user_contents(0)[-1], "这是什么")

    def test_consecutive_media_messages_are_all_non_empty(self):
        """真机就是连续几条纯媒体消息——每一条都不能是空的。"""
        engine, spy = self._engine_with_spy()
        for _ in range(3):
            engine.handle(IncomingMessage(user_id="1", text="", is_private=True,
                                          user_name="x",
                                          images=[vision.ImageRef(blob=PNG)]))
        self.assertEqual(len(spy.calls), 3)
        for i in range(3):
            self.assertTrue(spy.user_contents(i)[-1].strip(),
                            "第 %d 轮的 user 消息是空的" % (i + 1))


if __name__ == "__main__":
    unittest.main()
