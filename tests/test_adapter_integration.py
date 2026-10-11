# -*- coding: utf-8 -*-
"""适配层集成测试：真的跑一遍 `_handle`。

**这个文件补的是最后一层空白。**

`test_adapter_extract.py` 只能做静态检查（"抽了但忘了接线"），
因为它测的 `media_input` 是纯函数，而真正的事件处理 `_handle`
包在 `if HAS_NONEBOT:` 里——零依赖的 CI 里那段代码根本不存在。

`tests/_nonebot_stub.py` 提供了一个最小 NoneBot 桩，于是这里可以
**真的把事件喂给 `_handle`**，验证：

    QQ 事件 -> 抽取媒体 -> IncomingMessage -> 引擎 -> 调用模型

具体的观察点是：引擎收到的 `IncomingMessage` 里到底有没有图片/语音/视频，
以及有没有走到"发送回复"那一步（假 Bot 会记录它收到了什么）。

**它不替代真机验证。** 桩证明不了 NoneBot 真的会派发事件、
也证明不了 QQ 给的字段和文档一致。它只堵住"适配层自身接线错了"
这一类问题——而这类问题在真机上极难定位（图片那次就是这样：
抽取函数写对了，但没人验证它是否真的被调用）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tests import _nonebot_stub as stub
from tests.test_engine import make_engine

from aliceqq import asr as asr_mod
from aliceqq import video, vision
from aliceqq.config import VisionConfig
from aliceqq.engine import IncomingMessage

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WAV = b"RIFF" + b"\x00" * 100
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200


class _FakeASR:
    available = True

    def transcribe(self, audio, filename="audio.m4a"):
        return asr_mod.Transcription(True, text="艾利斯今天想去打游戏",
                                     provider="fake", seconds=2.0)


def _patch_media_read():
    """挡掉 `media.read_media_bytes`，返回按扩展名猜的假字节。

    **不挡的话每个带媒体的用例都要等 15.5 秒**：事件里的
    `https://cdn/...` 是假域名，`read_media_bytes` 会走完整套
    DNS 解析 + 20 秒超时 + 2 次重试。17 个用例跑了 89 秒，
    几乎全花在等这些超时上。

    返回的是**带正确魔数的假数据**，所以后面的解析路径
    （MIME 判断、解码）仍然走真实分支，只是不碰网络。
    """
    from aliceqq import media as media_mod

    original = media_mod.read_media_bytes

    def fake(source, timeout=20, retries=2):
        s = str(source or "").lower()
        if s.endswith((".jpg", ".jpeg")):
            return b"\xff\xd8\xff" + b"\x00" * 64
        if s.endswith(".png"):
            return PNG
        if s.endswith((".amr", ".m4a", ".mp3", ".ogg", ".wav")):
            return WAV
        if s.endswith((".mp4", ".mov", ".mkv")):
            return MP4
        return b""                    # 未知来源当成读不到，和真实行为一致

    media_mod.read_media_bytes = fake
    return lambda: setattr(media_mod, "read_media_bytes", original)


def _patch_vision(description="画面里有一只猫"):
    """把视觉接口换成假的，返回还原函数。

    **不 patch 的话测试会真的去请求视觉接口**：第一次跑这组用例时
    每个带图片的用例都在等 DNS 超时（8~17 秒），17 个用例跑了 137 秒。
    单元/集成测试必须离线、确定、快——这三条是同一个要求。
    """
    original = vision.net.open

    class _Resp:
        def read(self):
            return json.dumps({"choices": [
                {"message": {"content": description}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    vision.net.open = lambda request, timeout=None: _Resp()
    return lambda: setattr(vision.net, "open", original)


class HandlerHarness(unittest.TestCase):
    """装好桩、替换掉适配层的引擎，并把 _handle 跑起来。

    **适配层只导入一次（setUpClass）**，不要每个用例都 load_adapter()。
    第一版就是这么写的，结果 17 个用例跑了 80 秒——因为
    `load_adapter()` 每次都会清空 sys.modules 再重新导入一个 800 行的模块。
    handler 里引用的是模块全局的 `_engine`，所以每个用例只需要把
    `mod._engine` 换成自己的引擎，**不需要重新导入**。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = stub.load_adapter()
        assert "message" in stub.HANDLERS, "桩没有注册 message handler"
        # **静音适配层的日志**。用户 .env 里 ALICE_DEBUG=1 时，
        # 模块导入阶段就把 `_debug` 打开了，于是每个用例都会往
        # test runner 的输出里灌一大段事件原文和引擎决策，
        # 把真正的测试结果淹掉（实测就是这样）。
        # 这里只在测试期间替换成空实现，测完还原。
        cls._saved_logs = (cls.mod._log, cls.mod._debug_log)
        cls.mod._log = lambda message: None
        cls.mod._debug_log = lambda message: None

    @classmethod
    def tearDownClass(cls):
        cls.mod._log, cls.mod._debug_log = cls._saved_logs

    def setUp(self):
        self.handle = stub.HANDLERS["message"]
        self.engine = make_engine()
        # **必须用离线 mock**：make_engine 默认装配的是真实 LLM，
        # 集成测试跑起来会真的去请求 API（第一次跑就这么发生了）。
        self.engine.llm = make_engine().llm
        # 视觉接口也要挡住，否则每个带图片的用例都在等 DNS 超时
        self._restore_vision = _patch_vision()
        # 媒体下载同样要挡住：事件里的 cdn 域名是假的，不挡就是 15.5 秒/次
        self._restore_media = _patch_media_read()
        self._saved = (self.mod._engine, self.mod._config)
        self.mod._engine = self.engine
        # 让引擎真正用上本轮的视觉/语音能力
        self.engine.config.vision = VisionConfig(
            enabled=True, base_url="https://x/v1", api_key="k", model="m")
        self.engine.vision_client = vision.build_client(self.engine.config.vision)
        self.engine.asr_client = _FakeASR()
        self.engine.config.search.enabled = False
        self.engine.config.features.web_search = False
        self.bot = stub.Bot()

    def tearDown(self):
        self._restore_vision()
        self._restore_media()
        self.mod._engine, self.mod._config = self._saved

    def _run(self, event):
        return asyncio.run(self.handle(self.bot, event))

    def _capture_incoming(self):
        """记录下来引擎 handle 收到的 IncomingMessage（不改行为）。"""
        seen = []
        original = self.engine.handle

        def spy(message):
            seen.append(message)
            return original(message)

        self.engine.handle = spy
        return seen


class ImageEventTest(HandlerHarness):
    def test_image_segment_reaches_the_engine(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("image", url="https://cdn/a.jpg", file="a.jpg")])
        self._run(event)
        self.assertEqual(len(seen), 1, "引擎应当收到一条消息")
        self.assertTrue(seen[0].has_images, "图片段没有走到引擎")

    def test_image_only_message_reaches_the_engine(self):
        # 纯图片（不说话）是最容易被"空消息"挡掉的用法
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001", message=[stub._Segment("image", url="https://cdn/a.jpg")])
        self._run(event)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "")
        self.assertTrue(seen[0].has_images)

    def test_bot_sends_something_back(self):
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("text", text="看看这张图"),
                     stub._Segment("image", url="https://cdn/a.jpg")])
        self._run(event)
        self.assertTrue(self.bot.sent, "适配层没有发送任何回复")


class AudioEventTest(HandlerHarness):
    def test_record_segment_reaches_the_engine(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("record", url="https://cdn/v.amr",
                                   file="v.amr", duration="9")])
        self._run(event)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].has_audios, "语音段没有走到引擎")
        self.assertEqual(seen[0].audios[0].duration, 9.0)

    def test_voice_only_message_is_answered(self):
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("record", url="https://cdn/v.amr")])
        self._run(event)
        self.assertTrue(self.bot.sent, "纯语音消息没有被回复")


class VideoEventTest(HandlerHarness):
    def test_video_segment_reaches_the_engine(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("video", url="https://cdn/v.mp4",
                                   file="v.mp4", duration="12")])
        self._run(event)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].has_videos, "视频段没有走到引擎")
        self.assertEqual(seen[0].videos[0].duration, 12.0)


class MixedEventTest(HandlerHarness):
    def test_all_three_modalities_in_one_message(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("text", text="都看看"),
                     stub._Segment("image", url="https://cdn/a.jpg"),
                     stub._Segment("record", url="https://cdn/v.amr"),
                     stub._Segment("video", url="https://cdn/x.mp4")])
        self._run(event)
        self.assertEqual(len(seen), 1)
        msg = seen[0]
        self.assertTrue(msg.has_images)
        self.assertTrue(msg.has_audios)
        self.assertTrue(msg.has_videos)
        self.assertEqual(msg.text, "都看看")


class EventMetadataTest(HandlerHarness):
    def test_private_event_metadata(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(user_id="10001")
        self._run(event)
        self.assertTrue(seen[0].is_private)
        self.assertEqual(seen[0].user_id, "10001")

    def test_group_event_metadata(self):
        seen = self._capture_incoming()
        event = stub.GroupMessageEvent(
            group_id="88888", user_id="10002",
            message=[stub._Segment("text", text="大家好")])
        self._run(event)
        self.assertFalse(seen[0].is_private)
        self.assertEqual(seen[0].group_id, "88888")

    def test_self_message_is_ignored(self):
        # 机器人自己的消息不能回，否则自问自答
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(user_id="999", self_id="999",
                                         message=[stub._Segment("text", text="hi")])
        self._run(event)
        self.assertEqual(seen, [], "机器人自己的消息不该进引擎")

    def test_message_id_and_timestamp_are_carried(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(user_id="10001", message_id="42", time=1234.5)
        self._run(event)
        self.assertEqual(seen[0].message_id, "42")
        self.assertEqual(seen[0].timestamp, 1234.5)


class RobustnessTest(HandlerHarness):
    def test_text_only_message_still_works(self):
        # 加媒体抽取不能把最普通的纯文本路径弄坏
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001", message=[stub._Segment("text", text="你好")])
        self._run(event)
        self.assertEqual(seen[0].text, "你好")
        self.assertFalse(seen[0].has_media_input)

    def test_unknown_segment_type_is_ignored(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("text", text="hi"),
                     stub._Segment("json", data="{}"),
                     stub._Segment("face", faceIndex="1")])
        self._run(event)
        self.assertEqual(seen[0].text, "hi")
        self.assertFalse(seen[0].has_media_input)

    def test_empty_message_does_not_crash(self):
        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(user_id="10001", message=[])
        self._run(event)          # 不该抛异常
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "")

    def test_broken_segment_does_not_crash_the_handler(self):
        class Broken:
            type = "image"

            @property
            def data(self):
                raise RuntimeError("坏段")

        seen = self._capture_incoming()
        event = stub.PrivateMessageEvent(user_id="10001", message=[Broken()])
        self._run(event)          # 不该抛异常
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].has_images)


class AdapterImportTest(unittest.TestCase):
    """导入适配层本身不能炸——**这是 CI 抓到的第一个真问题**。

    CI 上四个 job 全部失败在「跑全部单元测试」，报的是：

        File ".../adapters/nonebot_plugin_alice.py", line 149
            _sing_python = sys.executable
        NameError: name 'sys' is not defined

    那个文件用了 `sys.executable` 但**没有 `import sys`**。

    **为什么本地永远测不出来**：那一行的上下文是

        _sing_python = (_config.env("SINGING_PYTHON")
                        or r"D:\\DSHWORK\\envs\\gptsovits\\Scripts\\python.exe")
        if not os.path.isfile(_sing_python):      # ← 本地这个文件**存在**
            _sing_python = sys.executable         # ← 所以这行不执行

    本机那个硬编码路径确实存在，所以 `if` 判定为假，`sys.executable`
    那一行**从来没被执行过**。CI（Ubuntu、Windows）上没有这个路径，
    于是走到那一行就炸。

    更糟的是：本地连 `if HAS_NONEBOT:` 整块都不执行（两套环境里
    只有 `.venv` 装了 nonebot，gptsovits 环境没有）。所以这个 bug
    需要"装了 NoneBot **且** 那个硬编码路径不存在"两个条件同时成立
    才会出现——本地恰好凑不齐。

    这组用例把两个条件都造出来。
    """

    def test_adapter_imports_when_default_singing_python_missing(self):
        """默认 SINGING_PYTHON 路径不存在时，导入不能炸。

        用假的 `os.path.isfile` 让那个硬编码路径判定为不存在，
        强制走到 `sys.executable` 那一行——就是 CI 上的真实情形。
        """
        import os as os_mod

        real_isfile = os_mod.path.isfile

        def fake_isfile(path):
            text = str(path)
            if text.endswith("python.exe") and "gptsovits" in text:
                return False
            return real_isfile(path)

        saved = os_mod.environ.pop("SINGING_PYTHON", None)
        os_mod.path.isfile = fake_isfile
        try:
            mod = stub.load_adapter()
        finally:
            os_mod.path.isfile = real_isfile
            if saved is not None:
                os_mod.environ["SINGING_PYTHON"] = saved

        import sys as sys_mod
        self.assertTrue(getattr(mod, "HAS_NONEBOT", False))
        self.assertEqual(getattr(mod, "_sing_python", None), sys_mod.executable,
                         "默认路径不存在时应当回退到当前解释器")

    def test_sys_is_imported_at_module_level(self):
        """`sys` 必须在模块级导入。

        静态断言，比"跑一遍看炸不炸"更直接：
        那一行依赖运行时条件，靠跑不一定覆盖得到。
        """
        import ast
        src = (ROOT / "aliceqq" / "adapters" / "nonebot_plugin_alice.py"
               ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in tree.body:                 # 只看模块顶层
            if isinstance(node, ast.Import):
                imported.update((a.asname or a.name).split(".")[0]
                                for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(a.asname or a.name for a in node.names)
        self.assertIn("sys", imported,
                      "适配层用了 sys.executable，必须在模块级 import sys")

    def test_no_other_bare_sys_usage_without_import(self):
        """防止同类问题复发：文件里出现 sys. 就必须有 import sys。"""
        src = (ROOT / "aliceqq" / "adapters" / "nonebot_plugin_alice.py"
               ).read_text(encoding="utf-8")
        if "sys." in src:
            self.assertIn("\nimport sys", src.replace("\r\n", "\n"),
                          "用了 sys. 但没有模块级 import sys")


class FullPathTest(HandlerHarness):
    """端到端：事件 -> 抽取 -> 引擎 -> 视觉/语音 -> 回复。"""

    def test_image_event_triggers_vision_and_reply(self):
        restore = _patch_vision("画面里有一只橘猫")
        try:
            event = stub.PrivateMessageEvent(
                user_id="10001",
                message=[stub._Segment("text", text="这是什么"),
                         stub._Segment("image", url="https://cdn/cat.jpg",
                                       file="cat.jpg")])
            self._run(event)
        finally:
            restore()
        self.assertTrue(self.bot.sent, "没有发出回复")
        sent = " ".join(str(m) for m in self.bot.sent)
        self.assertTrue(sent.strip(), "回复是空的")

    def test_voice_event_triggers_transcription_and_reply(self):
        event = stub.PrivateMessageEvent(
            user_id="10001",
            message=[stub._Segment("record", url="https://cdn/v.amr")])
        self._run(event)
        self.assertTrue(self.bot.sent, "纯语音消息没有发出回复")


if __name__ == "__main__":
    unittest.main()
