"""V4 多媒体与主动互动测试。

覆盖：素材库、媒体决策（话题匹配/限频/概率）、语音仅在明确要求时、
生成失败如实说明、AI 生成图来源标记、群聊主动话题门槛、素材指令权限。
全部离线。
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
import unittest
from unittest import mock

from aliceqq import proactive as proactive_mod
from aliceqq.config import AppConfig
from aliceqq.engine import IncomingMessage, MODE_ALICE
from aliceqq.media import MediaLibrary, normalize_tags
from aliceqq.proactive import ProactiveTask
from tests.test_engine import group, make_engine, minute_safe_base, private


def _media_dir() -> str:
    """在**工作区内**建临时素材目录。

    不能用 tempfile.mkdtemp：沙箱下系统临时目录不可写，会 PermissionError。
    """
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "test-media")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "case-" + uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    return path


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


class LibraryTestBase(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.library = MediaLibrary(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class MediaLibraryTest(LibraryTestBase):
    def test_add_load_roundtrip(self):
        item = self.library.add("sticker", "yawn.png", PNG, tags="困,累,睡觉",
                                desc="打哈欠")
        self.assertTrue(self.library.exists(item))
        reloaded = MediaLibrary(self.root)
        reloaded.load()
        self.assertEqual(len(reloaded.all()), 1)
        self.assertEqual(reloaded.all()[0].tags, ["困", "累", "睡觉"])

    def test_rejects_bad_extension_and_oversize(self):
        with self.assertRaises(ValueError):
            self.library.add("sticker", "evil.exe", PNG, tags="x")
        with self.assertRaises(ValueError):
            self.library.add("sticker", "big.png", b"x" * (9 * 1024 * 1024), tags="x")

    def test_pick_requires_tag_match(self):
        self.library.add("sticker", "a.png", PNG, tags="困", desc="困")
        # 话题不匹配 -> 必须返回 None，绝不"随便挑一张"
        self.assertIsNone(self.library.pick("今天天气不错", kind="sticker"))
        self.assertIsNotNone(self.library.pick("好困啊", kind="sticker"))

    def test_pick_ignores_missing_files(self):
        item = self.library.add("sticker", "a.png", PNG, tags="困", desc="困")
        os.remove(self.library.path_of(item))
        self.assertIsNone(self.library.pick("好困啊", kind="sticker"))

    def test_remove(self):
        item = self.library.add("sticker", "a.png", PNG, tags="困")
        self.assertTrue(self.library.remove(item.id))
        self.assertEqual(self.library.all(), [])
        self.assertFalse(self.library.remove("不存在"))

    def test_catalog_lists_tags_not_paths(self):
        self.library.add("sticker", "a.png", PNG, tags="困,累", desc="打哈欠")
        text = self.library.catalog_text()
        self.assertIn("打哈欠", text)
        self.assertIn("困", text)
        # 不能把内部文件路径泄露进提示词
        self.assertNotIn(self.root, text)
        self.assertNotIn(".png", text)

    def test_catalog_marks_generated(self):
        # 生成图存为 kind=image + origin=generated。
        # 早期存成独立的 kind="generated" 会让它不出现在清单里，既看不见也选不中。
        self.library.add("image", "g.png", PNG, tags="猫", desc="生成的猫",
                         origin="generated")
        text = self.library.catalog_text()
        self.assertIn("AI 生成", text)
        self.assertIn("生成的猫", text)

    def test_empty_catalog_says_so(self):
        self.assertIn("没有可用素材", self.library.catalog_text())

    def test_normalize_tags(self):
        self.assertEqual(normalize_tags("困, 累；睡觉 打哈欠"), ["困", "累", "睡觉", "打哈欠"])
        self.assertEqual(normalize_tags(["A", "a", "B"]), ["a", "b"])


class MediaCQTest(unittest.TestCase):
    """CQ 码里的文件路径必须是绝对路径。

    实测踩过：素材库根目录是相对路径 "media" 时生成的是
    file:///media/sticker/x.png —— 会被解析成"主机名 media"，
    NapCat 找不到文件，媒体静默发不出去。
    """

    def test_cq_path_is_absolute(self):
        # 从**核心包**导入，而不是适配层。
        # 适配层整块包在 `if HAS_NONEBOT` 里，而且它的模块顶层会调用
        # `get_driver()`——没装 NoneBot 时导入直接抛
        # "NoneBot has not been initialized"，导致整个测试文件收集失败。
        # 逻辑搬到 aliceqq.media 之后，这几条用例在零依赖环境也能跑。
        from aliceqq.media import cq_code_for
        from aliceqq.engine import MediaAttachment

        root = _media_dir()
        try:
            library = MediaLibrary(root)
            item = library.add("sticker", "a.png", PNG, tags="困", desc="困")
            outcome = type("O", (), {})()
            outcome.media = [MediaAttachment(kind="sticker",
                                             path=str(library.path_of(item)))]
            cq = cq_code_for(outcome)
            self.assertIn("[CQ:image,file=", cq)
            body = cq.split("file=", 1)[1].rstrip("]").replace("%2C", ",")
            self.assertTrue(body.startswith("file:///"), body)
            local = body[len("file:///"):]
            self.assertTrue(os.path.isabs(local),
                            f"CQ 里的路径必须是绝对路径，实际：{body}")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_voice_uses_record_tag(self):
        from aliceqq.media import cq_code_for
        from aliceqq.engine import MediaAttachment

        root = _media_dir()
        try:
            library = MediaLibrary(root)
            item = library.add("voice", "v.mp3", b"ID3" + b"0" * 32, tags="问候",
                               desc="问候语")
            outcome = type("O", (), {})()
            outcome.media = [MediaAttachment(kind="voice", path=str(library.path_of(item)))]
            self.assertIn("[CQ:record,file=", cq_code_for(outcome))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_missing_file_is_skipped_not_fatal(self):
        from aliceqq.media import cq_code_for
        from aliceqq.engine import MediaAttachment

        outcome = type("O", (), {})()
        outcome.media = [MediaAttachment(kind="sticker", path="不存在的文件.png")]
        self.assertEqual(cq_code_for(outcome), "",
                         "文件不存在应跳过，不能因为坏路径让整条回复发不出去")


class MediaDecisionTest(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self.root)
        self.engine.config.media.enabled = True
        self.engine.config.media.sticker_min_interval = 0
        self.engine.config.media.sticker_probability = 1.0
        self.engine.config.features.emoji = True
        self._seed()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _seed(self):
        self.engine.media_library.add("sticker", "sleep.png", PNG,
                                      tags="困,累,睡觉", desc="打哈欠")
        self.engine.media_library.add("image", "ba.png", PNG,
                                      tags="蔚蓝档案,游戏", desc="蔚蓝档案图")

    def test_sticker_attached_on_tag_match(self):
        _, outcome = private("好困啊，昨天没睡好", engine=self.engine)
        self.assertEqual(len(outcome.media), 1)
        self.assertEqual(outcome.media[0].kind, "sticker")

    def test_sticker_not_attached_without_match(self):
        _, outcome = private("今天数学考试考得不错", engine=self.engine)
        self.assertEqual(outcome.media, [], "话题不匹配时不应发送任何图片")

    def test_media_disabled_globally(self):
        self.engine.config.features.emoji = False
        _, outcome = private("好困啊", engine=self.engine)
        self.assertEqual(outcome.media, [])

    def test_group_can_disable_media(self):
        self.engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        settings = self.engine.storage.get_group_settings("88888")
        settings.image = False
        self.engine.storage.save_group_settings("88888", settings)
        _, outcome = group("好困啊", user="20001", gid="88888", engine=self.engine,
                           mentioned=True, at=minute_safe_base())
        self.assertEqual(outcome.media, [], "群级关闭媒体后不应发送")

    def test_probability_zero_never_sends(self):
        self.engine.config.media.sticker_probability = 0.0
        for _ in range(5):
            _, outcome = private("好困啊", engine=self.engine)
            self.assertEqual(outcome.media, [])

    def test_rate_limit_blocks_second_sticker(self):
        self.engine.config.media.sticker_min_interval = 600
        base = minute_safe_base()
        _, first = private("好困啊", engine=self.engine, at=base)
        self.assertEqual(len(first.media), 1)
        _, second = private("还是好困", engine=self.engine, at=base + 60)
        self.assertEqual(second.media, [], "间隔未到时不应再发表情包")

    def test_at_most_one_media(self):
        _, outcome = private("好困啊，来张蔚蓝档案的图", engine=self.engine)
        self.assertLessEqual(len(outcome.media), 1)


class VoiceTest(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self.root)
        self.engine.config.media.enabled = True
        self.engine.config.media.sticker_min_interval = 0
        self.engine.config.media.sticker_probability = 0.0

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    class _FakeTTS:
        available = True

        def __init__(self, ok=True, error=""):
            self.ok = ok
            self.error = error
            self.calls = 0

        def synthesize(self, text):
            from aliceqq.generation import SpeechResult
            self.calls += 1
            if self.ok:
                return SpeechResult(True, data=b"AUDIO")
            return SpeechResult(False, error=self.error)

    def test_no_voice_without_explicit_request(self):
        self.engine.tts_client = self._FakeTTS()
        self.engine.config.features.voice = True
        _, outcome = private("今天过得怎么样", engine=self.engine)
        self.assertEqual(outcome.media, [])
        self.assertEqual(self.engine.tts_client.calls, 0, "没要求就不该合成语音")

    def test_voice_when_explicitly_requested(self):
        self.engine.tts_client = self._FakeTTS()
        self.engine.config.features.voice = True
        _, outcome = private("用语音说一下今天天气", engine=self.engine)
        self.assertEqual(len(outcome.media), 1)
        self.assertEqual(outcome.media[0].kind, "voice")

    def test_voice_failure_is_reported_not_faked(self):
        self.engine.tts_client = self._FakeTTS(ok=False, error="HTTP 401（Key 无效）")
        self.engine.config.features.voice = True
        _, outcome = private("用语音说句话", engine=self.engine)
        self.assertEqual(outcome.media, [])
        # V4：工具失败必须说明，不能假装成功
        self.assertIn("没能生成成功", outcome.reply)
        self.assertNotIn("已发送", outcome.reply)

    def test_voice_disabled_group_blocks_and_explains(self):
        self.engine.tts_client = self._FakeTTS()
        self.engine.config.features.voice = True
        self.engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        settings = self.engine.storage.get_group_settings("88888")
        settings.voice = False
        self.engine.storage.save_group_settings("88888", settings)
        _, outcome = group("用语音说句话", user="20001", gid="88888", engine=self.engine,
                           mentioned=True, at=minute_safe_base())
        self.assertEqual(outcome.media, [])
        self.assertIn("没能生成成功", outcome.reply)


class ImageGenerationTest(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self.root)
        self.engine.config.media.enabled = True
        self.engine.config.media.sticker_min_interval = 0
        self.engine.config.media.sticker_probability = 0.0

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    class _FakeImage:
        available = True

        def __init__(self, ok=True, error=""):
            self.ok = ok
            self.error = error
            self.calls = 0

        def generate(self, prompt):
            from aliceqq.generation import ImageResult
            self.calls += 1
            if self.ok:
                return ImageResult(True, data=PNG)
            return ImageResult(False, error=self.error)

    def test_prefers_library_over_generation(self):
        self.engine.media_library.add("image", "ba.png", PNG, tags="蔚蓝档案", desc="档案图")
        self.engine.image_client = self._FakeImage()
        self.engine.config.features.image = True
        _, outcome = private("给我来一张蔚蓝档案的图", engine=self.engine)
        self.assertEqual(len(outcome.media), 1)
        self.assertEqual(self.engine.image_client.calls, 0,
                         "有现成素材就不该调用生成（V4 优先级）")

    def test_generates_only_when_no_match(self):
        self.engine.image_client = self._FakeImage()
        self.engine.config.features.image = True
        _, outcome = private("画一张赛博朋克风格的爱丽丝", engine=self.engine)
        self.assertEqual(len(outcome.media), 1)
        self.assertEqual(self.engine.image_client.calls, 1)

    def test_generated_image_marked_generated(self):
        self.engine.image_client = self._FakeImage()
        self.engine.config.features.image = True
        _, outcome = private("画一张爱丽丝", engine=self.engine)
        self.assertEqual(outcome.media[0].origin, "generated")
        self.assertTrue(outcome.media[0].is_generated,
                        "AI 生成图必须标记来源，不能当作官方素材")

    def test_generation_failure_is_reported(self):
        self.engine.image_client = self._FakeImage(ok=False, error="HTTP 429（额度限制）")
        self.engine.config.features.image = True
        _, outcome = private("画一张爱丽丝", engine=self.engine)
        self.assertEqual(outcome.media, [])
        self.assertIn("没能生成成功", outcome.reply)

    def test_disabled_image_explains_quota_or_switch(self):
        self.engine.image_client = self._FakeImage()
        self.engine.config.features.image = False
        _, outcome = private("画一张爱丽丝", engine=self.engine)
        self.assertEqual(outcome.media, [])
        self.assertIn("没能生成成功", outcome.reply)
        self.assertIn("关闭", outcome.reply)


class MediaCatalogTest(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _catalog(self, text="在吗"):
        _, _, variables = self.engine.build_messages(
            IncomingMessage(user_id="1", text=text, is_private=True, user_name="x"),
            MODE_ALICE)
        return variables["media_catalog"]

    def test_reports_unavailable_capabilities_truthfully(self):
        self.engine.config.features.voice = False
        catalog = self._catalog()
        # 语音未启用 -> 必须明说不可用，模型才不会承诺发语音
        self.assertIn("语音：当前不可用", catalog)

    def test_reports_generation_not_configured(self):
        self.engine.config.features.image = True
        self.engine.image_client = None
        self.assertIn("未接入", self._catalog())

    def test_reports_configured_tts(self):
        class _T:
            available = True
        self.engine.config.features.voice = True
        self.engine.tts_client = _T()
        self.assertIn("已接入", self._catalog())


class GroupProactiveTest(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.storage = self.engine.storage
        self.config = self.engine.config
        self.config.features.proactive_group = True
        self.config.proactive.enabled = True
        self.config.proactive.frequency = "normal"
        self.config.proactive.quiet_hours = []
        self.config.proactive.group_min_idle_minutes = 60
        self.config.proactive.group_max_per_day = 1
        self.config.proactive.group_templates = ["来聊聊游戏吧", "有人想聊新番吗"]

    def _opt_in(self, gid, enabled=True):
        self.storage.upsert_group(gid, enabled=True, extra={"approved": True})
        settings = self.storage.get_group_settings(gid)
        settings.proactive = True
        self.storage.save_group_settings(gid, settings)

    def test_requires_explicit_group_opt_in(self):
        self.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        # 群主没显式开启（proactive 为 None）-> 不发言
        tasks = proactive_mod.plan_group_topics(self.config, self.storage, now=time.time())
        self.assertEqual(tasks, [])

    def test_speaks_when_group_opted_in_and_idle(self):
        self._opt_in("88888")
        now = time.time()
        self.storage.kv_set("bot_last_reply:88888", now - 7200)  # 2 小时前
        tasks = proactive_mod.plan_group_topics(self.config, self.storage, now=now)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].group_id, "88888")
        self.assertTrue(tasks[0].is_group)

    def test_respects_min_idle(self):
        self._opt_in("88888")
        now = time.time()
        self.storage.kv_set("bot_last_reply:88888", now - 60)  # 只安静 1 分钟
        self.assertEqual(proactive_mod.plan_group_topics(self.config, self.storage, now=now), [])

    def test_daily_cap(self):
        self._opt_in("88888")
        now = time.time()
        self.storage.kv_set("bot_last_reply:88888", now - 7200)
        tasks = proactive_mod.plan_group_topics(self.config, self.storage, now=now)
        self.assertEqual(len(tasks), 1)
        proactive_mod.mark_group_sent(self.storage, "88888", now)
        self.assertEqual(proactive_mod.plan_group_topics(self.config, self.storage, now=now), [],
                         "达到每日上限后不应再主动发言")

    def test_quiet_hours_block(self):
        self._opt_in("88888")
        now = time.time()
        hour = time.localtime(now).tm_hour
        self.config.proactive.quiet_hours = [hour]
        self.storage.kv_set("bot_last_reply:88888", now - 7200)
        self.assertEqual(proactive_mod.plan_group_topics(self.config, self.storage, now=now), [])

    def test_global_switch_off(self):
        self._opt_in("88888")
        self.config.features.proactive_group = False
        now = time.time()
        self.storage.kv_set("bot_last_reply:88888", now - 7200)
        self.assertEqual(proactive_mod.plan_group_topics(self.config, self.storage, now=now), [])

    def test_templates_rotate(self):
        self._opt_in("88888")
        now = time.time()
        self.storage.kv_set("bot_last_reply:88888", now - 7200)
        first = proactive_mod.plan_group_topics(self.config, self.storage, now=now)[0].text
        proactive_mod.mark_group_sent(self.storage, "88888", now)
        # 清掉当日计数，只验证模板轮换
        self.storage.kv_set(f"proactive_group_sent:88888:{proactive_mod._period_day(now)}", 0)
        second = proactive_mod.plan_group_topics(self.config, self.storage, now=now)[0].text
        self.assertNotEqual(first, second, "不应每次都发同一句")


class MediaCommandTest(unittest.TestCase):
    def setUp(self):
        self.root = _media_dir()
        self.engine = make_engine()
        self.engine.media_library = MediaLibrary(self.root)
        self.engine.config.bot.owner_ids = ["10001"]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_non_owner_cannot_manage(self):
        _, outcome = private("/素材列表", user="99999", engine=self.engine)
        self.assertIn("所有者", outcome.reply)

    def test_list_and_delete(self):
        item = self.engine.media_library.add("sticker", "a.png", PNG, tags="困", desc="困")
        _, listed = private("/素材列表", user="10001", engine=self.engine)
        self.assertIn(item.id, listed.reply)
        self.assertIn("困", listed.reply)
        _, deleted = private(f"/素材删除 {item.id}", user="10001", engine=self.engine)
        self.assertIn("已删除", deleted.reply)
        self.assertEqual(self.engine.media_library.all(), [])

    def test_upload_marks_pending(self):
        _, outcome = private("/素材上传 困,累｜打哈欠", user="10001", engine=self.engine)
        # 断言语义而不是固定措辞，避免改文案就挂测试
        self.assertIn("发过来", outcome.reply)
        pending = self.engine.storage.kv_get("media_pending:10001")
        self.assertEqual(pending["tags"], "困,累")
        self.assertEqual(pending["desc"], "打哈欠")

    def test_upload_with_tag_only(self):
        # 只给标签也能用，且不输出"说明「」"这种别扭的话（真机反馈）
        _, outcome = private("/素材上传 难受", user="10001", engine=self.engine)
        self.assertNotIn("说明「」", outcome.reply)
        self.assertIn("难受", outcome.reply)
        pending = self.engine.storage.kv_get("media_pending:10001")
        self.assertEqual(pending["tags"], "难受")
        self.assertEqual(pending["desc"], "")

    def test_upload_without_tag_is_rejected(self):
        _, outcome = private("/素材上传 ｜只有说明", user="10001", engine=self.engine)
        self.assertIn("标签不能为空", outcome.reply)

    def test_empty_library_says_no_media_sent(self):
        _, outcome = private("/素材列表", user="10001", engine=self.engine)
        self.assertIn("不会发送任何图片", outcome.reply)
        self.assertIn("不会声称已经发送", outcome.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
