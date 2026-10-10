"""V3 公开运营与权限控制测试。

覆盖：群级功能开关、按能力的独立额度、群间隔离、隐私边界、运营数据不含内容。
"""
from __future__ import annotations

import time
import unittest

from aliceqq.config import GroupSettings
from aliceqq.engine import IncomingMessage, MODE_ALICE
from tests.test_engine import group, make_engine, minute_safe_base, private


class GroupSettingsTest(unittest.TestCase):
    def test_group_admin_can_set_and_read_flags(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["99999"]
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20001"]})
        _, outcome = group("/群设置 搜索 关", user="20001", gid="88888", engine=engine)
        self.assertIn("已更新", outcome.reply)
        self.assertIn("联网检索：关", outcome.reply)
        # 落库并能被其它路径读到
        settings = engine.storage.get_group_settings("88888")
        self.assertFalse(settings.search)

    def test_ordinary_member_cannot_change_settings(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["99999"]
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        _, outcome = group("/群设置 搜索 关", user="30003", gid="88888", engine=engine)
        self.assertIn("OWNER_IDS", outcome.reply)
        self.assertIsNone(engine.storage.get_group_settings("88888").search,
                          "无权限者不得改动设置")

    def test_everyone_can_view_group_config(self):
        engine = make_engine()
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        _, outcome = group("/群配置", user="30003", gid="88888", engine=engine)
        self.assertIn("本群配置", outcome.reply)
        self.assertIn("联网检索", outcome.reply)

    def test_invalid_value_is_rejected_with_usage(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["20001"]
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20001"]})
        _, outcome = group("/群设置 回复概率 9", user="20001", gid="88888", engine=engine)
        self.assertIn("0~1", outcome.reply)
        _, unknown = group("/群设置 不存在的项目 1", user="20001", gid="88888", engine=engine)
        self.assertIn("不支持设置", unknown.reply)

    def test_reset_to_global(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["20001"]
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20001"]})
        group("/群设置 搜索 关", user="20001", gid="88888", engine=engine)
        _, outcome = group("/群设置 搜索 默认", user="20001", gid="88888", engine=engine)
        self.assertIn("跟随全局", outcome.reply)
        self.assertIsNone(engine.storage.get_group_settings("88888").search)

    def test_settings_do_not_leak_across_groups(self):
        # V3 硬性要求：一个群的专属设定不得泄露到另一个群
        engine = make_engine()
        engine.config.bot.owner_ids = ["20001"]
        # 全局开启搜索，这样"群 A 覆盖为关 / 群 B 跟随全局为开"才有区分度
        # （测试夹具默认关掉联网以免联网，若不开，两个群都会显示"关"，断言就失去意义）
        engine.config.features.web_search = True
        engine.config.search.enabled = True
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20001"]})
        engine.storage.upsert_group("99999", enabled=True,
                                    extra={"approved": True, "admins": ["20001"]})
        group("/群设置 搜索 关", user="20001", gid="88888", engine=engine)
        group("/群设置 群额度 5", user="20001", gid="88888", engine=engine)

        other = engine.storage.get_group_settings("99999")
        self.assertIsNone(other.search, "群 A 的搜索开关不应影响群 B")
        self.assertIsNone(other.quota_daily, "群 A 的额度不应影响群 B")
        # 渲染到提示词里的群配置也要按群区分
        _, _, variables_a = engine.build_messages(
            IncomingMessage(user_id="20001", text="在吗", is_private=False, group_id="88888",
                            user_name="甲"), MODE_ALICE)
        _, _, variables_b = engine.build_messages(
            IncomingMessage(user_id="20001", text="在吗", is_private=False, group_id="99999",
                            user_name="甲"), MODE_ALICE)
        # 群 A：被显式覆盖为关；群 B：跟随全局为开
        self.assertIn("联网检索=关", variables_a["group_config"])
        self.assertIn("联网检索=开", variables_b["group_config"])
        self.assertIn("本群每日额度=5", variables_a["group_config"])
        self.assertNotIn("本群每日额度", variables_b["group_config"])


class FeatureToggleTest(unittest.TestCase):
    def test_group_search_disabled_blocks_search(self):
        engine = make_engine()
        engine.config.search.enabled = True
        engine.config.features.web_search = True
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        settings = engine.storage.get_group_settings("88888")
        settings.search = False
        engine.storage.save_group_settings("88888", settings)

        called = {"n": 0}

        class Spy:
            name = "spy"

            def search(self, query, limit=5):
                called["n"] += 1
                from aliceqq.search import SearchOutcome
                return SearchOutcome(True, query, "spy", results=[])

            def format_for_model(self, outcome, max_results=4):
                return "（无结果）"

        engine.search_service = Spy()
        group("今天有什么新活动", user="20001", gid="88888", engine=engine, mentioned=True,
              at=minute_safe_base())
        self.assertEqual(called["n"], 0, "群已关闭检索时不该发起检索请求")

    def test_private_chat_ignores_group_toggle(self):
        # 群级开关只作用于那个群，不影响私聊
        engine = make_engine()
        engine.config.search.enabled = True
        engine.config.features.web_search = True
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        settings = engine.storage.get_group_settings("88888")
        settings.search = False
        engine.storage.save_group_settings("88888", settings)

        called = {"n": 0}

        class Spy:
            name = "spy"

            def search(self, query, limit=5):
                called["n"] += 1
                from aliceqq.search import SearchOutcome, SearchResult
                return SearchOutcome(True, query, "spy", results=[
                    SearchResult("蔚蓝档案最新活动", "https://example.com/a", "内容",
                                 "example.com", True)])

            def format_for_model(self, outcome, max_results=4):
                return "（spy 结果）"

        engine.search_service = Spy()
        private("蔚蓝档案今天有什么新活动", engine=engine)
        self.assertEqual(called["n"], 1, "私聊不受群的开关影响")


class FeatureQuotaTest(unittest.TestCase):
    def _engine_with_caps(self, cap_private=2, cap_group=2):
        engine = make_engine()
        engine.config.search.enabled = True
        engine.config.features.web_search = True
        engine.config.quota.daily_search_private = cap_private
        engine.config.quota.daily_search_group = cap_group
        engine.config.quota.daily_search_global = 1000
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        return engine

    def _spy(self, engine, calls):
        from aliceqq.search import SearchOutcome, SearchResult

        class Spy:
            name = "spy"

            def search(self, query, limit=5):
                calls.append(query)
                return SearchOutcome(True, query, "spy", results=[
                    SearchResult(f"{query} 相关结果", "https://example.com/a", "摘要",
                                 "example.com", True)])

            def format_for_model(self, outcome, max_results=4):
                return "（spy 结果）"

        engine.search_service = Spy()

    def test_search_quota_is_enforced_per_user(self):
        engine = self._engine_with_caps(cap_private=2)
        calls = []
        self._spy(engine, calls)
        base = minute_safe_base()
        for i in range(2):
            private("蔚蓝档案今天有什么活动", engine=engine, at=base + i * 120)
        self.assertEqual(len(calls), 2, "两次检索应被允许")
        _, outcome = private("蔚蓝档案今天有什么活动", engine=engine, at=base + 300)
        self.assertEqual(len(calls), 2, "额度用尽后不该再发起检索")
        self.assertIn("额度", outcome.debug.get("search_skipped", ""))

    def test_search_quota_private_and_group_are_separate(self):
        # V3：群额度与个人额度必须分别判断
        engine = self._engine_with_caps(cap_private=1, cap_group=1)
        calls = []
        self._spy(engine, calls)
        base = minute_safe_base()
        private("蔚蓝档案今天有什么活动", user="20001", engine=engine, at=base)
        group("蔚蓝档案今天有什么活动", user="20002", gid="88888", engine=engine,
              mentioned=True, at=base + 120)
        self.assertEqual(len(calls), 2, "私聊额度用尽不应影响群额度")
        # 再各来一次，都应被额度挡住
        private("蔚蓝档案今天有什么活动", user="20001", engine=engine, at=base + 240)
        group("蔚蓝档案今天有什么活动", user="20002", gid="88888", engine=engine,
              mentioned=True, at=base + 360)
        self.assertEqual(len(calls), 2)

    def test_quota_exhausted_does_not_fake_success(self):
        engine = self._engine_with_caps(cap_private=1)
        calls = []
        self._spy(engine, calls)
        base = minute_safe_base()
        private("蔚蓝档案今天有什么活动", engine=engine, at=base)
        _, outcome = private("蔚蓝档案今天有什么活动", engine=engine, at=base + 120)
        # 不应声称已经检索过
        self.assertNotIn("检索到", outcome.prompt_system or "")
        self.assertIn("额度", outcome.debug.get("search_skipped", ""))


class PrivacyBoundaryTest(unittest.TestCase):
    def test_group_admin_cannot_see_other_users_memory(self):
        engine = make_engine()
        # A 在私聊留下记忆
        private("记住我住在火星基地第七区", user="10001", engine=engine)
        # B 是群管理员，在群里问"你记得我吗"
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20002"]})
        _, outcome = group("你记得我吗", user="20002", gid="88888", engine=engine)
        self.assertNotIn("火星基地", outcome.prompt_system or "")
        self.assertNotIn("火星基地", outcome.reply or "")

    def test_group_admin_memory_view_shows_only_own(self):
        engine = make_engine()
        private("记住我喜欢猫", user="10001", engine=engine)
        engine.storage.upsert_group("88888", enabled=True,
                                    extra={"approved": True, "admins": ["20002"]})
        _, outcome = group("/记忆", user="20002", gid="88888", engine=engine)
        self.assertNotIn("猫", outcome.reply)

    def test_ops_report_contains_no_content(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["10001"]
        private("记住我住在火星基地第七区", user="10001", engine=engine)
        _, outcome = private("/运营", user="10001", engine=engine)
        self.assertIn("运营数据", outcome.reply)
        self.assertIn("用户数", outcome.reply)
        # 运营报表绝不能带出用户内容
        self.assertNotIn("火星基地", outcome.reply)
        self.assertNotIn("我喜欢", outcome.reply)

    def test_non_owner_cannot_view_ops(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["99999"]
        _, outcome = private("/运营", user="10001", engine=engine)
        self.assertIn("所有者", outcome.reply)

    def test_user_cannot_learn_about_other_users(self):
        engine = make_engine()
        private("你好", user="10001", engine=engine)
        # 另一个用户问"有别人在用吗" —— 系统层面不注入任何此类信息
        _, outcome = private("有别人也在用你吗", user="20002", engine=engine)
        self.assertNotIn("10001", outcome.prompt_system or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
