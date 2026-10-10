"""V5 第一批功能测试：模式会话隔离、群配置扩展、群欢迎/告别、公告。

全部离线：只测业务逻辑，不发任何网络请求。
"""
from __future__ import annotations

import time
import unittest
import uuid

from aliceqq.config import AppConfig, GroupSettings, PROJECT_ROOT
from aliceqq.db import SCOPE_PRIVATE, Storage
from aliceqq.engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT
from tests.test_engine import group, make_engine, private


def _store():
    tmp = PROJECT_ROOT / "data" / "test-run" / f"v5-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=True)
    return Storage(tmp / "alice.db")


def _engine(**overrides) -> Engine:
    return make_engine(**overrides)


def _msg(user="10001", group_id="", private_chat=True, text="你好", gid=None):
    return IncomingMessage(user_id=user, text=text, is_private=private_chat,
                           group_id="" if private_chat else (gid or group_id or "88888"),
                           user_name="小星", mentioned=not private_chat)


# ======================================================================
# 一、模式按会话隔离
# ======================================================================
class SessionModeTest(unittest.TestCase):
    """V5：模式只影响当前会话。

    这是本轮最重要的一处行为变更。原先是按用户全局存的，
    在 A 群切成助手后，B 群和私聊都会跟着变——同一个用户在两个群
    必须能是不同模式（V5 强调群隔离）。
    """

    def test_switch_in_one_group_does_not_leak_to_another(self):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        out = eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                         group_id="100", user_name="小星"))
        self.assertEqual(out.mode, MODE_ASSISTANT)
        # A 群：助手
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="100")), MODE_ASSISTANT)
        # B 群：仍然是默认的爱丽丝
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="200")), MODE_ALICE)

    def test_group_switch_does_not_affect_private(self):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                   group_id="100", user_name="小星"))
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x",
                                                          is_private=True)), MODE_ALICE)

    def test_private_switch_does_not_retroactively_change_existing_group(self):
        """私聊切模式只改"全局默认"，不改已经待过的群的既有模式。

        语义：私聊是用户自己的主场景，在那里选一次等于给**新会话**定默认；
        但对已经有记录的群（他已经在里面聊过、模式已定）不能追溯修改。
        """
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        # 先在群里留下痕迹（模式已定为爱丽丝）
        eng.storage.set_mode("u1", MODE_ALICE, scope="100")
        eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=True, user_name="小星"))
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="100")), MODE_ALICE)

    def test_private_switch_seeds_brand_new_group(self):
        """没去过的新群沿用在私聊选过的模式。"""
        eng = _engine()
        eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=True, user_name="小星"))
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="从未去过的群")), MODE_ASSISTANT)

    def test_switch_reply_states_the_scope(self):
        """回复里要说明只对当前会话生效，否则用户会以为全局都切了。"""
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        out = eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                         group_id="100", user_name="小星"))
        self.assertIn("本群", out.reply)
        eng2 = _engine()
        out2 = eng2.handle(IncomingMessage(user_id="u1", text="/助手", is_private=True,
                                           user_name="小星"))
        self.assertIn("私聊", out2.reply)

    def test_new_group_inherits_global_default(self):
        """用户第一次去的新群要沿用它最近一次选的模式，不能突然跳回爱丽丝。"""
        eng = _engine()
        eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=True, user_name="小星"))
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="brand-new")), MODE_ASSISTANT)

    def test_two_users_in_same_group_are_independent(self):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                   group_id="100", user_name="甲"))
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="100")), MODE_ASSISTANT)
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u2", text="x", is_private=False,
                                                          group_id="100")), MODE_ALICE)

    def test_migration_carries_old_global_mode_into_sessions(self):
        """老库升级：user_profile.mode 要灌进 session_mode，否则模式全丢。"""
        store = _store()
        store.ensure_profile("old-user", "老王", default_mode="alice")
        # 模拟升级前的老数据：user_profile 里是助手，session_mode 里没有记录
        with store._cursor() as cur:
            cur.execute("UPDATE user_profile SET mode = ? WHERE user_id = ?",
                        (MODE_ASSISTANT, "old-user"))
        self.assertEqual(store.get_session_mode("old-user", SCOPE_PRIVATE), "")
        migrated = store.migrate_profile_modes_to_sessions()
        self.assertGreaterEqual(migrated, 1)
        self.assertEqual(store.get_session_mode("old-user", SCOPE_PRIVATE), MODE_ASSISTANT)

    def test_migration_does_not_overwrite_existing_session_records(self):
        store = _store()
        store.set_mode("u", MODE_ASSISTANT, scope="100")     # 升级后已经切过
        store.set_mode("u", MODE_ALICE, scope=SCOPE_PRIVATE)
        store.migrate_profile_modes_to_sessions()
        self.assertEqual(store.get_session_mode("u", "100"), MODE_ASSISTANT)

    def test_reset_session_mode_falls_back_to_global(self):
        store = _store()
        store.set_mode("u", MODE_ALICE, scope="100")
        store.reset_session_mode("u", "100")
        self.assertEqual(store.get_session_mode("u", "100"), "")

    def test_scope_default_is_private(self):
        """set_mode 不传 scope 时按私聊处理（保持旧调用可用）。"""
        store = _store()
        store.set_mode("u", MODE_ASSISTANT)
        self.assertEqual(store.get_session_mode("u", SCOPE_PRIVATE), MODE_ASSISTANT)


# ======================================================================
# 二、群级模式开关
# ======================================================================
class GroupModeSwitchTest(unittest.TestCase):
    def _engine_with_settings(self, **settings):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        s = GroupSettings()
        for key, value in settings.items():
            ok, why = s.apply(key, value)
            assert ok, why
        eng.storage.save_group_settings("100", s)
        return eng

    def test_group_can_forbid_assistant_mode(self):
        eng = self._engine_with_settings(助手="关")
        out = eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                         group_id="100", user_name="小星"))
        self.assertIn("已关闭", out.reply)
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="100")), MODE_ALICE)

    def test_group_can_forbid_alice_mode(self):
        eng = self._engine_with_settings(爱丽丝="关")
        out = eng.handle(IncomingMessage(user_id="u1", text="/爱丽丝", is_private=False,
                                         group_id="100", user_name="小星"))
        self.assertIn("已关闭", out.reply)

    def test_forbidden_mode_falls_back_even_if_previously_set(self):
        """即使本会话之前设过助手模式，群关掉后也要退回爱丽丝。"""
        eng = self._engine_with_settings(助手="关")
        eng.storage.set_mode("u1", MODE_ASSISTANT, scope="100")
        self.assertEqual(eng.current_mode(IncomingMessage(user_id="u1", text="x", is_private=False,
                                                          group_id="100")), MODE_ALICE)

    def test_private_chat_is_not_affected_by_group_switch(self):
        eng = self._engine_with_settings(助手="关")
        out = eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=True,
                                         user_name="小星"))
        self.assertEqual(out.mode, MODE_ASSISTANT)

    def test_other_group_is_not_affected(self):
        eng = self._engine_with_settings(助手="关")
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        out = eng.handle(IncomingMessage(user_id="u1", text="/助手", is_private=False,
                                         group_id="200", user_name="小星"))
        self.assertEqual(out.mode, MODE_ASSISTANT)


# ======================================================================
# 三、群配置扩展
# ======================================================================
class GroupSettingsTest(unittest.TestCase):
    def test_new_keys_are_editable(self):
        s = GroupSettings()
        for key, expected_field in (("爱丽丝", "allow_alice"), ("助手", "allow_assistant"),
                                    ("欢迎语", "welcome"), ("告别语", "farewell"),
                                    ("公告", "notice")):
            self.assertIn(key, GroupSettings.EDITABLE, key)
            self.assertEqual(GroupSettings.EDITABLE[key][0], expected_field)

    def test_bool_keys_accept_chinese_switches(self):
        s = GroupSettings()
        self.assertTrue(s.apply("助手", "关")[0])
        self.assertFalse(s.allow_assistant)
        self.assertTrue(s.apply("助手", "开")[0])
        self.assertTrue(s.allow_assistant)

    def test_text_keys_keep_spaces_and_punctuation(self):
        """欢迎语是一句话，不能按数值解析、也不能丢掉内部空格。"""
        s = GroupSettings()
        ok, _ = s.apply("欢迎语", "欢迎 {name} 加入本群！有问题随时问 ~")
        self.assertTrue(ok)
        self.assertEqual(s.welcome, "欢迎 {name} 加入本群！有问题随时问 ~")

    def test_text_keys_can_be_cleared(self):
        s = GroupSettings()
        s.apply("公告", "本群禁止刷屏")
        self.assertEqual(s.notice, "本群禁止刷屏")
        for off in ("关", "清空", "无"):
            s.apply("公告", off)
            self.assertIsNone(s.notice, off)

    def test_text_key_length_is_capped(self):
        s = GroupSettings()
        ok, why = s.apply("公告", "长" * 300)
        self.assertFalse(ok)
        self.assertIn("太长", why)

    def test_bool_key_rejects_garbage(self):
        s = GroupSettings()
        ok, why = s.apply("助手", "也许吧")
        self.assertFalse(ok)
        self.assertIn("开关值", why)

    def test_to_extra_exports_none_so_reset_works(self):
        """to_extra 必须带上 None，否则「恢复默认」在库里残留旧值。"""
        s = GroupSettings()
        s.apply("助手", "关")
        extra = s.to_extra()
        self.assertIn("allow_assistant", extra)
        self.assertFalse(extra["allow_assistant"])
        s.apply("助手", "默认")
        self.assertIsNone(s.to_extra()["allow_assistant"])

    def test_all_fields_survive_round_trip(self):
        s = GroupSettings()
        s.apply("欢迎语", "hi {name}")
        s.apply("公告", "公告内容")
        s.apply("助手", "关")
        restored = GroupSettings.from_extra(s.to_extra())
        self.assertEqual(restored.welcome, "hi {name}")
        self.assertEqual(restored.notice, "公告内容")
        self.assertFalse(restored.allow_assistant)

    def test_describe_shows_the_new_items(self):
        s = GroupSettings()
        text = s.describe(AppConfig())
        for label in ("爱丽丝模式", "助手模式", "入群欢迎语", "退群告别语", "群公告"):
            self.assertIn(label, text)


# ======================================================================
# 四、群欢迎 / 告别
# ======================================================================
class MemberEventTest(unittest.TestCase):
    def _engine_with(self, **settings):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        s = GroupSettings()
        for key, value in settings.items():
            s.apply(key, value)
        eng.storage.save_group_settings("100", s)
        return eng

    def test_no_config_means_no_message(self):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True)
        self.assertIsNone(eng.member_event("100", "9", "小明", "approve").reply)

    def test_welcome_substitutes_name_and_group(self):
        eng = self._engine_with(欢迎语="欢迎 {name} 加入本群（{group}）")
        out = eng.member_event("100", "9", "小明", "approve")
        self.assertEqual(out.reply, "欢迎 小明 加入本群（100）")
        self.assertEqual(out.decision, "member_event")

    def test_missing_name_uses_fallback(self):
        eng = self._engine_with(欢迎语="欢迎 {name}～")
        self.assertIn("新成员", eng.member_event("100", "9", "", "approve").reply)

    def test_braces_in_template_do_not_crash(self):
        """用 replace 而不是 str.format：用户写个表情符号不该抛异常。"""
        eng = self._engine_with(欢迎语="欢迎 {name} ⊂(・ω・)⊃ {未知变量}")
        out = eng.member_event("100", "9", "小明", "approve")
        self.assertIn("小明", out.reply)
        self.assertIn("{未知变量}", out.reply)

    def test_farewell_only_on_leave(self):
        eng = self._engine_with(告别语="{name} 再见")
        self.assertIn("再见", eng.member_event("100", "9", "小明", "leave").reply)
        # 被踢出时不发告别：那是冲突场景，发"再见"像嘲讽
        self.assertIsNone(eng.member_event("100", "9", "小明", "kick").reply)

    def test_disabled_group_sends_nothing(self):
        eng = self._engine_with(欢迎语="欢迎 {name}")
        eng.storage.upsert_group("100", enabled=False)
        self.assertIsNone(eng.member_event("100", "9", "小明", "approve").reply)

    def test_no_group_id_is_silent(self):
        eng = self._engine_with(欢迎语="欢迎 {name}")
        self.assertIsNone(eng.member_event("", "9", "小明", "approve").reply)

    def test_welcome_is_length_capped(self):
        eng = self._engine_with(欢迎语="欢" * 180 + " {name}")
        out = eng.member_event("100", "9", "小明", "approve")
        self.assertLessEqual(len(out.reply), 200)

    def test_welcome_does_not_leak_other_users_memory(self):
        """欢迎语里绝不能出现任何用户记忆——那是最容易出的隐私事故。"""
        eng = self._engine_with(欢迎语="欢迎 {name}")
        eng.storage.add_memory("9", "家住某小区 3 号楼", source="manual")
        out = eng.member_event("100", "9", "小明", "approve")
        self.assertNotIn("小区", out.reply)

    def test_other_group_settings_do_not_apply(self):
        eng = self._engine_with(欢迎语="欢迎 {name}")
        eng.storage.upsert_group("200", enabled=True)
        self.assertIsNone(eng.member_event("200", "9", "小明", "approve").reply)


# ======================================================================
# 五、公告命令
# ======================================================================
class NoticeCommandTest(unittest.TestCase):
    def _admin_engine(self):
        eng = _engine()
        eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
        # 把 owner 认成 10001，避免依赖 NoneBot 的群管理角色记录
        eng.config.bot.owner_ids = ["10001"]
        return eng

    def _send(self, eng, text, user="10001"):
        return eng.handle(IncomingMessage(user_id=user, text=text, is_private=False,
                                          group_id="100", user_name="小星", mentioned=True,
                                          is_command=text.startswith("/")))

    def test_admin_can_set_and_everyone_can_read(self):
        eng = self._admin_engine()
        out = self._send(eng, "/公告 每周日 20:00 群内活动")
        self.assertIn("已更新", out.reply)
        read = self._send(eng, "/公告", user="30003")
        self.assertIn("每周日 20:00 群内活动", read.reply)

    def test_read_when_not_set(self):
        eng = self._admin_engine()
        self.assertIn("还没有设置公告", self._send(eng, "/公告", user="30003").reply)

    def test_non_admin_cannot_set(self):
        eng = self._admin_engine()
        out = self._send(eng, "/公告 我要打广告", user="30003")
        self.assertNotIn("已更新", out.reply)
        self.assertIn("权限", out.reply)
        self.assertIn("还没有设置公告", self._send(eng, "/公告").reply)

    def test_admin_can_clear(self):
        eng = self._admin_engine()
        self._send(eng, "/公告 临时公告")
        self.assertIn("已清空", self._send(eng, "/公告 关").reply)
        self.assertIn("还没有设置公告", self._send(eng, "/公告").reply)

    def test_notice_is_group_scoped(self):
        eng = self._admin_engine()
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        self._send(eng, "/公告 只属于 100 群")
        other = eng.handle(IncomingMessage(user_id="10001", text="/公告", is_private=False,
                                           group_id="200", user_name="小星", mentioned=True,
                                           is_command=True))
        self.assertIn("还没有设置公告", other.reply)
        self.assertNotIn("只属于 100 群", other.reply)

    def test_private_chat_rejected(self):
        eng = self._admin_engine()
        out = eng.handle(IncomingMessage(user_id="10001", text="/公告", is_private=True,
                                         user_name="小星", is_command=True))
        self.assertIn("群里", out.reply)

    def test_alias_works(self):
        eng = self._admin_engine()
        self.assertIn("已更新", self._send(eng, "/notice 英文别名").reply)

    def test_notice_appears_in_group_config_view(self):
        eng = self._admin_engine()
        self._send(eng, "/公告 大家看这里")
        view = self._send(eng, "/群配置", user="30003")
        self.assertIn("大家看这里", view.reply)


# ======================================================================
# 六、帮助文案
# ======================================================================
class HelpTextTest(unittest.TestCase):
    def test_help_lists_notice_command(self):
        self.assertIn("/公告", _engine().help_text())

    def test_help_lists_group_mode_switches(self):
        text = _engine().help_text()
        self.assertIn("/群设置", text)


class PromptTemplateTest(unittest.TestCase):
    """V5 提示词：文件存在、变量全部能渲染、模式标记还在。"""

    def test_alice_template_renders_without_leftovers(self):
        import re
        eng = _engine()
        system, _, _ = eng.build_messages(
            IncomingMessage(user_id="10001", text="hi", is_private=True, user_name="小星"),
            MODE_ALICE)
        self.assertEqual(re.findall(r"\{\{[a-z_]+\}\}", system), [])
        self.assertIn("MODE-MARKER:ALICE", system)

    def test_assistant_template_renders_without_leftovers(self):
        import re
        eng = _engine()
        system, _, _ = eng.build_messages(
            IncomingMessage(user_id="10001", text="hi", is_private=True, user_name="小星"),
            MODE_ASSISTANT)
        self.assertEqual(re.findall(r"\{\{[a-z_]+\}\}", system), [])
        self.assertIn("MODE-MARKER:ASSISTANT", system)

    def test_v5_sections_present(self):
        eng = _engine()
        system, _, _ = eng.build_messages(
            IncomingMessage(user_id="10001", text="hi", is_private=True, user_name="小星"),
            MODE_ALICE)
        for title in ("模式系统", "角色行为", "知识库与联网搜索", "记忆和隐私", "群聊互动",
                      "群管理和权限", "多媒体", "主动消息", "公开服务与额度", "最终回复要求"):
            self.assertIn(title, system, title)

    def test_v5_keeps_hard_privacy_rules(self):
        eng = _engine()
        system, _, _ = eng.build_messages(
            IncomingMessage(user_id="10001", text="hi", is_private=True, user_name="小星"),
            MODE_ALICE)
        self.assertIn("不向群管理员展示普通用户的私聊记忆", system)
        self.assertIn("不得因为用户批评机器人而处罚用户", system)
        self.assertIn("不得伪造", system)

    def test_alice_template_keeps_language_style_guide(self):
        """V5 正文没写语体细节，但那是调试出来的关键约束，不能丢。"""
        eng = _engine()
        system, _, _ = eng.build_messages(
            IncomingMessage(user_id="10001", text="hi", is_private=True, user_name="小星"),
            MODE_ALICE)
        self.assertIn("第三人称自我叙述", system)
        self.assertIn("爱丽丝知道！", system)


if __name__ == "__main__":
    unittest.main(verbosity=2)
