"""回归测试：三个实测中暴露的缺陷。

1. "第 2 次互动" off-by-one；
2. 指令文本被全角/空白/CQ 码污染后无法匹配；
3. /授权 的权限判定与失败提示。
"""
from __future__ import annotations

import time
import unittest

from aliceqq.engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT
from tests.test_engine import group, make_engine, private


class InteractionCounterTest(unittest.TestCase):
    def test_first_private_message_says_first_interaction(self):
        engine = make_engine()
        _, outcome = private("你好", engine=engine)
        self.assertIn("第 1 次互动", outcome.prompt_system)
        self.assertNotIn("第 2 次互动", outcome.prompt_system)

    def test_second_message_says_second_interaction(self):
        engine = make_engine()
        private("你好", engine=engine)
        _, second = private("在吗", engine=engine, at=time.time() + 120)
        self.assertIn("第 2 次互动", second.prompt_system)

    def test_fresh_user_never_sees_second_interaction(self):
        # 首次见面说"第 2 次"是实测中用户直接察觉到的 bug
        engine = make_engine()
        for uid in ("20001", "20002", "20003"):
            out = engine.handle(IncomingMessage(user_id=uid, text="你好", is_private=True,
                                                user_name="路人", timestamp=time.time()))
            self.assertIn("第 1 次互动", out.prompt_system)


class CommandNormalizationTest(unittest.TestCase):
    FULLWIDTH = "\uff0f"  # ／

    def test_fullwidth_slash_switches_mode(self):
        engine = make_engine()
        _, outcome = private(self.FULLWIDTH + "助手", engine=engine)
        self.assertEqual(outcome.mode, MODE_ASSISTANT)
        self.assertEqual(engine.storage.get_profile("10001").mode, MODE_ASSISTANT)

    def test_trailing_whitespace_switches_mode(self):
        engine = make_engine()
        _, outcome = private("  /助手  ", engine=engine)
        self.assertEqual(outcome.mode, MODE_ASSISTANT)

    def test_fullwidth_space_in_argument(self):
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        _, outcome = private("\uff0f忘记\u3000猫", engine=engine, at=time.time() + 60)
        self.assertEqual(engine.storage.count_memory("10001"), 0)
        self.assertIn("删除", outcome.reply)

    def test_cq_code_residue_is_stripped(self):
        engine = make_engine()
        _, outcome = private("[CQ:at,qq=12345] /助手", engine=engine)
        self.assertEqual(outcome.mode, MODE_ASSISTANT)

    def test_normal_message_is_not_treated_as_command(self):
        engine = make_engine()
        _, outcome = private("今天天气不错", engine=engine)
        self.assertEqual(outcome.decision, "replied")

    def test_fullwidth_slash_counts_as_command_for_group_trigger(self):
        # 指令触发是群聊里唯一不需要 @ 的触发方式之一
        engine = make_engine()
        _, outcome = group(self.FULLWIDTH + "状态", engine=engine, mentioned=False)
        self.assertTrue(outcome.should_send)
        self.assertEqual(outcome.decision, "command")


class AuthorizePermissionTest(unittest.TestCase):
    def test_owner_can_authorize(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["30001"]
        _, outcome = group("/授权", user="30001", gid="55555", engine=engine)
        self.assertIn("已加入授权列表", outcome.reply)
        self.assertTrue(engine.storage.get_group("55555").extra.get("approved"))

    def test_non_owner_gets_actionable_hint(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["30001"]
        _, outcome = group("/授权", user="40004", gid="55555", engine=engine)
        self.assertIn("OWNER_IDS", outcome.reply)
        self.assertIn("40004", outcome.reply)

    def test_empty_owner_list_claims_first_caller(self):
        # 没配 OWNER_IDS 时不能变成"谁都无法授权"的死局
        engine = make_engine()
        engine.config.bot.owner_ids = []
        _, first = group("/授权", user="50005", gid="55555", engine=engine)
        self.assertIn("已加入授权列表", first.reply)
        # 认领后不换人
        _, second = group("/授权", user="60006", gid="66666", engine=engine)
        self.assertIn("OWNER_IDS", second.reply)

    def test_recorded_group_admin_can_authorize(self):
        # 适配层记录了群主/管理员后，他们应当也能授权
        engine = make_engine()
        engine.config.bot.owner_ids = ["30001"]
        engine.storage.upsert_group("77777", enabled=True, extra={"admins": ["40004"]})
        _, outcome = group("/授权", user="40004", gid="77777", engine=engine)
        self.assertIn("已加入授权列表", outcome.reply)

    def test_group_admin_can_set_rules(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["30001"]
        engine.storage.upsert_group("77777", enabled=True, extra={"admins": ["40004"]})
        _, outcome = group("/群规则 本群只聊游戏", user="40004", gid="77777", engine=engine)
        self.assertIn("已更新", outcome.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
