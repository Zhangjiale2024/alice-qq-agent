"""回归测试：模式切换相关。

用户反馈"切到助手后切不回爱丽丝"。根因是**指令的处理顺序**：
早期实现里指令排在"群聊触发判定"和"限流"之后，于是
  - 群里被回满限流后，/爱丽丝 完全没反应；
  - 群消息没命中触发时，指令被静默丢弃。
指令是零成本的本地操作，必须最先处理，且不占回复配额。
"""
from __future__ import annotations

import time
import unittest

from aliceqq.engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT
from tests.test_engine import group, make_engine, private


class ModeRoundTripTest(unittest.TestCase):
    def test_alice_assistant_alice_round_trip(self):
        engine = make_engine()
        sequence = [("/助手", MODE_ASSISTANT), ("你好", MODE_ASSISTANT),
                    ("/爱丽丝", MODE_ALICE), ("你好", MODE_ALICE)]
        for index, (text, expected) in enumerate(sequence):
            _, outcome = private(text, engine=engine, at=time.time() + index * 120)
            self.assertEqual(outcome.mode, expected, f"{text!r} 之后模式应为 {expected}")
        self.assertEqual(engine.storage.get_profile("10001").mode, MODE_ALICE)

    def test_switch_back_uses_alice_persona(self):
        engine = make_engine()
        private("/助手", engine=engine)
        private("你是谁", engine=engine, at=time.time() + 120)
        private("/爱丽丝", engine=engine, at=time.time() + 240)
        _, outcome = private("你是谁", engine=engine, at=time.time() + 360)
        self.assertEqual(outcome.mode, MODE_ALICE)
        # mock 下爱丽丝模式的自我介绍会提到游戏开发部
        self.assertTrue("爱丽丝" in outcome.reply)


class CommandAliasTest(unittest.TestCase):
    def test_ascii_and_case_variants_switch_to_alice(self):
        for command in ("/爱丽丝", "/alice", "/ALICE", "/Alice"):
            engine = make_engine()
            engine.storage.set_mode("10001", MODE_ASSISTANT)
            _, outcome = private(command, engine=engine)
            self.assertEqual(outcome.mode, MODE_ALICE, f"{command!r} 应该切到爱丽丝模式")

    def test_japanese_alias(self):
        engine = make_engine()
        engine.storage.set_mode("10001", MODE_ASSISTANT)
        _, outcome = private("/アリス", engine=engine)
        self.assertEqual(outcome.mode, MODE_ALICE)

    def test_unknown_command_is_not_chat(self):
        engine, outcome = private("/alice2")
        self.assertEqual(outcome.decision, "command")
        self.assertIn("不认识这条指令", outcome.reply)


class CommandPriorityTest(unittest.TestCase):
    @staticmethod
    def _minute_safe_base() -> float:
        """取一个"秒数居中"的时间戳作为基准。

        限流按 YYYYMMDDHHMM 计数，如果基准时间落在 :59，第二条消息就跨到下一分钟、
        计数被重置，用例会随运行时刻随机失败——这正是之前那个"偶发失败"的真身。
        """
        now = time.time()
        # 对齐到当前分钟的 :30，保证 +1s / +2s 都还在同一分钟内
        return now - (now % 60) + 30

    def test_command_works_after_group_rate_limit_exhausted(self):
        engine = make_engine(group__rate_limit_per_minute=1)
        base = self._minute_safe_base()
        # 先把这个群这个分钟的配额用掉
        _, first = group("爱丽丝 在吗", user="20001", gid="88888", engine=engine, at=base)
        self.assertTrue(first.should_send)
        _, limited = group("爱丽丝 再来一句", user="20002", gid="88888", engine=engine, at=base + 1)
        self.assertEqual(limited.decision, "rate_limited")
        # 限流状态下，指令仍然必须生效——这正是"切不回爱丽丝"的场景
        _, switch = group("/爱丽丝", user="20001", gid="88888", engine=engine, at=base + 2)
        self.assertEqual(switch.mode, MODE_ALICE, f"限流时指令被吞掉了：{switch.reason}")
        self.assertIn("爱丽丝模式", switch.reply)

    def test_command_does_not_consume_rate_limit(self):
        engine = make_engine(group__rate_limit_per_minute=1)
        base = self._minute_safe_base()
        group("/状态", user="20001", gid="88888", engine=engine, at=base)
        # 指令不占配额，所以第一条正常消息仍然能发出
        _, chat = group("爱丽丝 在吗", user="20001", gid="88888", engine=engine, at=base + 1)
        self.assertEqual(chat.decision, "replied", chat.reason)

    def test_command_works_without_trigger(self):
        # 既没 @ 也没关键词，普通消息会沉默，但指令必须响应
        engine = make_engine()
        _, plain = group("今天天气不错", user="20001", gid="88888", engine=engine,
                         mentioned=False, at=time.time())
        self.assertFalse(plain.should_send)
        _, command = group("/助手", user="20001", gid="88888", engine=engine,
                           mentioned=False, at=time.time() + 1)
        self.assertEqual(command.decision, "command")
        self.assertEqual(command.mode, MODE_ASSISTANT)

    def test_command_works_in_unauthorized_group(self):
        # 未授权群平时完全不理，但 owner 得能靠指令把群开起来，否则没有出路
        engine = make_engine(group__access="approved_groups")
        engine.config.bot.owner_ids = ["20001"]
        outcome = engine.handle(IncomingMessage(user_id="20001", text="/帮助", is_private=False,
                                                group_id="77777", mentioned=False))
        self.assertEqual(outcome.decision, "command")
        # V3 起帮助按权限分组展示
        self.assertIn("/爱丽丝", outcome.reply)
        self.assertIn("仅机器人所有者", outcome.reply)

    def test_mode_persists_across_scenes(self):
        engine = make_engine()
        private("/助手", engine=engine)
        _, in_group = group("你是谁", user="10001", gid="88888", engine=engine, at=time.time() + 120)
        self.assertEqual(in_group.mode, MODE_ASSISTANT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
