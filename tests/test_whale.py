"""DeepSeek 小鲸鱼角色卡：注册表、切换、风格护栏、分条。"""
from __future__ import annotations

import os
import time
import unittest

from aliceqq import characters as chars
from aliceqq import style as style_mod
from aliceqq.engine import IncomingMessage, MODE_ALICE, MODE_WHALE
from aliceqq.llm import BaseLLM, Reply
from tests.test_engine import make_engine, private


class RegistryTest(unittest.TestCase):
    def test_three_characters_registered(self):
        self.assertIn(MODE_WHALE, chars.CHARACTERS)
        self.assertIn(MODE_ALICE, chars.CHARACTERS)
        self.assertEqual(len(chars.all_ids()), 3)

    def test_by_command_matches_aliases(self):
        for command in ("/小鲸鱼", "/鲸鱼", "/deepseek", "/ds", "/DS"):
            self.assertEqual(chars.by_command(command).id, MODE_WHALE, command)
        self.assertEqual(chars.by_command("/爱丽丝").id, MODE_ALICE)
        self.assertIsNone(chars.by_command("/不存在的指令"))

    def test_unknown_mode_falls_back(self):
        # 脏数据不能让回复挂掉
        self.assertEqual(chars.get("garbage").id, chars.DEFAULT_MODE)
        self.assertEqual(chars.get(None).id, chars.DEFAULT_MODE)

    def test_only_whale_splits_messages(self):
        self.assertTrue(chars.CHARACTERS[MODE_WHALE].split_messages)
        self.assertFalse(chars.CHARACTERS[MODE_ALICE].split_messages)
        self.assertFalse(chars.CHARACTERS["assistant"].split_messages)

    def test_switch_help_lists_all(self):
        help_text = chars.switch_help()
        for character in chars.CHARACTERS.values():
            self.assertIn(character.name, help_text)


class SanitizeTest(unittest.TestCase):
    """角色卡的「AI 味黑名单」是硬要求，必须确定性清掉。"""

    def test_detects_blacklist(self):
        self.assertIn("首先", style_mod.detect("首先，我要说明"))
        self.assertIn("本小姐", style_mod.detect("本小姐建议"))
        self.assertEqual(style_mod.detect("草"), [])

    def test_strips_parenthetical_inner_monologue(self):
        """卡里明确警告：括号内容会被当成正式消息发到群里。"""
        report = style_mod.sanitize("（我不太想接话）你说是就是吧")
        self.assertEqual(report.cleaned, "你说是就是吧")
        self.assertEqual(len(report.stripped_parens), 1)

    def test_strips_halfwidth_and_bracket_parens(self):
        self.assertNotIn("(", style_mod.sanitize("(内心戏)你好").cleaned)
        self.assertNotIn("【", style_mod.sanitize("【旁白】你好").cleaned)

    def test_replaces_hime_sama(self):
        # "本小姐"是二创女仆腔，有等价说法就直接换，比删句保留更多信息
        report = style_mod.sanitize("服务器繁忙，那是在加载本小姐的智慧啦")
        self.assertIn("小鲸鱼", report.cleaned)
        self.assertNotIn("本小姐", report.cleaned)

    def test_removes_markdown(self):
        report = style_mod.sanitize("# 推荐\n- 第一个\n- 第二个")
        self.assertNotIn("#", report.cleaned)
        self.assertNotIn("- ", report.cleaned)
        self.assertTrue(report.markdown_removed)

    def test_drops_blacklisted_sentences_when_others_survive(self):
        report = style_mod.sanitize("首先我要说明。这个不错。")
        self.assertNotIn("首先", report.cleaned)
        self.assertIn("这个不错", report.cleaned)

    def test_never_returns_empty(self):
        """清洗后不能为空——宁可留一点 AI 味，也不能发空消息。"""
        for text in ("首先，我要说明一下。", "本小姐", "（内心戏）", "。", "草"):
            self.assertTrue(style_mod.sanitize(text).cleaned.strip(), text)

    def test_collapses_duplicated_pronoun(self):
        # "作为AI我觉得" 删掉"作为AI"后不该变成"我我觉得"
        self.assertEqual(style_mod.sanitize("作为AI我觉得不错").cleaned, "我觉得不错")

    def test_strips_trailing_period(self):
        self.assertEqual(style_mod.sanitize("草。").cleaned, "草")

    def test_reports_what_it_did(self):
        report = style_mod.sanitize("（旁白）你好。**重要**")
        self.assertTrue(report.changed)
        self.assertNotEqual(report.summary(), "未命中")


class SplitMessagesTest(unittest.TestCase):
    """分条：小鲸鱼的核心习惯，角色卡反复强调"一句话一条消息"。"""

    def test_splits_sentences(self):
        # 角色卡原文示例
        self.assertEqual(style_mod.split_messages("累了就睡。醒了继续累。"),
                         ["累了就睡。", "醒了继续累。"])

    def test_respects_existing_newlines(self):
        # 卡片示例："它强归它强 / 我便宜啊"
        self.assertEqual(style_mod.split_messages("它强归它强\n我便宜啊"),
                         ["它强归它强", "我便宜啊"])

    def test_single_short_message_stays_one(self):
        self.assertEqual(style_mod.split_messages("草"), ["草"])

    def test_caps_parts_without_losing_text(self):
        text = "第一句。第二句。第三句。第四句。第五句。"
        parts = style_mod.split_messages(text, max_parts=3)
        self.assertEqual(len(parts), 3)
        # 超出上限只能合并，不能丢内容
        joined = "".join(parts)
        for fragment in ("第一句", "第二句", "第三句", "第四句", "第五句"):
            self.assertIn(fragment, joined)

    def test_empty_input(self):
        self.assertEqual(style_mod.split_messages(""), [])
        self.assertEqual(style_mod.split_messages("   "), [])

    def test_long_message_is_chopped(self):
        parts = style_mod.split_messages("字" * 200, max_chars=40)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), "字" * 200)


class WhaleEngineTest(unittest.TestCase):
    class _Fixed(BaseLLM):
        def __init__(self, text=""):
            self.text = text

        def chat(self, messages, *, config, tools=None):
            return Reply(text=self.text, usage={})

    def setUp(self):
        self.engine = make_engine()
        self.engine.llm = self._Fixed()
        self.engine.config.bot.owner_ids = ["10001"]

    def _send(self, text, user="10001"):
        self.engine.storage.ensure_profile(user, "乐希", MODE_ALICE)
        return self.engine.handle(IncomingMessage(user_id=user, text=text, is_private=True,
                                                  user_name="乐希", timestamp=time.time()))

    def test_switch_command_sets_mode(self):
        outcome = self._send("/小鲸鱼")
        self.assertEqual(outcome.mode, MODE_WHALE)
        self.assertEqual(self.engine.storage.get_profile("10001").mode, MODE_WHALE)

    def test_whale_uses_its_own_template(self):
        self._send("/小鲸鱼")
        _, _, variables = self.engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="乐希"),
            MODE_WHALE)
        self.assertEqual(variables["mode"], MODE_WHALE)
        template = self.engine.template("deepseek_whale.md")
        self.assertIn("小鲸鱼", template)
        self.assertIn("AI 味黑名单", template)

    def test_whale_reply_is_sanitized(self):
        self._send("/小鲸鱼")
        self.engine.llm.text = "首先我要说明。（内心戏）本小姐觉得可以。"
        outcome = self._send("说点什么")
        self.assertNotIn("首先", outcome.reply)
        self.assertNotIn("本小姐", outcome.reply)
        self.assertNotIn("内心戏", outcome.reply)
        self.assertIn("style", outcome.debug)

    def test_whale_reply_is_split(self):
        self._send("/小鲸鱼")
        self.engine.llm.text = "累了就睡。醒了继续累。"
        outcome = self._send("说点什么")
        self.assertEqual(len(outcome.outgoing), 2)
        self.assertEqual(outcome.debug.get("split"), 2)

    def test_alice_reply_is_untouched(self):
        self.engine.llm.text = "首先我要说明。（内心戏）本小姐觉得可以。"
        outcome = self._send("测试")
        # 爱丽丝不该被小鲸鱼的护栏动到
        self.assertIn("本小姐", outcome.reply)
        self.assertEqual(outcome.replies, [])

    def test_outgoing_backward_compatible(self):
        # 没有 replies 时，outgoing 应回落到单条 reply
        self.engine.llm.text = "在的"
        outcome = self._send("测试")
        self.assertEqual(outcome.replies, [])
        self.assertEqual(outcome.outgoing, ["在的"])

    def test_outgoing_empty_when_no_reply(self):
        self.engine.llm.text = ""
        outcome = self._send("测试")
        self.assertEqual(outcome.outgoing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
