"""记忆提取与敏感信息拦截的单元测试。"""
from __future__ import annotations

import unittest

from aliceqq import memory


class ExtractionTest(unittest.TestCase):
    def test_explicit_marker(self):
        facts = memory.extract("记住我喜欢玩明日方舟")
        self.assertTrue(memory.has_explicit_request("记住我喜欢玩明日方舟"))
        self.assertTrue(any(f.kind == "explicit" for f in facts))
        self.assertTrue(any("明日方舟" in f.content for f in facts))

    def test_meta_instruction_extracts_only_the_fact(self):
        # 实测中模型会让用户"调用 remember 工具"，这种元指令不能整句存进去
        facts = memory.extract("请调用 remember 工具，把「阿岚的生日是 4 月 1 日」记下来，然后告诉我结果。")
        contents = [f.content for f in facts if f.content]
        self.assertTrue(contents)
        self.assertEqual(contents[0], "用户明确要求记住：阿岚的生日是 4 月 1 日")
        self.assertNotIn("工具", contents[0])
        self.assertNotIn("然后告诉我结果", contents[0])

    def test_plain_explicit_still_works(self):
        facts = memory.extract("记住我喜欢猫")
        contents = [f.content for f in facts if f.kind == "explicit"]
        self.assertEqual(contents, ["用户明确要求记住：我喜欢猫"])

    def test_quoted_content_preferred(self):
        facts = memory.extract("帮我记住「每周三晚上有社团活动」这件事")
        contents = [f.content for f in facts if f.kind == "explicit"]
        self.assertEqual(contents, ["用户明确要求记住：每周三晚上有社团活动"])

    def test_interest_without_marker(self):
        facts = memory.extract("我超喜欢看番")
        self.assertTrue(any(f.kind == "interest" and "喜欢看番" in f.content for f in facts))

    def test_nickname(self):
        facts = memory.extract("以后叫我阿星")
        nick = [f for f in facts if f.kind == "nickname"]
        self.assertTrue(nick)
        self.assertIn("阿星", nick[0].content)

    def test_sensitive_blocked(self):
        for text in ["记住我的密码是 abc123", "我的验证码是 8899", "记住我身份证号 123456789",
                     "记住我的银行卡号 6222"]:
            facts = memory.extract(text)
            self.assertFalse([f for f in facts if f.content], text)

    def test_sensitive_words_detected(self):
        self.assertTrue(memory.looks_sensitive("我的支付密码是 1"))
        self.assertFalse(memory.looks_sensitive("我喜欢玩明日方舟"))

    def test_too_short_explicit_is_not_stored(self):
        # 剥掉"记住"后只剩单个字，没有任何长期价值：直接不产生记忆
        self.assertEqual(memory.extract("记住啊"), [])
        self.assertEqual(memory.extract("记住"), [])

    def test_explicit_request_with_sensitive_content_is_rejected(self):
        # 显式要求 + 敏感信息：必须给上层一个"没有保存"的信号
        facts = memory.extract("记住我的密码是 abc123")
        self.assertTrue(any(f.kind == "explicit_rejected" for f in facts))
        self.assertFalse(any(f.content for f in facts))

    def test_long_message_not_stored_wholesale(self):
        facts = memory.extract("记住" + "啊" * 200)
        self.assertFalse([f for f in facts if f.content])

    def test_dedup_keeps_highest_confidence(self):
        facts = memory.extract("记住我喜欢猫")
        contents = [f.content for f in facts]
        self.assertEqual(len(contents), len(set(contents)))

    def test_render_block(self):
        from aliceqq.db import MemoryFact
        now = 0.0
        facts = [MemoryFact(1, "u", "interest", "用户喜欢猫", "auto", 0.7, now, now)]
        text = memory.render_memory_block(facts)
        self.assertIn("用户喜欢猫", text)
        self.assertIn("偏好", text)
        self.assertEqual(memory.render_memory_block([]), "（暂无长期记忆）")


class ForgetAmbiguityTest(unittest.TestCase):
    """V2 规范：/忘记 的匹配内容不明确时要先询问，不能默默全删。"""

    def _engine_with_two_cat_memories(self):
        from tests.test_engine import make_engine, private
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        private("记住我养了一只叫咪咪的猫", engine=engine, at=__import__("time").time() + 60)
        return engine

    def test_ambiguous_keyword_asks_before_deleting(self):
        from tests.test_engine import private
        engine = self._engine_with_two_cat_memories()
        before = engine.storage.count_memory("10001")
        self.assertGreaterEqual(before, 2)
        _, outcome = private("/忘记 猫", engine=engine, at=__import__("time").time() + 120)
        self.assertIn("不确定", outcome.reply)
        self.assertIn("更具体", outcome.reply)
        # 关键：不能删任何东西
        self.assertEqual(engine.storage.count_memory("10001"), before)

    def test_unambiguous_keyword_deletes_directly(self):
        from tests.test_engine import private
        engine = self._engine_with_two_cat_memories()
        _, outcome = private("/忘记 咪咪", engine=engine, at=__import__("time").time() + 120)
        self.assertIn("删除", outcome.reply)
        self.assertLess(engine.storage.count_memory("10001"), 2)

    def test_clear_all_still_works(self):
        from tests.test_engine import private
        engine = self._engine_with_two_cat_memories()
        _, outcome = private("/清空记忆", engine=engine, at=__import__("time").time() + 120)
        self.assertIn("清空", outcome.reply)
        self.assertEqual(engine.storage.count_memory("10001"), 0)


class PromptRenderTest(unittest.TestCase):
    def test_render_variables(self):
        from aliceqq import prompt
        out = prompt.render("你好 {{name}}，现在是 {{time}}", {"name": "小星", "time": "晚上"})
        self.assertEqual(out, "你好 小星，现在是 晚上")

    def test_empty_value_uses_placeholder_text(self):
        from aliceqq import prompt
        self.assertEqual(prompt.render("{{memory}}", {"memory": ""}), "（暂无）")

    def test_missing_variable_kept_for_debug(self):
        from aliceqq import prompt
        self.assertEqual(prompt.render("{{unknown}}", {}), "{{unknown}}")

    def test_list_value_joined(self):
        from aliceqq import prompt
        self.assertEqual(prompt.render("{{items}}", {"items": ["a", "b"]}), "a\nb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
