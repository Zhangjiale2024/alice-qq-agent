"""诚实护栏测试：必须确定性地拦住"假装执行"。"""
from __future__ import annotations

import unittest

from aliceqq import honesty
from aliceqq.engine import IncomingMessage, MODE_ALICE
from tests.test_engine import make_engine, private


class HonestyModuleTest(unittest.TestCase):
    def test_detects_web_search_request(self):
        self.assertIn("web_search", honesty.requested_capabilities("帮我上网查一下今天的新闻"))
        self.assertIn("web_search", honesty.requested_capabilities("搜一下最近有什么新番"))

    def test_detects_image_request(self):
        self.assertIn("image_generation", honesty.requested_capabilities("帮我画一张图"))

    def test_normal_chat_has_no_capability(self):
        self.assertEqual(honesty.requested_capabilities("今天天气不错"), [])
        self.assertEqual(honesty.requested_capabilities("你在干什么"), [])

    def test_blocks_fabricated_search(self):
        verdict = honesty.check("帮我上网查一下今天的新闻",
                                "爱丽丝去查一下，稍等。（正在打开联网检索……）")
        self.assertTrue(verdict.replaced)
        self.assertEqual(verdict.capability, "web_search")
        self.assertIn("还没接上外部网络", verdict.text)

    def test_honest_refusal_is_not_replaced(self):
        verdict = honesty.check("帮我上网查一下今天的新闻",
                                "爱丽丝现在还没接上外部网络，查不到最新消息。")
        self.assertFalse(verdict.replaced)

    def test_normal_chat_reply_never_touched(self):
        verdict = honesty.check("今天好累", "那今天就先不做任务了，休息一下。")
        self.assertFalse(verdict.replaced)

    def test_blocks_fabricated_image(self):
        verdict = honesty.check("帮我画一张爱丽丝的图", "好，正在生成……画好了！")
        self.assertTrue(verdict.replaced)
        self.assertEqual(verdict.capability, "image_generation")


class HonestyGuardInEngineTest(unittest.TestCase):
    """端到端：模型假装执行时，引擎必须替换成如实说明。"""

    class FabricatingLLM:
        name = "fabricating"

        def __init__(self, scripted: str):
            self.scripted = scripted

        def chat(self, messages, *, config, tools=None):
            from aliceqq.llm import Reply
            return Reply(text=self.scripted, provider="fabricating", model="fake")

    def _engine_with(self, scripted: str):
        engine = make_engine()
        engine.llm = self.FabricatingLLM(scripted)
        return engine

    def test_engine_replaces_fabricated_capability_claim(self):
        engine = self._engine_with("爱丽丝去查一下，稍等。（正在打开联网检索……）")
        _, outcome = private("帮我上网查一下今天的新闻", engine=engine)
        self.assertIn("还没接上外部网络", outcome.reply)
        self.assertNotIn("正在打开联网检索", outcome.reply)
        self.assertIn("honesty_replaced", outcome.debug)

    def test_engine_keeps_normal_reply(self):
        engine = self._engine_with("那今天就先不做任务了，去存档休息吧。")
        _, outcome = private("今天好累", engine=engine)
        self.assertIn("存档", outcome.reply)
        self.assertNotIn("honesty_replaced", outcome.debug)

    def test_engine_keeps_honest_refusal(self):
        engine = self._engine_with("爱丽丝现在还没接上外部网络，查不到最新消息。")
        _, outcome = private("帮我查一下今天的新闻", engine=engine)
        self.assertIn("还没接上外部网络", outcome.reply)
        self.assertNotIn("honesty_replaced", outcome.debug)

    def test_engine_strips_leaked_think_aloud(self):
        # 真实模型实测会出现这种"把推理过程当回复"的泄漏
        leaked = ("用户明确要求记住生日，这是一条稳定的长期信息。不过长期记忆区块里已经出现了这条内容，"
                  "说明系统已经保存成功，我直接自然确认即可，不需要重复调用。\n\n"
                  "爱丽丝收到！这条已经写进存档了。")
        engine = self._engine_with(leaked)
        _, outcome = private("记住我的生日是 4 月 1 日", engine=engine)
        self.assertNotIn("长期记忆区块", outcome.reply)
        self.assertNotIn("不需要重复调用", outcome.reply)
        self.assertIn("已经写进存档", outcome.reply)
        self.assertTrue(outcome.debug.get("sanitized"))

    def test_engine_handles_reply_that_is_all_think_aloud(self):
        engine = self._engine_with("我应该直接确认，不需要调用工具。")
        _, outcome = private("记住我的生日是 4 月 1 日", engine=engine)
        self.assertEqual(outcome.decision, "error")
        self.assertNotIn("我应该", outcome.reply or "")

    def test_normal_reply_not_sanitized(self):
        engine = self._engine_with("那今天就先休息吧，爱丽丝陪着你。")
        _, outcome = private("今天好累", engine=engine)
        self.assertFalse(outcome.debug.get("sanitized"))
        self.assertIn("休息", outcome.reply)

    def test_false_save_claim_gets_corrected(self):
        # 模型顺着用户说"记下了"，但提取器其实什么都没保存 -> 必须追加修正
        engine = self._engine_with("好，爱丽丝记下了。")
        # "记住" 后面没有实际内容：提取器什么都没存，模型却很可能顺着说"记下了"
        _, outcome = private("记住", engine=engine)
        self.assertEqual(outcome.memory_saved, [])
        self.assertEqual(outcome.memory_rejected, [])
        self.assertTrue(outcome.memory_pending_review)
        self.assertIn("没有存进记忆", outcome.reply)

    def test_real_save_does_not_get_correction_note(self):
        engine = self._engine_with("好，爱丽丝记下了。")
        _, outcome = private("记住我住在火星基地第七区", engine=engine)
        self.assertTrue(outcome.memory_saved)
        self.assertFalse(outcome.memory_pending_review)
        self.assertNotIn("没有存进记忆", outcome.reply)

    def test_tool_catalog_marks_unavailable(self):
        from aliceqq import tools
        engine = make_engine()
        catalog = tools.catalog_text(engine.config)
        self.assertIn("未启用联网检索", catalog)
        self.assertIn("绝对不能尝试调用", catalog)
        specs = tools.registry(engine.config)
        self.assertFalse(specs["web_search"].available)


if __name__ == "__main__":
    unittest.main(verbosity=2)
