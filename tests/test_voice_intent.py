"""语音意图识别 + "声称发了媒体但其实没发"的护栏。

两个都是**实测踩出来的**：

1. 用户说"你来唱给我听，把语音发给我"——`_wants_voice` 返回 False，
   因为旧模式只认【发→语音】的顺序，也不认"唱"。
   结果：没合成语音，而模型在正文里自己编了"语音尝试发送中……"。

2. 模型会**编造媒体发送结果**。V5 明确"不编造……媒体发送结果"，
   所以正文声称发了媒体、而本轮实际没挂媒体时，必须把那些说法清掉。
"""
from __future__ import annotations

import unittest

from aliceqq.engine import Engine
from tests.test_engine import make_engine


class VoiceIntentTest(unittest.TestCase):
    SHOULD_MATCH = [
        "你来唱给我听，把语音发给我",     # 截图里那句：语音→发的反序 + 唱歌
        "把语音发给我",
        "发个语音",
        "发语音",
        "来段语音",
        "来一条语音",
        "唱给我听",
        "唱一首",
        "唱一段",
        "用爱丽丝的声音唱",
        "用温柔的声音安慰我",
        "说句话",
        "念一下",
        "读出来",
        "voice",
    ]
    SHOULD_NOT_MATCH = [
        "今天天气不错",
        "这游戏怎么玩",
        "帮我看下代码",
        "这个角色强不强",
        "唱歌好听吗这个角色",
        "我们在群里聊点别的",
    ]

    def test_natural_voice_requests_are_detected(self):
        for text in self.SHOULD_MATCH:
            with self.subTest(text=text):
                self.assertTrue(Engine._wants_voice(text), text)

    def test_unrelated_text_is_not_detected(self):
        for text in self.SHOULD_NOT_MATCH:
            with self.subTest(text=text):
                self.assertFalse(Engine._wants_voice(text), text)

    def test_regression_voice_before_send_verb(self):
        """回归：旧模式只认「发…语音」，漏掉「语音…发」这种语序。"""
        self.assertTrue(Engine._wants_voice("把语音发给我"))

    def test_regression_singing_request(self):
        """回归：唱歌请求原来一条都不认。"""
        self.assertTrue(Engine._wants_voice("唱给我听"))


class MediaOverclaimTest(unittest.TestCase):
    """模型声称发了媒体、但实际没发 —— 必须清理，且不能误伤正常陈述。"""

    OVERCLAIMS = [
        "现在爱丽丝把这段发成语音——启动！",
        "（语音尝试发送中……）",
        "语音已发给你了",
        "这就发送语音",
        "马上发出录音",
        "图片已经发过去了",
        "表情发送给你了",
    ]
    HARMLESS = [
        "爱丽丝的语音是合成音，不是官方配音哦。",
        "这段歌词爱丽丝背不下来，只能唱开头一段。",
        "如果这次还是发不出来，爱丽丝就把词写给你看。",
        "要听吗？",
        "爱丽丝收到点歌指令了！",
        "爱丽丝还不知道这首歌的完整歌词。",
    ]

    def test_overclaim_is_stripped_when_no_media(self):
        for text in self.OVERCLAIMS:
            with self.subTest(text=text):
                out = Engine._strip_media_overclaim(text, has_media=False)
                self.assertNotEqual(out, text, "没清理：%s" % text)

    def test_stripped_text_says_so(self):
        """清理之后要如实说明，不能留下"我刚才说发了"却不明说。"""
        out = Engine._strip_media_overclaim("语音已发给你了", has_media=False)
        self.assertIn("并没有", out)

    def test_kept_when_media_really_sent(self):
        for text in self.OVERCLAIMS:
            with self.subTest(text=text):
                self.assertEqual(Engine._strip_media_overclaim(text, has_media=True), text)

    def test_harmless_text_is_untouched(self):
        for text in self.HARMLESS:
            with self.subTest(text=text):
                self.assertEqual(Engine._strip_media_overclaim(text, has_media=False), text)

    def test_empty_text(self):
        self.assertEqual(Engine._strip_media_overclaim("", has_media=False), "")

    def test_only_removes_the_offending_sentence(self):
        text = "爱丽丝收到点歌指令了！\n语音已发给你了\n这段歌词爱丽丝背不下来。"
        out = Engine._strip_media_overclaim(text, has_media=False)
        self.assertIn("收到点歌指令", out)
        self.assertIn("背不下来", out)
        self.assertNotIn("语音已发给你了", out)


class MediaOverclaimIntegrationTest(unittest.TestCase):
    """走完整流程：模型编了"发了语音"但媒体没挂上 -> 正文里不该留那句话。"""

    def test_engine_strips_overclaim_when_voice_unavailable(self):
        from unittest import mock
        from aliceqq.llm import Reply

        eng = make_engine()
        eng.config.features.voice = False          # 语音能力关掉 -> 不会挂媒体
        fake = "爱丽丝唱给你听！\n语音已发给你了\n这段歌词记不全。"
        with mock.patch.object(eng.llm, "chat",
                               return_value=Reply(text=fake, provider="mock", model="mock")):
            from aliceqq.engine import IncomingMessage
            out = eng.handle(IncomingMessage(user_id="u1", text="把语音发给我",
                                            is_private=True, user_name="小星"))
        self.assertEqual(len(out.media), 0)
        self.assertNotIn("语音已发给你了", out.reply)
        self.assertIn("并没有", out.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
