"""回归测试：群聊"发什么都回复"。

根因是两个缺陷叠加：
1. 默认关键词里有"老师""游戏""ba"这类通用词，几乎任何群聊都会命中；
2. 话题延续只看时间窗口，机器人说过话后 90 秒内任何人说话都算"相关"。
"""
from __future__ import annotations

import time
import unittest

from aliceqq.config import AppConfig, filter_keywords
from aliceqq.engine import IncomingMessage
from tests.test_engine import group, make_engine


class KeywordSanitizeTest(unittest.TestCase):
    def test_generic_words_are_dropped(self):
        kept = filter_keywords(["爱丽丝", "老师", "游戏", "ba", "蔚蓝档案"])
        self.assertIn("爱丽丝", kept)
        self.assertIn("蔚蓝档案", kept)
        self.assertNotIn("老师", kept)
        self.assertNotIn("游戏", kept)
        self.assertNotIn("ba", kept)

    def test_short_ascii_dropped(self):
        self.assertEqual(filter_keywords(["ab", "abc"]), ["abc"])

    def test_defaults_have_no_generic_words(self):
        keywords = filter_keywords(AppConfig().group.keywords)
        for bad in ("老师", "游戏", "ba", "任务", "副本", "升级"):
            self.assertNotIn(bad, keywords)


class GroupNoiseTest(unittest.TestCase):
    """群里与爱丽丝无关的话，一句都不该回。"""

    def test_plain_chat_never_triggers(self):
        engine = make_engine()
        noises = ["今天中午吃什么", "老师布置的作业好多", "晚上一起打游戏吗",
                  "这个副本真难", "我又升级了", "ba 是什么意思", "明天要下雨"]
        for index, text in enumerate(noises):
            _, outcome = group(text, user=f"30{index:03d}", engine=engine,
                               mentioned=False, at=time.time() + index)
            self.assertFalse(outcome.should_send,
                             f"「{text}」不该触发回复，但触发了：{outcome.reason}")

    def test_keyword_group_bypassed_the_bug(self):
        # 修复前这两句都会命中关键词
        engine = make_engine()
        _, math_teacher = group("老师今天讲的内容没听懂", user="40001", engine=engine, mentioned=False,
                                at=time.time())
        _, game_talk = group("晚上打游戏吗", user="40002", engine=engine, mentioned=False,
                             at=time.time() + 1)
        self.assertFalse(math_teacher.should_send)
        self.assertFalse(game_talk.should_send)


class TopicContinuityTest(unittest.TestCase):
    """话题延续必须"同一个人接着说"，而不是谁说话都算。"""

    def _seed_conversation(self):
        engine = make_engine()
        base = time.time()
        # A 被 @ 并得到回复
        _, first = group("爱丽丝在吗", user="50001", gid="88888", engine=engine, at=base)
        self.assertTrue(first.should_send)
        return engine, base

    def test_same_user_followup_within_window_is_answered(self):
        engine, base = self._seed_conversation()
        _, follow = group("那你觉得哪个好玩", user="50001", gid="88888", engine=engine,
                          mentioned=False, at=base + 20)
        self.assertTrue(follow.should_send, follow.reason)
        self.assertIn("relevant_topic", follow.debug.get("trigger", []))

    def test_other_user_in_window_is_not_answered(self):
        # 这是"发什么都回"的核心症状：修复前 B 在窗口内说话也会被当成延续话题
        engine, base = self._seed_conversation()
        _, other = group("我今晚要加班", user="50002", gid="88888", engine=engine,
                         mentioned=False, at=base + 20)
        self.assertFalse(other.should_send,
                         f"窗口内其他人不该被当成延续话题，但回复了：{other.reason}")

    def test_followup_after_window_is_not_answered(self):
        engine, base = self._seed_conversation()
        _, late = group("在吗", user="50001", gid="88888", engine=engine,
                        mentioned=False, at=base + 3600)
        self.assertFalse(late.should_send, late.reason)

    def test_own_cooldown_applies_to_topic_continuation(self):
        # 同一人 1 秒后接着说：话题延续不能绕过冷却
        engine, base = self._seed_conversation()
        _, spam = group("还有一个问题", user="50001", gid="88888", engine=engine,
                        mentioned=False, at=base + 1)
        self.assertFalse(spam.should_send, f"冷却未生效：{spam.reason}")

    def test_mention_always_works_even_without_topic(self):
        engine = make_engine()
        _, outcome = group("爱丽丝帮我看看这个", user="60001", gid="88888", engine=engine,
                           mentioned=True, at=time.time())
        self.assertTrue(outcome.should_send)

    def test_same_user_only_can_be_disabled(self):
        engine = make_engine(group__relevant_topic_same_user_only=False)
        base = time.time()
        group("爱丽丝在吗", user="50001", gid="88888", engine=engine, at=base)
        _, other = group("我今晚要加班", user="50002", gid="88888", engine=engine,
                         mentioned=False, at=base + 20)
        self.assertTrue(other.should_send)
        self.assertIn("relevant_topic", other.debug.get("trigger", []))


class TriggerConfigTest(unittest.TestCase):
    def test_removing_relevant_topic_disables_continuation(self):
        engine = make_engine(group__triggers=["mention", "keyword", "command"])
        base = time.time()
        group("爱丽丝在吗", user="50001", gid="88888", engine=engine, at=base)
        _, follow = group("那你觉得呢", user="50001", gid="88888", engine=engine,
                          mentioned=False, at=base + 20)
        self.assertFalse(follow.should_send)


if __name__ == "__main__":
    unittest.main(verbosity=2)
