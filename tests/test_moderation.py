"""V5 第二批测试：反广告/关键词审核 + 警告/撤回/禁言/移出 + 管理员确认。

这是**安全关键**模块：误判会把正常群友当成广告处罚，所以测试的重点不是
"能不能识别广告"，而是**"不该处罚的绝不能处罚"**。
"""
from __future__ import annotations

import time
import unittest
import uuid

from aliceqq import moderation as mod
from aliceqq.config import AppConfig, GroupSettings, PROJECT_ROOT
from aliceqq.db import Storage
from aliceqq.engine import Engine, IncomingMessage
from tests.test_engine import make_engine


def _store():
    tmp = PROJECT_ROOT / "data" / "test-run" / f"mod-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=True)
    return Storage(tmp / "alice.db")


def _group_engine(owner="10001", **settings):
    eng = make_engine()
    eng.config.bot.owner_ids = [owner]
    eng.storage.upsert_group("100", enabled=True, extra={"approved": True})
    s = GroupSettings()
    s.apply("审核", "开")
    for key, value in settings.items():
        ok, why = s.apply(key, value)
        assert ok, f"{key}: {why}"
    eng.storage.save_group_settings("100", s)
    return eng


def _msg(eng, text, user="30003", mentioned=False, is_command=False, group_id="100"):
    return eng.handle(IncomingMessage(user_id=user, text=text, is_private=False,
                                      group_id=group_id, user_name="群友",
                                      mentioned=mentioned, is_command=is_command))


# ======================================================================
# 一、判定层：不该处罚的绝不能处罚
# ======================================================================
class NeverPunishTest(unittest.TestCase):
    """V5 明确：不得因为用户批评机器人而处罚用户。

    这是一组**保护性**用例：任何人改动规则时，这些必须先过。
    """

    CRITICISM = [
        "爱丽丝你好蠢啊",
        "这个机器人不太好用",
        "你这个bot真垃圾",
        "AI就是笨",
        "我觉得爱丽丝回答得不对",
        "你这功能有问题吧",
        "为什么你总是答非所问",
        "请问这个怎么用",
        "求助，打不过这个boss",
        "我们开黑吧",
        "今天天气不错",
        "官方公告说今天更新维护",
        "这个角色的剧情我很喜欢",
        "推荐一下练哪个角色",
        "群里有大佬带一下吗",
        "我的号怎么登不上去了",       # 含"号"，但不是卖号
        "这游戏抽卡概率是不是有问题",
        "谁有攻略链接发一下",          # 含"链接"
    ]

    def test_none_of_these_are_flagged(self):
        for text in self.CRITICISM:
            with self.subTest(text=text):
                verdict = mod.judge_ad(text)
                self.assertFalse(verdict.hit, f"被误判为广告：{text} → {verdict.reason}")

    def test_criticism_helper_detects_criticism(self):
        self.assertTrue(mod.is_never_punish("爱丽丝你好蠢"))
        self.assertTrue(mod.is_never_punish("这个机器人不太好用"))
        self.assertFalse(mod.is_never_punish("加群 12345678"))


class AdDetectionTest(unittest.TestCase):
    """真正该抓的广告要抓住。"""

    def test_invite_and_contact(self):
        cases = [
            ("加群 12345678 免费领取礼包", mod.CONF_HIGH),
            ("群号：987654321 速来", mod.CONF_HIGH),
            ("加微信 abc12345 详聊", mod.CONF_HIGH),
            ("代练低价出，加我微信 xyz789", mod.CONF_HIGH),
        ]
        for text, floor in cases:
            with self.subTest(text=text):
                verdict = mod.judge_ad(text)
                self.assertTrue(verdict.hit, text)
                self.assertEqual(verdict.confidence, floor, text)

    def test_promo_with_contact_is_medium(self):
        verdict = mod.judge_ad("卖号 私聊我")
        self.assertTrue(verdict.hit)
        self.assertEqual(verdict.confidence, mod.CONF_MEDIUM)
        self.assertEqual(verdict.rule, "contact")

    def test_promo_alone_is_only_low(self):
        """光是"卖号"两个字不足以定罪——可能只是在吐槽自己的号。"""
        verdict = mod.judge_ad("卖号")
        self.assertTrue(verdict.hit)
        self.assertEqual(verdict.confidence, mod.CONF_LOW)
        self.assertEqual(verdict.suggested, mod.ACTION_NONE)

    def test_normal_context_downgrades_invite(self):
        """带正常上下文词的邀请信息降级为只提示，不自动处罚。"""
        verdict = mod.judge_ad("官方公告里的活动群号：12345678")
        self.assertTrue(verdict.hit)
        self.assertEqual(verdict.confidence, mod.CONF_LOW)

    def test_ecommerce_link_with_promo_tone(self):
        verdict = mod.judge_ad("https://taobao.com/xyz 限时优惠下单")
        self.assertTrue(verdict.hit)
        self.assertIn(verdict.confidence, (mod.CONF_MEDIUM, mod.CONF_HIGH))

    def test_zero_width_evasion_is_normalized(self):
        """广告常用零宽字符规避：匹配前要清理，但证据展示用原文。"""
        verdict = mod.judge_ad("加\u200b群 12345678")
        self.assertTrue(verdict.hit)

    def test_empty_and_whitespace(self):
        self.assertFalse(mod.judge_ad("").hit)
        self.assertFalse(mod.judge_ad("   ").hit)
        self.assertFalse(mod.judge_ad("\u200b\u200b").hit)


class RepeatSpamTest(unittest.TestCase):
    def test_repeated_message_flagged(self):
        verdict = mod.judge_ad("测试消息", history=["测试消息", "测试消息"])
        self.assertTrue(verdict.hit)
        self.assertEqual(verdict.rule, "repeat")

    def test_different_messages_not_flagged(self):
        verdict = mod.judge_ad("测试消息", history=["别的消息", "又一条"])
        self.assertFalse(verdict.hit)

    def test_threshold_is_respected(self):
        rules = mod.ModerationRules(repeat_threshold=5)
        verdict = mod.judge_ad("x", history=["x", "x"], rules=rules)
        self.assertFalse(verdict.hit)


class RuleConfigTest(unittest.TestCase):
    def test_ban_words(self):
        rules = mod.ModerationRules(ban_words=["刷屏怪"])
        verdict = mod.judge_ad("刷屏怪又来了", rules=rules)
        self.assertTrue(verdict.hit)
        self.assertEqual(verdict.rule, "ban_word")

    def test_allow_words_win(self):
        rules = mod.ModerationRules(ban_words=["代练"], allow_words=["代练"])
        self.assertFalse(mod.judge_ad("我们代练一下", rules=rules).hit)

    def test_build_rules_splits_word_lists(self):
        """回归：build_rules 曾经把「词1,词2」整行字符串当列表，
        于是退化成逐字匹配，任何消息都能命中。"""
        s = GroupSettings()
        s.apply("敏感词", "刷屏怪,卖号")
        s.apply("白名单", "副本")
        rules = mod.build_rules(s)
        self.assertEqual(rules.ban_words, ["刷屏怪", "卖号"])
        self.assertEqual(rules.allow_words, ["副本"])
        # 逐字匹配的话"卖"也会命中这句正常聊天
        self.assertFalse(mod.judge_ad("今天副本打完了", rules=rules).hit)

    def test_build_rules_defaults_for_missing_settings(self):
        rules = mod.build_rules(None)
        self.assertTrue(rules.enabled)
        self.assertTrue(rules.dry_run, "没配置过的群应当默认只观察")

    def test_dry_run_off_is_stored_as_false(self):
        """回归：`/群设置 审核试运行 关` 必须真的存成 False。

        早期 apply() 把"关"存成 None（=跟随默认），而默认是"开"，
        于是怎么关都关不掉，表现为"开了审核但永远只有观察、不会真警告"。
        """
        s = GroupSettings()
        s.apply("审核试运行", "关")
        self.assertIs(s.moderation_dry_run, False)
        self.assertFalse(mod.build_rules(s).dry_run, "关掉试运行就该真的执行警告")
        # 「默认」才应该回到跟随全局（= True）
        s.apply("审核试运行", "默认")
        self.assertIsNone(s.moderation_dry_run)
        self.assertTrue(mod.build_rules(s).dry_run)

    def test_confirmation_policy(self):
        rules = mod.ModerationRules()
        for action in (mod.ACTION_MUTE, mod.ACTION_KICK, mod.ACTION_RECALL):
            self.assertTrue(mod.needs_confirmation(action, rules), action)
        self.assertFalse(mod.needs_confirmation(mod.ACTION_WARN, rules))

    def test_escalation_never_auto_kicks(self):
        rules = mod.ModerationRules(warn_threshold=3)
        self.assertEqual(mod.escalate_action(1, rules), mod.ACTION_WARN)
        self.assertEqual(mod.escalate_action(3, rules), mod.ACTION_MUTE)
        # 踢出是不可逆的，永远不自动升级到它
        self.assertNotEqual(mod.escalate_action(99, rules), mod.ACTION_KICK)


# ======================================================================
# 二、引擎侧：试运行 / 真警告 / 阈值升级
# ======================================================================
class ModerationFlowTest(unittest.TestCase):
    AD = "加群 12345678 免费领取礼包"

    def test_dry_run_records_but_does_not_warn(self):
        eng = _group_engine(审核试运行="开")
        out = _msg(eng, self.AD)
        self.assertEqual(out.decision, "moderation_review")
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)
        logs = eng.storage.recent_moderation("100")
        self.assertTrue(any(row["result"] == "skipped" for row in logs))

    def test_real_mode_warns(self):
        eng = _group_engine(审核试运行="关")
        out = _msg(eng, self.AD)
        self.assertEqual(out.decision, "moderation_warn")
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 1)

    def test_threshold_escalates_to_pending_mute_not_auto_mute(self):
        eng = _group_engine(审核试运行="关", 警告上限="2")
        _msg(eng, self.AD)
        out = _msg(eng, "代练低价出，加我微信 abc123")   # 第 2 次，达阈值
        self.assertEqual(out.decision, "moderation_pending")
        self.assertIn("确认", out.reply)
        pending = eng.storage.get_pending_action("100")
        self.assertIsNotNone(pending, "禁言必须进待确认队列，不能自动执行")
        self.assertEqual(pending["action"], mod.ACTION_MUTE)

    def test_warning_counts_are_per_group(self):
        eng = _group_engine(审核试运行="关")
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        _msg(eng, self.AD)
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 1)
        self.assertEqual(eng.storage.get_warning_count("200", "30003"), 0)

    def test_warning_counts_are_per_user(self):
        eng = _group_engine(审核试运行="关")
        _msg(eng, self.AD, user="30003")
        _msg(eng, self.AD, user="30004")
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 1)
        self.assertEqual(eng.storage.get_warning_count("100", "30004"), 1)

    def test_moderation_disabled_does_nothing(self):
        eng = _group_engine(审核="关", 审核试运行="关")
        out = _msg(eng, self.AD)
        self.assertNotIn("moderation", out.decision)
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)

    def test_admins_are_exempt(self):
        """群主/管理员自己发的消息不审——否则管理员发个群号就被自己罚了。"""
        eng = _group_engine(owner="10001", 审核试运行="关")
        out = _msg(eng, self.AD, user="10001")
        self.assertNotIn("moderation", out.decision)
        self.assertEqual(eng.storage.get_warning_count("100", "10001"), 0)

    def test_criticism_never_creates_warning(self):
        eng = _group_engine(审核试运行="关")
        for text in ("爱丽丝你好蠢", "这个机器人真难用", "我觉得你答得不对"):
            _msg(eng, text)
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)
        # 也不应该留下"审核观察"的日志
        logs = [r for r in eng.storage.recent_moderation("100") if r["action"] == "review"]
        self.assertEqual(logs, [])

    def test_low_confidence_never_auto_acts(self):
        eng = _group_engine(审核试运行="关")
        out = _msg(eng, "我想卖号了")          # promo_weak -> low
        self.assertNotEqual(out.decision, "moderation_warn")
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)

    def test_private_chat_is_not_moderated(self):
        eng = _group_engine(审核试运行="关")
        out = eng.handle(IncomingMessage(user_id="30003", text=self.AD, is_private=True,
                                         user_name="群友"))
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)
        self.assertNotIn("moderation", out.decision)

    def test_auto_warn_off_records_only(self):
        eng = _group_engine(审核试运行="关", 自动警告="关")
        out = _msg(eng, self.AD)
        self.assertEqual(out.decision, "moderation_review")
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)

    def test_repeat_spam_through_engine(self):
        """重复刷屏要能通过引擎真的触发（历史来自数据库）。"""
        eng = _group_engine(审核试运行="关")
        for _ in range(3):
            _msg(eng, "重复的一句话")
        self.assertGreaterEqual(eng.storage.get_warning_count("100", "30003"), 1)


# ======================================================================
# 三、管理员命令与确认流程
# ======================================================================
class ModerationCommandTest(unittest.TestCase):
    def test_review_status_requires_admin(self):
        eng = _group_engine(owner="10001")
        out = _msg(eng, "/审核", user="30003", is_command=True)
        self.assertIn("权限", out.reply)

    def test_review_status_shows_state(self):
        eng = _group_engine(owner="10001", 审核试运行="关", 警告上限="3", 敏感词="刷屏怪")
        out = _msg(eng, "/审核", user="10001", is_command=True)
        self.assertIn("审核", out.reply)
        self.assertIn("刷屏怪", out.reply)

    def test_manual_warn_requires_admin(self):
        eng = _group_engine(owner="10001")
        out = _msg(eng, "/警告 30003 刷屏", user="30003", is_command=True)
        self.assertIn("权限", out.reply)
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 0)

    def test_manual_warn_works(self):
        eng = _group_engine(owner="10001")
        out = _msg(eng, "/警告 30003 刷屏", user="10001", is_command=True)
        self.assertIn("已警告", out.reply)
        self.assertEqual(eng.storage.get_warning_count("100", "30003"), 1)

    def test_cannot_warn_self(self):
        eng = _group_engine(owner="10001")
        out = _msg(eng, "/警告 10001 手滑", user="10001", is_command=True)
        self.assertIn("自己", out.reply)

    def test_manual_warn_usage_hint(self):
        eng = _group_engine(owner="10001")
        self.assertIn("用法", _msg(eng, "/警告", user="10001", is_command=True).reply)

    def test_confirm_executes_and_returns_action_request(self):
        eng = _group_engine(owner="10001")
        eng.storage.create_pending_action("100", mod.ACTION_MUTE, target_id="30003",
                                         payload={"seconds": 600}, reason="刷屏",
                                         requested_by="10001")
        out = _msg(eng, "/确认", user="10001", is_command=True)
        request = out.debug.get("action")
        self.assertIsNotNone(request)
        self.assertEqual(request["action"], mod.ACTION_MUTE)
        self.assertEqual(request["target_id"], "30003")
        self.assertEqual(request["payload"]["seconds"], 600)
        # 执行后不该再有待确认
        self.assertIsNone(eng.storage.get_pending_action("100"))

    def test_cancel_discards(self):
        eng = _group_engine(owner="10001")
        eng.storage.create_pending_action("100", mod.ACTION_KICK, target_id="30003",
                                         reason="广告", requested_by="10001")
        out = _msg(eng, "/取消", user="10001", is_command=True)
        self.assertIn("取消", out.reply)
        self.assertIsNone(eng.storage.get_pending_action("100"))

    def test_confirm_without_pending(self):
        eng = _group_engine(owner="10001")
        self.assertIn("没有", _msg(eng, "/确认", user="10001", is_command=True).reply)

    def test_confirm_requires_admin(self):
        eng = _group_engine(owner="10001")
        eng.storage.create_pending_action("100", mod.ACTION_MUTE, target_id="30003")
        out = _msg(eng, "/确认", user="30003", is_command=True)
        self.assertIn("权限", out.reply)
        self.assertIsNotNone(eng.storage.get_pending_action("100"), "不该被非管理员消耗掉")

    def test_pending_is_group_scoped(self):
        eng = _group_engine(owner="10001")
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        eng.storage.create_pending_action("100", mod.ACTION_MUTE, target_id="30003")
        # 在 200 群用管理员身份确认：不该看到 100 群的待确认
        out = _msg(eng, "/确认", user="10001", is_command=True, group_id="200")
        self.assertIn("没有", out.reply)
        self.assertIsNotNone(eng.storage.get_pending_action("100"))

    def test_queue_action_needs_confirmation_for_high_risk(self):
        eng = _group_engine(owner="10001")
        eng._current_settings = None
        out = eng.queue_action("100", mod.ACTION_KICK, target_id="30003",
                               reason="广告", operator_id="10001")
        self.assertEqual(out.decision, "moderation_pending")
        self.assertIsNotNone(eng.storage.get_pending_action("100"))

    def test_queue_action_executes_low_risk_directly(self):
        """警告是低风险动作：直接产出动作请求，不进待确认队列。"""
        eng = _group_engine(owner="10001")
        eng._current_settings = None
        out = eng.queue_action("100", mod.ACTION_WARN, target_id="30003",
                               reason="刷屏", operator_id="10001")
        self.assertIsNone(eng.storage.get_pending_action("100"),
                          "警告不该需要确认")
        self.assertEqual(out.debug.get("action", {}).get("action"), mod.ACTION_WARN)
        self.assertEqual(out.debug.get("action", {}).get("target_id"), "30003")


class ActionResultTest(unittest.TestCase):
    """V5：不得伪造成功——执行结果必须如实回执。"""

    def test_success_is_reported(self):
        eng = _group_engine(owner="10001")
        text = eng.record_action_result("100", mod.ACTION_MUTE, True, target_id="30003")
        self.assertIn("已完成", text)
        self.assertIn("禁言", text)

    def test_failure_states_the_reason(self):
        eng = _group_engine(owner="10001")
        text = eng.record_action_result("100", mod.ACTION_MUTE, False, target_id="30003",
                                        detail="机器人不是管理员")
        self.assertIn("失败", text)
        self.assertIn("机器人不是管理员", text)

    def test_failure_without_detail_still_honest(self):
        eng = _group_engine(owner="10001")
        text = eng.record_action_result("100", mod.ACTION_KICK, False, target_id="30003")
        self.assertIn("失败", text)

    def test_both_outcomes_are_audited(self):
        eng = _group_engine(owner="10001")
        eng.record_action_result("100", mod.ACTION_MUTE, True, target_id="a")
        eng.record_action_result("100", mod.ACTION_MUTE, False, target_id="b", detail="x")
        logs = eng.storage.recent_moderation("100")
        results = {row["target_id"]: row["result"] for row in logs}
        self.assertEqual(results.get("a"), "ok")
        self.assertEqual(results.get("b"), "failed", "失败也必须留痕")


class GroupConfigViewTest(unittest.TestCase):
    """管理员要能从 /群配置 一眼看到审核状态，否则没法判断规则生效没有。"""

    def test_describe_shows_moderation(self):
        s = GroupSettings()
        s.apply("审核试运行", "关")
        s.apply("敏感词", "刷屏怪")
        s.apply("警告上限", "3")
        text = s.describe(AppConfig())
        for label in ("审核", "审核试运行", "自动警告", "升级禁言阈值"):
            self.assertIn(label, text, label)
        self.assertIn("刷屏怪", text)

    def test_describe_defaults_read_as_on(self):
        """默认即为"开"的项，None 也要显示成"开"，不能显示成空白让管理员困惑。"""
        text = GroupSettings().describe(AppConfig())
        self.assertIn("审核：开", text)
        self.assertIn("自动警告：开", text)

    def test_group_config_command_shows_moderation(self):
        eng = _group_engine(owner="10001", 审核试运行="关", 敏感词="广告狗")
        out = _msg(eng, "/群配置", user="30003", is_command=True)
        self.assertIn("审核", out.reply)
        self.assertIn("广告狗", out.reply)


class GroupKeywordTest(unittest.TestCase):
    """V5：群主可配触发关键词（在全局之外**追加**）。"""

    def _engine(self, group="100"):
        eng = make_engine()
        eng.config.bot.owner_ids = ["10001"]
        eng.storage.upsert_group(group, enabled=True, extra={"approved": True})
        return eng

    def _send(self, eng, text, user="30003", group="100", at=None):
        return eng.handle(IncomingMessage(user_id=user, text=text, is_private=False,
                                          group_id=group, user_name="群友",
                                          mentioned=False, is_command=text.startswith("/"),
                                          timestamp=at))

    def test_group_keyword_triggers_reply(self):
        eng = self._engine()
        self._send(eng, "/群设置 关键词 原神", user="10001")
        out = self._send(eng, "原神好玩吗")
        self.assertEqual(out.decision, "replied")
        self.assertIn("keyword", out.debug.get("trigger") or [])

    def test_group_keyword_is_group_scoped(self):
        eng = self._engine()
        eng.storage.upsert_group("200", enabled=True, extra={"approved": True})
        self._send(eng, "/群设置 关键词 原神", user="10001")
        out = self._send(eng, "原神好玩吗", group="200")
        self.assertEqual(out.decision, "silent", "另一个群不该受本群关键词影响")

    def test_keywords_are_added_not_replaced(self):
        """追加语义：设了本群关键词之后，全局关键词必须仍然有效。"""
        eng = self._engine("300")
        self._send(eng, "/群设置 关键词 原神", user="10001", group="300")
        out = self._send(eng, "爱丽丝你好", group="300")
        self.assertEqual(out.decision, "replied", "全局关键词被覆盖了")

    def test_keyword_list_is_split(self):
        s = GroupSettings()
        s.apply("关键词", "原神，星穹铁道, 崩坏")
        self.assertEqual(s.keyword_list(), ["原神", "星穹铁道", "崩坏"])

    def test_clearing_keywords(self):
        s = GroupSettings()
        s.apply("关键词", "原神")
        s.apply("关键词", "关")
        self.assertIsNone(s.group_keywords)
        self.assertEqual(s.keyword_list(), [])


class HelpTextV5Test(unittest.TestCase):
    def test_help_lists_moderation_commands(self):
        text = make_engine().help_text()
        for cmd in ("/审核", "/警告", "/确认", "/取消"):
            self.assertIn(cmd, text, cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
