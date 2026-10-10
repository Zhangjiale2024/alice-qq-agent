"""V1 核心行为测试：使用 mock LLM，完全离线。"""
from __future__ import annotations

import shutil
import time
import unittest
import uuid
from pathlib import Path

from aliceqq.config import AppConfig, PROJECT_ROOT
from aliceqq.db import Storage
from aliceqq.engine import Engine, IncomingMessage, MODE_ALICE, MODE_ASSISTANT

# 注意：不使用系统临时目录——沙箱下 Temp 目录不允许 SQLite 加锁，
# 所以测试库统一放在工作区的 data/test-run/ 下。
_TEST_ROOT = PROJECT_ROOT / "data" / "test-run"

# 每个用例一个库目录，跑得越多留得越多：实测累积到 10486 个目录 / 672MB。
# 所以每次建库前顺手清掉**稍早以前**的（默认 30 分钟），既不会误删
# 正在并发跑的用例（整套测试约 1 分钟），也不会让仓库无限膨胀。
_STALE_SECONDS = 30 * 60


def _prune_stale_test_dirs(now: float | None = None) -> int:
    """删掉过期的测试库目录，返回删除数量。"""
    if not _TEST_ROOT.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - _STALE_SECONDS
    removed = 0
    for entry in _TEST_ROOT.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def make_engine(**overrides) -> Engine:
    # 用 uuid 而不是递增计数器：多进程/多次运行不会撞到同一个库文件
    tmp = _TEST_ROOT / f"case-{uuid.uuid4().hex[:12]}"
    _prune_stale_test_dirs()
    tmp.mkdir(parents=True, exist_ok=True)
    config = AppConfig()
    # 每个用例必须用自己的数据库文件，否则用例之间会互相污染（共享默认 data/alice.db）
    config.bot.db_path = str(tmp / "alice.db")
    config.bot.group_config_path = str(tmp / "groups.json")
    config.bot.owner_ids = ["99999"]
    config.proactive.enabled = False
    # 单元测试必须完全离线且确定性：默认关闭联网检索。
    # 需要测检索的用例自行注入假 provider（见 tests/test_search.py）。
    config.search.enabled = False
    config.features.web_search = False
    for key, value in overrides.items():
        section, _, name = key.partition("__")
        setattr(getattr(config, section), name, value)
    return Engine(config, Storage(config.db_file))


def minute_safe_base() -> float:
    """取"当前分钟 :30"的时间戳。

    限流按 YYYYMMDDHHMM 计数；若基准落在 :59，紧邻的下一句就跨到新的一分钟、
    计数被重置，用例会随运行时刻随机失败。所有涉及限流的用例都应使用它。
    """
    now = time.time()
    return now - (now % 60) + 30


def private(text: str, user: str = "10001", name: str = "小星", engine: Engine | None = None,
            at: float | None = None):
    eng = engine or make_engine()
    return eng, eng.handle(IncomingMessage(user_id=user, text=text, is_private=True,
                                           user_name=name, timestamp=at or time.time()))


def group(text: str, user: str = "20001", gid: str = "88888", mentioned: bool = True,
          engine: Engine | None = None, name: str = "群友", at: float | None = None):
    eng = engine or make_engine()
    eng.storage.upsert_group(gid, enabled=True, extra={"approved": True})
    return eng, eng.handle(IncomingMessage(user_id=user, text=text, is_private=False, group_id=gid,
                                           user_name=name, mentioned=mentioned,
                                           timestamp=at or time.time()))


class ConfigSecrecyTest(unittest.TestCase):
    """配置导出绝不能泄漏任何凭据。

    这条是实测发现的：早期只脱敏了 llm.api_key，
    后来新增的博查/百度/S2 key 会原样出现在 public_dict / to_json 里。
    """

    def _loaded_engine(self, **secrets):
        engine = make_engine()
        engine.config.llm.api_key = secrets.get("llm", "")
        engine.config.search.bocha_api_key = secrets.get("bocha", "")
        engine.config.search.baidu_api_key = secrets.get("baidu", "")
        engine.config.academic.semantic_scholar_api_key = secrets.get("s2", "")
        return engine

    def test_all_secret_fields_are_masked(self):
        engine = self._loaded_engine(llm="sk-llm-secret-1", bocha="sk-bocha-secret-2",
                                     baidu="sk-baidu-secret-3", s2="s2-secret-4")
        dump = engine.config.to_json()
        for secret in ("sk-llm-secret-1", "sk-bocha-secret-2", "sk-baidu-secret-3", "s2-secret-4"):
            self.assertNotIn(secret, dump, f"凭据泄漏到配置导出：{secret}")

    def test_masked_fields_still_marked(self):
        engine = self._loaded_engine(bocha="sk-bocha-secret-2")
        public = engine.config.public_dict()
        self.assertEqual(public["search"]["bocha_api_key"], "***")
        # 未配置的保持空串，不要误标
        self.assertEqual(public["search"]["baidu_api_key"], "")

    def test_live_values_are_unchanged(self):
        # 脱敏只影响导出，不能影响真正用于请求的值
        engine = self._loaded_engine(bocha="sk-bocha-secret-2")
        self.assertEqual(engine.config.search.bocha_api_key, "sk-bocha-secret-2")


class ModeSwitchTest(unittest.TestCase):
    def test_default_mode_is_alice(self):
        engine, outcome = private("你好")
        self.assertEqual(outcome.mode, MODE_ALICE)
        self.assertTrue(outcome.should_send)

    def test_switch_to_assistant_and_back(self):
        engine = make_engine()
        _, first = private("/助手", engine=engine)
        self.assertEqual(first.mode, MODE_ASSISTANT)
        self.assertIn("助手模式", first.reply)
        self.assertEqual(engine.storage.get_profile("10001").mode, MODE_ASSISTANT)

        _, second = private("帮我解释一下递归", engine=engine, at=time.time() + 120)
        self.assertEqual(second.mode, MODE_ASSISTANT)
        self.assertNotIn("爱丽丝在待机中", second.reply or "")

        _, third = private("/爱丽丝", engine=engine, at=time.time() + 240)
        self.assertEqual(third.mode, MODE_ALICE)
        self.assertEqual(engine.storage.get_profile("10001").mode, MODE_ALICE)

    def test_assistant_switch_disabled(self):
        engine = make_engine(bot__assistant_mode_enabled=False)
        _, outcome = private("/助手", engine=engine)
        self.assertEqual(outcome.mode, MODE_ALICE)
        self.assertIn("未启用", outcome.reply)


class MemoryTest(unittest.TestCase):
    def test_explicit_memory_saved_and_reported(self):
        engine, outcome = private("记住我喜欢玩明日方舟，也喜欢猫")
        self.assertTrue(outcome.memory_saved)
        facts = [f.content for f in engine.storage.list_memory("10001")]
        self.assertTrue(any("明日方舟" in f for f in facts), facts)

    def test_nickname_memory_used_in_context(self):
        engine, _ = private("叫我小星就好")
        profile = engine.storage.get_profile("10001")
        self.assertEqual(profile.nickname, "小星")
        self.assertEqual(profile.preferred_name, "小星")

    def test_sensitive_content_is_rejected(self):
        engine, outcome = private("记住我的银行卡密码是 123456")
        facts = [f.content for f in engine.storage.list_memory("10001")]
        self.assertFalse(any("123456" in f for f in facts), facts)
        self.assertFalse(outcome.memory_saved)

    def test_memory_is_isolated_per_user(self):
        engine = make_engine()
        private("记住我喜欢猫", user="10001", engine=engine)
        _, view = private("/记忆", user="10002", name="路人", engine=engine, at=time.time() + 120)
        self.assertNotIn("猫", view.reply)
        self.assertEqual(engine.storage.count_memory("10002"), 0)
        self.assertGreaterEqual(engine.storage.count_memory("10001"), 1)

    def test_view_and_delete_memory_commands(self):
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        _, listed = private("/记忆", engine=engine, at=time.time() + 60)
        self.assertIn("猫", listed.reply)
        _, deleted = private("/忘记 猫", engine=engine, at=time.time() + 120)
        self.assertIn("删除", deleted.reply)
        self.assertEqual(engine.storage.count_memory("10001"), 0)
        _, cleared = private("/清空记忆", engine=engine, at=time.time() + 180)
        # 上一步已经把唯一一条删掉了，这里应当是"确认是空的"而不是谎称清空了若干条
        self.assertTrue("清空" in cleared.reply or "空的" in cleared.reply, cleared.reply)

    def test_memory_view_is_not_tagged_as_newly_saved(self):
        # 旧实现会给 /记忆 的回显也加"（已经记下了）"，那是误伤
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        _, listed = private("/记忆", engine=engine, at=time.time() + 60)
        self.assertIn("猫", listed.reply)
        self.assertNotIn("已经记下", listed.reply)
        # 而真正写入的那一轮必须有确认
        _, saved = private("记住我喜欢狗", engine=engine, at=time.time() + 120)
        self.assertIn("已经记下", saved.reply)

    def test_forget_without_keyword_clears_everything(self):
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        private("记住我喜欢打游戏", engine=engine, at=time.time() + 30)
        self.assertGreaterEqual(engine.storage.count_memory("10001"), 2)
        _, cleared = private("/忘记", engine=engine, at=time.time() + 60)
        self.assertIn("清空", cleared.reply)
        self.assertEqual(engine.storage.count_memory("10001"), 0)

    def test_forget_unknown_keyword_reports_nothing_found(self):
        engine = make_engine()
        private("记住我喜欢猫", engine=engine)
        _, outcome = private("/忘记 量子力学", engine=engine, at=time.time() + 60)
        self.assertIn("没有找到", outcome.reply)
        self.assertGreaterEqual(engine.storage.count_memory("10001"), 1)

    def test_memory_view_does_not_leak_across_users(self):
        engine = make_engine()
        private("记住我喜欢猫", user="10001", engine=engine)
        private("/记忆", user="10001", engine=engine, at=time.time() + 30)
        group("爱丽丝 在吗", user="10002", engine=engine, at=time.time() + 60)
        # 群里生成提示词时，memory 里只应该包含 10002 的记忆
        _, messages, variables = engine.build_messages(
            IncomingMessage(user_id="10002", text="爱丽丝", is_private=False, group_id="88888",
                            user_name="群友"), MODE_ALICE)
        self.assertNotIn("猫", variables["user_memory"])


class GroupSceneTest(unittest.TestCase):
    def test_group_silent_without_trigger(self):
        engine, outcome = group("今天天气不错啊", mentioned=False)
        self.assertFalse(outcome.should_send)
        self.assertTrue(outcome.record_only)
        self.assertEqual(engine.storage.get_counter("rl_group", "88888",
                                                    time.strftime("%Y%m%d%H%M")), 0)

    def test_group_replies_when_mentioned(self):
        engine, outcome = group("爱丽丝在吗，推荐个游戏")
        self.assertTrue(outcome.should_send)
        self.assertIn("mention", outcome.debug.get("trigger", []))

    def test_group_replies_on_keyword(self):
        engine, outcome = group("有人玩蔚蓝档案吗", mentioned=False)
        self.assertTrue(outcome.should_send)
        self.assertIn("keyword", outcome.debug.get("trigger", []))

    def test_unauthorized_group_is_denied(self):
        engine = make_engine(group__access="approved_groups")
        outcome = engine.handle(IncomingMessage(user_id="20001", text="爱丽丝", is_private=False,
                                               group_id="77777", mentioned=True))
        self.assertEqual(outcome.decision, "denied")
        self.assertFalse(outcome.should_send)

    def test_approved_group_allowed_but_disabled_group_denied(self):
        engine = make_engine(group__access="approved_groups")
        engine.storage.upsert_group("88888", enabled=True, extra={"approved": True})
        _, ok = group("爱丽丝", engine=engine)
        self.assertTrue(ok.should_send)
        engine.storage.upsert_group("88888", enabled=False)
        outcome = engine.handle(IncomingMessage(user_id="20001", text="爱丽丝", is_private=False,
                                               group_id="88888", mentioned=True,
                                               timestamp=time.time() + 120))
        self.assertEqual(outcome.decision, "denied")

    def test_group_rate_limit(self):
        engine = make_engine(group__rate_limit_per_minute=2)
        base = minute_safe_base()
        for i in range(3):
            _, outcome = group(f"爱丽丝 第{i}条", engine=engine, at=base + i)
        self.assertEqual(outcome.decision, "rate_limited")
        self.assertFalse(outcome.should_send)

    def test_group_reply_is_shorter_than_private(self):
        engine = make_engine(group__max_reply_chars=20)
        _, outcome = group("爱丽丝 讲个很长的故事")
        self.assertLessEqual(len(outcome.reply), 21)


class PrivateLimitTest(unittest.TestCase):
    def test_private_rate_limit(self):
        engine = make_engine(private__rate_limit_per_minute=1)
        base = minute_safe_base()
        _, first = private("你好", engine=engine, at=base)
        self.assertTrue(first.should_send, first.reason)
        _, second = private("再说一句", engine=engine, at=base + 1)
        self.assertEqual(second.decision, "rate_limited")

    def test_quota_exceeded_message(self):
        engine = make_engine(quota__daily_private=1)
        base = minute_safe_base()
        private("你好", engine=engine, at=base)
        _, second = private("还在吗", engine=engine, at=base + 120)
        self.assertEqual(second.decision, "quota_exceeded")
        self.assertIn("额度", second.reply)

    def test_private_disabled(self):
        engine = make_engine(private__enabled=False)
        _, outcome = private("你好", engine=engine)
        self.assertEqual(outcome.decision, "disabled")


class PromptTest(unittest.TestCase):
    def test_current_message_not_duplicated_in_context(self):
        engine = make_engine()
        private("第一句话", engine=engine)
        _, messages, _ = engine.build_messages(
            IncomingMessage(user_id="10001", text="第二句话", is_private=True, user_name="小星"), MODE_ALICE)
        users = [m["content"] for m in messages if m["role"] == "user"]
        self.assertEqual(users.count("第二句话"), 1)
        self.assertNotIn("第二句话", users[:-1])

    def test_all_variables_rendered(self):
        from aliceqq import prompt as prompt_mod
        engine = make_engine()
        _, _, variables = engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="小星"), MODE_ALICE)
        template = engine.template("alice_system.md")
        report = prompt_mod.list_variables(template, variables)
        self.assertEqual(report["missing"], [])

    def test_no_unrendered_placeholder_in_prompt(self):
        engine = make_engine()
        system, _, _ = engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="小星"), MODE_ALICE)
        self.assertNotIn("{{", system)

    def test_speaker_memory_is_injected_in_group_prompt(self):
        # 记忆是按用户走的：本人在群里说话，他自己的记忆应当可见
        engine = make_engine()
        private("记住我住在火星基地第七区", user="10001", engine=engine)
        system, _, variables = engine.build_messages(
            IncomingMessage(user_id="10001", text="爱丽丝", is_private=False, group_id="88888",
                            user_name="小星"), MODE_ALICE)
        self.assertIn("火星基地第七区", variables["user_memory"])
        self.assertIn("群号：88888", system)

    def test_other_user_memory_never_injected(self):
        # 同一个群里，B 说话时绝不能看到 A 的记忆
        engine = make_engine()
        private("记住我住在火星基地第七区", user="10001", engine=engine)
        _, _, variables = engine.build_messages(
            IncomingMessage(user_id="10002", text="爱丽丝", is_private=False, group_id="88888",
                            user_name="路人"), MODE_ALICE)
        self.assertNotIn("火星基地第七区", variables["user_memory"])
        self.assertEqual(variables["user_memory"], "（暂无长期记忆）")

    def test_private_memory_visible_in_private_prompt(self):
        engine = make_engine()
        private("记住我住在火星基地第七区", user="10001", engine=engine)
        _, _, variables = engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="小星"), MODE_ALICE)
        self.assertIn("火星基地第七区", variables["user_memory"])

    def test_group_history_is_not_injected_into_private_prompt(self):
        # 反过来也要成立：群里的发言不能出现在私聊上下文里
        engine = make_engine()
        group("爱丽丝 我在群里聊了火星基地第七区", user="10001", engine=engine)
        _, messages, _ = engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="小星"), MODE_ALICE)
        joined = "\n".join(m.get("content", "") for m in messages)
        self.assertNotIn("我在群里聊了", joined)


class GracefulDegradationTest(unittest.TestCase):
    def test_broken_llm_returns_honest_message(self):
        class BrokenLLM:
            name = "broken"

            def chat(self, messages, *, config, tools=None):
                from aliceqq.llm import LLMError
                raise LLMError("连接超时")

        engine = make_engine()
        engine.llm = BrokenLLM()
        _, outcome = private("你好", engine=engine)
        self.assertEqual(outcome.decision, "error")
        self.assertIn("信号", outcome.reply)

    def test_unknown_command_is_answered(self):
        engine, outcome = private("/不存在的指令")
        self.assertIn("帮助", outcome.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
