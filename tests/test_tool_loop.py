# -*- coding: utf-8 -*-
"""多轮工具编排（ReAct 循环）测试。

**全部离线**：用脚本化的假 LLM，按调用次序返回预设的工具调用或文本。

以前这里只有一个来回：执行工具 -> 回灌 -> 再问一次。
现在改成真正的循环，模型可以连续调多个工具。
这个文件守着循环的四条边界：

  1. 能连续调多个工具（这是"Agent"和"带工具的聊天"的区别）
  2. 有轮数上限，撞上限要如实说明而不是假装答完
  3. 完全相同的调用不重复执行（省额度、也避免模型绕圈）
  4. 中途出错不再被静默吞掉
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq.llm import LLMError, Reply
from aliceqq.engine import IncomingMessage, MAX_TOOL_ROUNDS
from tests.test_engine import make_engine


def _tool_call(name, args=None, call_id=None):
    return {"id": call_id or ("call_" + name), "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args or {}, ensure_ascii=False)}}


class ScriptedLLM:
    """按脚本依次返回回复。脚本用尽后重复最后一条。

    记录每次调用收到的 messages，便于断言 role=tool 回灌是否正确。
    """

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, *, config, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[idx]
        if isinstance(item, Exception):
            raise item
        return item

    # 断言辅助：某一轮里出现了哪些 role
    def roles_of_call(self, i):
        return [m.get("role") for m in self.calls[i]["messages"]]


def _text(t):
    return Reply(t, provider="fake", model="fake")


def _calls(*calls):
    return Reply("", provider="fake", model="fake", tool_calls=list(calls))


class MultiRoundTest(unittest.TestCase):
    def _engine(self, script):
        engine = make_engine()
        engine.llm = ScriptedLLM(script)
        return engine

    def test_model_can_call_two_tools_in_sequence(self):
        # 第 1 次：要 remember；第 2 次：要 web_search；第 3 次：给结论
        engine = self._engine([
            _calls(_tool_call("remember", {"content": "喜欢猫"})),
            _calls(_tool_call("academic_search", {"query": "x"})),
            _text("两件事都办好了。"),
        ])
        msg = IncomingMessage(user_id="10001", text="记住我喜欢猫，再帮我查一篇论文",
                              is_private=True, user_name="x")
        outcome = engine.handle(msg)
        self.assertEqual(outcome.debug["tool_rounds"], 2,
                         "应当跑了两轮工具循环")
        self.assertEqual(outcome.debug["tools_used"], ["remember", "academic_search"])
        self.assertIn("两件事都办好了", outcome.reply)

    def test_second_round_messages_carry_the_first_tool_result(self):
        engine = self._engine([
            _calls(_tool_call("remember", {"content": "喜欢猫"})),
            _text("好了。"),
        ])
        engine.handle(IncomingMessage(user_id="10001", text="记住我喜欢猫",
                                      is_private=True, user_name="x"))
        # 第 2 次调用（第二轮的请求）里应当能看到 role=assistant 和 role=tool
        roles = engine.llm.roles_of_call(1)
        self.assertIn("tool", roles)
        self.assertIn("assistant", roles)

    def test_no_tool_call_means_no_loop(self):
        engine = self._engine([_text("直接回答。")])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="你好",
                                                is_private=True, user_name="x"))
        self.assertEqual(outcome.debug.get("tool_rounds", 0), 0)
        self.assertEqual(len(engine.llm.calls), 1, "没调工具就不该多问一次")

    def test_loop_stops_when_model_produces_text(self):
        # 第 2 次给了文字就不再继续
        engine = self._engine([
            _calls(_tool_call("remember", {"content": "a"})),
            _text("完成。"),
        ])
        engine.handle(IncomingMessage(user_id="10001", text="记住 a",
                                      is_private=True, user_name="x"))
        self.assertEqual(len(engine.llm.calls), 2)

    def test_multiple_tools_in_one_round_are_all_executed(self):
        # 模型一轮里可以同时要两个工具
        engine = self._engine([
            _calls(_tool_call("remember", {"content": "a"}, "c1"),
                   _tool_call("remember", {"content": "b"}, "c2")),
            _text("都记下了。"),
        ])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 a 和 b",
                                                is_private=True, user_name="x"))
        self.assertEqual(outcome.debug["tool_calls_total"], 2)
        self.assertEqual(outcome.debug["tool_rounds"], 1)


class RoundLimitTest(unittest.TestCase):
    """轮数上限。模型可能一直调工具不出结论。"""

    def test_endless_tool_calls_are_capped(self):
        # 每次都要工具，永远不给文字。
        # 模型调用次数 = 1（初始那次）+ limit（每轮循环一次）。
        # 初始那次在外面，因为要先问一次才知道模型想不想调工具。
        engine = make_engine()
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"}))])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        limit = engine._tool_rounds_limit()
        self.assertEqual(outcome.debug["tool_rounds"], limit)
        self.assertEqual(len(engine.llm.calls), limit + 1,
                         "不该超过上限（含循环外那一次初始调用）")

    def test_hitting_the_limit_is_reported_not_hidden(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"}))])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertIn("tool_round_limit", outcome.debug,
                      "撞上限应当记进 debug，而不是无声结束")
        self.assertTrue(outcome.reply, "撞上限也必须给出回复")

    def test_limit_is_configurable_and_clamped(self):
        engine = make_engine()
        engine.config.llm.max_tool_rounds = 1
        self.assertEqual(engine._tool_rounds_limit(), 1)
        engine.config.llm.max_tool_rounds = 99
        self.assertLessEqual(engine._tool_rounds_limit(), 6,
                             "上限要被夹住，不能让配置写出无限循环")
        engine.config.llm.max_tool_rounds = 0
        self.assertGreaterEqual(engine._tool_rounds_limit(), 1)

    def test_unexecuted_tool_calls_are_cleared(self):
        # 撞上限时模型还想调工具，那些**没执行**的不能留在 reply 里
        # 让上层以为执行过了
        engine = make_engine()
        engine.config.llm.max_tool_rounds = 1
        engine.llm = ScriptedLLM([_calls(_tool_call("remember", {"content": "x"}))])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertEqual(outcome.debug["tool_rounds"], 1)
        self.assertIn("tool_round_limit", outcome.debug)

    def test_default_limit_is_small(self):
        # 默认值不该太大：多轮意味着多次网络往返，QQ 场景等不起
        self.assertLessEqual(MAX_TOOL_ROUNDS, 4)


class DuplicateCallTest(unittest.TestCase):
    """完全相同的调用不重复执行。"""

    def test_identical_call_is_not_executed_twice(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "猫"}, "c1")),
            _calls(_tool_call("remember", {"content": "猫"}, "c2")),
            _text("好了。"),
        ])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住我喜欢猫",
                                                is_private=True, user_name="x"))
        # 两次调用都出现在结果里（第二次带说明），但记忆只该写一条。
        #
        # 这里其实同时守两件事：去重（第二次没执行）**和**
        # 启发式让路（`_remember_from_message` 看到工具写过了就不写）。
        # 两条都坏了才会变成 3 条，只坏一条是 2 条——所以再断言一次
        # 记忆来源，确保留下的是工具写的那条。
        self.assertEqual(engine.storage.count_memory("10001"), 1,
                         "同一个调用被执行了两次，会重复写入")
        facts = engine.storage.list_memory("10001")
        self.assertEqual(facts[0].source, "tool",
                         "应当保留工具写入的那条（它比启发式切出来的更准）")

    def test_duplicate_is_reported_to_the_model(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "猫"}, "c1")),
            _calls(_tool_call("remember", {"content": "猫"}, "c2")),
            _text("好了。"),
        ])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住我喜欢猫",
                                                is_private=True, user_name="x"))
        messages = [r["message"] for r in outcome.tool_results]
        self.assertTrue(any("已经执行过" in m for m in messages),
                        "重复调用要明确告诉模型，否则它会以为是失败而重试")

    def test_same_tool_with_different_args_is_allowed(self):
        # 带不同参数不算重复：先记 A 再记 B 是正当用法
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "猫"}, "c1")),
            _calls(_tool_call("remember", {"content": "狗"}, "c2")),
            _text("都记下了。"),
        ])
        engine.handle(IncomingMessage(user_id="10001", text="记住猫和狗",
                                      is_private=True, user_name="x"))
        # 只该是工具写的两条；启发式那条要让路
        self.assertEqual(engine.storage.count_memory("10001"), 2)
        self.assertTrue(all(f.source == "tool"
                            for f in engine.storage.list_memory("10001")),
                        "有非工具来源的记忆混进来了，说明启发式没有让路")


class LoopErrorTest(unittest.TestCase):
    """中途出错。以前是 `except LLMError: pass`，完全无声。"""

    def test_llm_error_during_loop_is_reported(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "x"})),
            LLMError("第二轮回灌失败"),
        ])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertIn("tool_loop_error", outcome.debug,
                      "循环中的模型失败必须留下痕迹，不能静默")
        self.assertIn("第二轮回灌失败", outcome.debug["tool_loop_error"])

    def test_reply_still_produced_after_loop_error(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "x"})),
            LLMError("boom"),
        ])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertTrue(outcome.reply, "出错也要给用户一个交代")
        # 不吹成"已经处理了"——工具确实执行了，但模型没能组织回答
        self.assertTrue(outcome.reply.strip())


class ToolExceptionTest(unittest.TestCase):
    def test_tool_exception_does_not_break_the_turn(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("remember", {"content": "x"})),
            _text("好了。"),
        ])

        def boom(*a, **k):
            raise RuntimeError("工具炸了")

        engine._execute_tool = boom
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertTrue(outcome.reply)
        self.assertFalse(outcome.tool_results[0]["ok"])
        self.assertIn("工具炸了", outcome.tool_results[0]["message"])


class MalformedArgumentsTest(unittest.TestCase):
    def test_bad_json_arguments_do_not_break_the_loop(self):
        engine = make_engine()
        bad = {"id": "c1", "type": "function",
               "function": {"name": "remember", "arguments": "{不是合法 json"}}
        engine.llm = ScriptedLLM([_calls(bad), _text("好了。")])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="记住 x",
                                                is_private=True, user_name="x"))
        self.assertTrue(outcome.reply)
        self.assertEqual(len(outcome.tool_results), 1)

    def test_unknown_tool_name_is_handled(self):
        engine = make_engine()
        engine.llm = ScriptedLLM([
            _calls(_tool_call("nonexistent_tool", {"x": 1})), _text("好了。")])
        outcome = engine.handle(IncomingMessage(user_id="10001", text="干点什么",
                                                is_private=True, user_name="x"))
        self.assertTrue(outcome.reply)
        self.assertFalse(outcome.tool_results[0]["ok"])


if __name__ == "__main__":
    unittest.main()
