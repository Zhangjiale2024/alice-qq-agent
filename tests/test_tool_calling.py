"""工具调用（function calling）的测试。

为什么专门加这个文件：项目里工具调用曾经**静默失效**过很长一段时间——
`tools.py` 的注册表、JSON Schema、`engine` 里"执行工具 + role=tool 回灌"
的循环**全都写好了**，`llm.py` 也支持 `tools=` 参数并会解析 `tcool_calls`，
但 `engine.generate_reply()` 调用 `llm.chat()` 时**从来没有把 schema 传进去**。

于是 `reply.tool_calls` 恒为空、回灌循环永远不执行，
模型只能在提示词里"看到"工具目录、无法真正发起调用——
**而且没有任何报错**，测试也全绿。

这段代码存在的意义就是卡死这个接线：
  1. schema 必须真的传给模型（不是只在提示词里描述）；
  2. 不可用的工具**不能**出现在 schema 里（递刀子问题）；
  3. 模型发起调用后，结果必须以 role=tool 回灌并触发二次生成。
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from aliceqq import tools
from aliceqq.config import AppConfig
from aliceqq.engine import Engine, IncomingMessage
from aliceqq.llm import Reply
from tests.test_engine import make_engine, private


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()

    def test_schema_is_openai_compatible(self):
        """必须是合法的 OpenAI tools 格式，否则接口会直接报 400。"""
        schemas = tools.openai_tool_schemas(self.config, search_available=True,
                                            academic_available=True)
        self.assertTrue(schemas, "至少要暴露一个工具")
        for item in schemas:
            self.assertEqual(item.get("type"), "function")
            fn = item["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertEqual(fn["parameters"].get("type"), "object")
            self.assertIn("properties", fn["parameters"])
            # 必须是可 JSON 序列化的（要放进请求体）
            json.dumps(item)

    def test_remember_available_when_memory_on(self):
        schemas = tools.openai_tool_schemas(self.config)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("remember", names)

    def test_unavailable_tools_are_not_offered(self):
        """不可用的工具不能进 schema。

        这是刻意的设计：把不可用的工具递给模型等于给它递刀子——
        它会尝试调用、拿到失败结果，或者在回复里演一遍"我去查一下"。
        提示词里仍会列出它们并标注"不能尝试调用"，两条信息一致。
        """
        schemas = tools.openai_tool_schemas(self.config, search_available=False,
                                            academic_available=False)
        names = [s["function"]["name"] for s in schemas]
        self.assertNotIn("web_search", names)
        self.assertNotIn("academic_search", names)

    def test_available_search_is_offered(self):
        schemas = tools.openai_tool_schemas(self.config, search_available=True,
                                            academic_available=True)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("web_search", names)
        self.assertIn("academic_search", names)


class _RecordingLLM:
    """假的 LLM：记录每次调用收到的 tools，并按脚本返回 tool_calls。"""

    name = "fake"

    def __init__(self, first: Reply, second: Reply | None = None):
        self.calls: list[dict] = []
        self._replies = [first, second or Reply("好的。", provider="fake", model="fake")]
        self._index = 0

    def chat(self, messages, *, config, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return reply


def _tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class WiringTest(unittest.TestCase):
    """接线测试：schema 必须真的传进 llm.chat()。"""

    def _engine_with(self, llm) -> Engine:
        engine = make_engine()
        engine.llm = llm
        return engine

    def test_schema_is_passed_to_the_model(self):
        """**这条用例就是为那个静默失效的漏洞写的。**

        漏洞版本：engine 调用 `llm.chat(messages, config=...)` 不传 tools，
        于是模型永远拿不到工具定义，`reply.tool_calls` 恒为空。
        这里直接断言"调用时 tools 非空"。
        """
        llm = _RecordingLLM(Reply("你好呀。", provider="fake", model="fake"))
        engine = self._engine_with(llm)
        private("你好", engine=engine)

        self.assertTrue(llm.calls, "应该至少调用过一次模型")
        tools_arg = llm.calls[0]["tools"]
        self.assertIsNotNone(tools_arg,
                             "engine 必须把 tools 传给 llm.chat()——"
                             "不传的话 tool_calls 永远是空的，整套 function calling 失效")
        names = [t["function"]["name"] for t in tools_arg]
        self.assertIn("remember", names)

    def test_tool_result_is_fed_back_as_role_tool(self):
        """模型发起调用后，结果要以 role=tool 回灌，并触发二次生成。"""
        llm = _RecordingLLM(
            Reply("", provider="fake", model="fake",
                  tool_calls=[_tool_call("remember", {"content": "我不吃香菜"})]),
            Reply("记住了。", provider="fake", model="fake"),
        )
        engine = self._engine_with(llm)
        _, outcome = private("记住我不吃香菜", engine=engine)

        self.assertEqual(len(llm.calls), 2, "工具调用后应该再问一次模型")
        second = llm.calls[1]["messages"]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        self.assertTrue(tool_msgs, "回灌消息里必须有 role=tool 的结果")
        self.assertEqual(tool_msgs[0].get("tool_call_id"), "c1")
        self.assertIn("ok", tool_msgs[0]["content"])
        # 工具真的执行了
        self.assertTrue(outcome.tool_results)
        self.assertEqual(outcome.tool_results[0]["name"], "remember")
        self.assertTrue(outcome.tool_results[0]["ok"])

    def test_second_call_also_carries_tools(self):
        """二次生成也要带 tools —— 模型可能想接着调下一个工具。"""
        llm = _RecordingLLM(
            Reply("", provider="fake", model="fake",
                  tool_calls=[_tool_call("remember", {"content": "x"})]),
            Reply("好。", provider="fake", model="fake"),
        )
        engine = self._engine_with(llm)
        private("记住 x", engine=engine)
        self.assertIsNotNone(llm.calls[1]["tools"],
                             "二次生成不传 tools 会截断多步工具调用")

    def test_unknown_tool_is_rejected_gracefully(self):
        """模型乱调一个不存在的工具，不能崩，要如实返回失败。"""
        llm = _RecordingLLM(
            Reply("", provider="fake", model="fake",
                  tool_calls=[_tool_call("delete_everything", {})]),
            Reply("做不到。", provider="fake", model="fake"),
        )
        engine = self._engine_with(llm)
        _, outcome = private("随便", engine=engine)
        self.assertTrue(outcome.tool_results)
        self.assertFalse(outcome.tool_results[0]["ok"])

    def test_sensitive_content_is_refused_by_tool(self):
        """即使模型坚持调用 remember，敏感信息也必须被服务端拒绝。"""
        llm = _RecordingLLM(
            Reply("", provider="fake", model="fake",
                  tool_calls=[_tool_call("remember",
                                         {"content": "我的身份证号是 123456789012345678"})]),
            Reply("不能记这个。", provider="fake", model="fake"),
        )
        engine = self._engine_with(llm)
        _, outcome = private("记住我的身份证号", engine=engine)
        self.assertTrue(outcome.tool_results)
        self.assertFalse(outcome.tool_results[0]["ok"],
                         "敏感信息必须被拒绝，不能因为模型请求了就保存")


if __name__ == "__main__":
    unittest.main(verbosity=2)
