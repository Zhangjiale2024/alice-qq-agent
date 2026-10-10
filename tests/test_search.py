"""V2 联网检索测试。

单元测试用假 provider，保证离线且确定性；真实联网只在 live 用例里做，
默认跳过（设置 ALICE_LIVE_SEARCH=1 才跑）。
"""
from __future__ import annotations

import os
import time
import unittest

from aliceqq import honesty
from aliceqq.config import AppConfig
from aliceqq.engine import IncomingMessage, MODE_ALICE, MODE_ASSISTANT
from aliceqq.search import (BingProvider, DuckDuckGoProvider, SearchOutcome, SearchResult,
                            SearchService, _DuckDuckGoParser, _split_bing_blocks, _bing_title,
                            _bing_url, clean_query, disambiguate, has_topic, is_safe_url,
                            needs_search)
from tests.test_engine import make_engine, private

SAMPLE_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fbaike.baidu.com%2Fitem%2F%E5%A4%A9%E7%AB%A5%E7%88%B1%E4%B8%BD%E4%B8%9D">天童爱丽丝 - 百度百科</a>
  <a class="result__snippet" href="#">爱丽丝是游戏开发部在废墟中发现的机器人少女。</a>
</div>
<div class="result">
  <a class="result__a" href="https://bluearchive.fandom.com/wiki/Tendou_Alice">Tendou Alice | Blue Archive Wiki</a>
  <a class="result__snippet" href="#">Alice is a student of Millennium Science School.</a>
</div>
<div class="result">
  <a class="result__a" href="http://127.0.0.1/admin">内网地址不该出现</a>
  <a class="result__snippet" href="#">danger</a>
</div>
</body></html>
"""


class FakeProvider:
    name = "fake"

    def __init__(self, results=None, ok=True, error=""):
        self.results = results if results is not None else [
            SearchResult("天童爱丽丝 - 百度百科", "https://baike.baidu.com/item/x",
                         "爱丽丝是机器人少女", "baike.baidu.com", True),
            SearchResult("Blue Archive Wiki", "https://bluearchive.fandom.com/wiki/Alice",
                         "Alice wields a railgun", "bluearchive.fandom.com", True),
        ]
        self.ok = ok
        self.error = error
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        return SearchOutcome(self.ok, query, self.name, results=self.results, error=self.error)


SAMPLE_BING_HTML = '''
<ol id="b_results">
<li class="b_algo" data-id iid=SERP.5330>
  <h2 class=""><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1aHR0cHM6Ly9iYWlrZS5iYWlkdS5jb20vaXRlbS8lRTUlQTQlQTklRTclQUIlQTUlRTclODglQjElRTQlQjglQkQlRTQlQjglOUQvNjMyODUzMjE&amp;ntb=1">天童爱丽丝 - 百度百科</a></h2>
  <div class="b_caption"><p class="b_lineclamp2">爱丽丝是游戏开发部在废墟中发现的机器人少女。</p></div>
</li>
<li class="b_algo" data-id iid=SERP.5340>
  <h2 class=""><a target="_blank" href="http://127.0.0.1/evil">内网地址</a></h2>
  <div class="b_caption"><p class="b_lineclamp2">danger</p></div>
</li>
<li class="b_algo" data-id iid=SERP.5350>
  <h2 class=""><a target="_blank" href="https://bluearchive.fandom.com/wiki/Tendou_Alice">Tendou Alice | Blue Archive Wiki</a></h2>
  <div class="b_caption"><p class="b_lineclamp2">Alice is a student of Millennium.</p></div>
</li>
</ol>
'''


class BingParsingTest(unittest.TestCase):
    def test_blocks_and_fields(self):
        blocks = _split_bing_blocks(SAMPLE_BING_HTML)
        self.assertEqual(len(blocks), 3)
        self.assertIn("百度百科", _bing_title(blocks[0]))
        # Bing 的跳转链接要还原成真实地址
        self.assertTrue(_bing_url(blocks[0]).startswith("https://baike.baidu.com/item/"))
        self.assertIn("fandom", _bing_url(blocks[2]))

    def test_unwrap_ignores_unsafe_target(self):
        blocks = _split_bing_blocks(SAMPLE_BING_HTML)
        self.assertFalse(is_safe_url(_bing_url(blocks[1])))

    def test_disambiguation_is_off_by_default(self):
        # 360 能正确理解系列名，默认不该改动查询
        self.assertEqual(disambiguate("蔚蓝档案 最新活动"), "蔚蓝档案 最新活动")

    def test_disambiguation_when_enabled(self):
        # 换成会把"蔚蓝档案"拆词匹配的检索源（如 cn.bing.com）时应打开
        from aliceqq import search as search_mod
        search_mod.set_disambiguation(True)
        try:
            self.assertIn("手游", disambiguate("蔚蓝档案 最新活动"))
            self.assertIn("蔚蓝档案 手游", clean_query("蔚蓝 最新活动"))
            self.assertEqual(disambiguate("明日方舟 活动"), "明日方舟 活动")
        finally:
            search_mod.set_disambiguation(False)

    def test_bing_reports_structure_change_instead_of_faking_empty(self):
        empty = BingProvider(host="example.invalid", timeout=1)
        outcome = empty.search("x")
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.error)


class ParsingTest(unittest.TestCase):
    def test_ddg_parser_extracts_results(self):
        parser = _DuckDuckGoParser()
        parser.feed(SAMPLE_DDG_HTML)
        titles = [r["title"] for r in parser.results]
        self.assertIn("天童爱丽丝 - 百度百科", titles)
        self.assertEqual(len(parser.results), 3)

    def test_query_cleaning_strips_commands(self):
        # 消歧默认关闭，所以查询保持原样（360 不需要消歧）
        self.assertEqual(clean_query("爱丽丝帮我查一下今天蔚蓝档案有什么新活动"),
                         "今天蔚蓝档案有什么新活动")
        self.assertEqual(clean_query("搜一下 明日方舟 最新活动"), "明日方舟 最新活动")

    def test_unsafe_urls_rejected(self):
        for url in ("http://127.0.0.1/x", "http://192.168.1.1/", "http://localhost/x",
                    "http://10.0.0.1/", "file:///c:/x", "ftp://example.com/x"):
            self.assertFalse(is_safe_url(url), url)
        self.assertTrue(is_safe_url("https://baike.baidu.com/item/x"))

    def test_has_topic_rejects_vague_queries(self):
        # 实测：这类没有话题词的模糊问题交给搜索引擎，只会返回无关内容
        for text in ("现在游戏里正在开的活动是什么", "最新消息是什么", "帮我查一下",
                     "最近有什么活动", "今天有什么新闻"):
            self.assertFalse(has_topic(text), f"{text!r} 不应被判为有话题")

    def test_has_topic_accepts_named_subjects(self):
        for text in ("蔚蓝档案最近有什么活动", "明日方舟 最新活动", "基沃托斯是什么"):
            self.assertTrue(has_topic(text), f"{text!r} 应当被判为有话题")

    def test_needs_search_intent(self):
        for text in ("今天有什么新活动", "最近有什么新番", "这个多少钱", "帮我查一下最新版本"):
            self.assertTrue(needs_search(text), text)
        for text in ("你好呀", "陪我聊聊天", "今天好累"):
            self.assertFalse(needs_search(text), text)

    def test_format_includes_url_and_source(self):
        service = SearchService(FakeProvider(), fetch_pages=0)
        text = service.format_for_model(service.search("x"))
        self.assertIn("baike.baidu.com", text)
        self.assertIn("https://", text)
        self.assertIn("可信来源", text)


class FreshnessTest(unittest.TestCase):
    """时效判定：把"信息有多旧"算出来告诉模型，而不是让它自己猜。"""

    def test_latest_date_extraction(self):
        from aliceqq.search import _latest_date_in
        self.assertEqual(_latest_date_in("发布于 2026-04-09，另有 2026/03/06 的旧文"),
                         "2026-04-09")
        self.assertEqual(_latest_date_in("2025年12月31日 更新"), "2025-12-31")
        self.assertEqual(_latest_date_in("没有日期"), "")

    def test_staleness_days(self):
        from aliceqq.search import staleness_days
        now = 1759800000.0  # 固定时间，避免用例随时钟漂移
        self.assertGreaterEqual(staleness_days("2026-10-01", now=now), 0)
        self.assertEqual(staleness_days("没有日期", now=now), -1)

    def test_freshest_age_picks_newest_result(self):
        from aliceqq.search import freshest_content_age
        outcome = SearchOutcome(True, "q", "fake", results=[
            SearchResult("旧", "https://a.com", "", "a.com", True, content="2020-01-01 的内容"),
            SearchResult("新", "https://b.com", "", "b.com", True, content="2030-01-01 的内容"),
        ])
        age = freshest_content_age(outcome)
        self.assertGreaterEqual(age, 0)
        # 应该取最新那条（2030-01-01），所以距今不会是"很多年"
        self.assertLess(age, 365 * 20)

    def test_stale_content_gets_warning_in_context(self):
        from aliceqq.search import _format_results
        outcome = SearchOutcome(True, "q", "fake", results=[
            SearchResult("旧活动", "https://a.com", "", "a.com", True,
                         content="2019-01-01 举办的活动"),
        ])
        text = _format_results(outcome)
        self.assertIn("时效提示", text)
        self.assertIn("很可能已经过时", text)

    def test_fresh_content_gets_positive_warning(self):
        from aliceqq.search import _format_results
        import time as _time
        today = _time.strftime("%Y-%m-%d")
        outcome = SearchOutcome(True, "q", "fake", results=[
            SearchResult("今天的公告", "https://a.com", "", "a.com", True,
                         content=f"{today} 发布的公告"),
        ])
        text = _format_results(outcome)
        self.assertIn("时效提示", text)
        self.assertIn("可以作为较新的信息使用", text)

    def test_unknown_date_tells_model_not_to_claim_latest(self):
        from aliceqq.search import _format_results
        outcome = SearchOutcome(True, "q", "fake", results=[
            SearchResult("无日期", "https://a.com", "", "a.com", True, content="没有日期"),
        ])
        text = _format_results(outcome)
        self.assertIn("不要声称这是", text)


class TopicMiningTest(unittest.TestCase):
    """上文话题挖掘——中文没有空格，不能按空白切词。"""

    def test_mines_known_game_name(self):
        from aliceqq.search import mine_topic
        self.assertEqual(mine_topic("我在玩蔚蓝档案，最近有点肝不动了"), "蔚蓝档案")
        self.assertEqual(mine_topic("最近在打明日方舟"), "明日方舟")

    def test_mines_compound_name(self):
        from aliceqq.search import mine_topic
        # 未登记的作品名也能靠"以档案/方舟等结尾"的模式识别出来；
        # 前缀要能吃下 2-3 字（"蔚蓝档案" vs "封印者档案"）
        self.assertEqual(mine_topic("封印者档案 好玩吗"), "封印者档案")
        self.assertEqual(mine_topic("最近在玩星穹铁道"), "星穹铁道")

    def test_returns_empty_when_no_topic(self):
        from aliceqq.search import mine_topic
        self.assertEqual(mine_topic("看番看到深夜"), "")
        self.assertEqual(mine_topic("今天好累"), "")

    def test_exclude_avoids_repeating_current_message(self):
        from aliceqq.search import mine_topic
        self.assertEqual(mine_topic("蔚蓝档案有什么活动", exclude="蔚蓝档案"), "")


class SearchIntegrationTest(unittest.TestCase):
    def _engine_with_search(self, provider, **overrides):
        engine = make_engine(**overrides)
        engine.config.search.enabled = True
        engine.config.features.web_search = True
        engine.search_service = SearchService(provider, fetch_pages=0)
        return engine

    def test_auto_search_on_time_sensitive_question(self):
        provider = FakeProvider()
        engine = self._engine_with_search(provider)
        _, outcome = private("今天蔚蓝档案有什么新活动吗", engine=engine)
        self.assertTrue(provider.calls, "时效性问题应当触发检索")
        self.assertIsNotNone(outcome.search)
        self.assertIn("baike.baidu.com", outcome.prompt_system)

    def test_plain_chat_does_not_search(self):
        provider = FakeProvider()
        engine = self._engine_with_search(provider)
        private("陪我聊聊天吧", engine=engine)
        self.assertEqual(provider.calls, [], "普通闲聊不该联网")

    def test_search_disabled_marks_tool_unavailable(self):
        engine = make_engine()
        from aliceqq import tools
        catalog = tools.catalog_text(engine.config, bool(engine.search_service))
        self.assertIn("未启用联网检索", catalog)

    def test_search_enabled_marks_tool_available(self):
        engine = self._engine_with_search(FakeProvider())
        from aliceqq import tools
        catalog = tools.catalog_text(engine.config, bool(engine.search_service))
        self.assertIn("web_search: 联网检索最新信息。", catalog)
        self.assertNotIn("web_search: 联网检索最新信息。（", catalog)

    def test_tool_call_path_returns_results(self):
        engine = self._engine_with_search(FakeProvider())
        outcome = engine.handle(IncomingMessage(
            user_id="10001", text="帮我搜一下爱丽丝的设定", is_private=True, user_name="小星",
            timestamp=time.time()))
        # 无论模型是否主动调用工具，结果都应可被检索到
        self.assertIsNotNone(engine._pending_search)
        self.assertEqual(len(engine._pending_search.results), 2)

    def test_vague_query_asks_instead_of_searching(self):
        # 没有话题词的模糊问题：不该硬搜（会拿到别的游戏的信息），而应反问
        provider = FakeProvider()
        engine = self._engine_with_search(provider)
        _, outcome = private("今天有什么新活动", engine=engine)
        self.assertEqual(provider.calls, [], "缺少话题时不该发起检索")
        self.assertIn("不太确定问的是哪件事", outcome.reply)

    def test_failed_search_triggers_honest_fallback(self):
        provider = FakeProvider(ok=False, error="连接超时")
        engine = self._engine_with_search(provider)

        class ClaimingLLM:
            name = "claiming"

            def chat(self, messages, *, config, tools=None):
                from aliceqq.llm import Reply
                return Reply(text="根据最新消息，活动从明天开始，来源：https://example.com",
                             provider="claiming", model="fake")

        engine.llm = ClaimingLLM()
        # 带上话题词，才会真正发起检索
        _, outcome = private("蔚蓝档案今天有什么新活动", engine=engine)
        self.assertIn("检索没有成功", outcome.reply)
        self.assertNotIn("https://example.com", outcome.reply)
        self.assertTrue(outcome.debug.get("search_failure_replaced") is not None
                        or "search_failure_replaced" in outcome.debug)

    def test_successful_search_reply_not_replaced(self):
        engine = self._engine_with_search(FakeProvider())

        class GoodLLM:
            name = "good"

            def chat(self, messages, *, config, tools=None):
                from aliceqq.llm import Reply
                return Reply(text="查到了：活动明天开始。来源：baike.baidu.com",
                             provider="good", model="fake")

        engine.llm = GoodLLM()
        # 查询需要带话题词，才会真正发起检索（否则走"反问澄清"分支）
        _, outcome = private("蔚蓝档案今天有什么新活动", engine=engine)
        self.assertIn("baike.baidu.com", outcome.reply)
        self.assertNotIn("search_failure_replaced", outcome.debug)

    def test_search_result_not_leaked_across_users(self):
        provider = FakeProvider()
        engine = self._engine_with_search(provider)
        private("今天有什么新活动", user="10001", engine=engine)
        _, _, variables = engine.build_messages(
            IncomingMessage(user_id="10002", text="你好", is_private=True, user_name="别人"), MODE_ALICE)
        self.assertIn("没有进行检索", variables["search_results"])

    def test_assistant_mode_gets_search_results(self):
        provider = FakeProvider()
        engine = self._engine_with_search(provider)
        engine.storage.set_mode("10001", MODE_ASSISTANT)
        _, outcome = private("今天发生了什么新闻", engine=engine)
        self.assertIn("baike.baidu.com", outcome.prompt_system)


@unittest.skipUnless(os.environ.get("ALICE_LIVE_SEARCH") == "1", "需要联网，设置 ALICE_LIVE_SEARCH=1 才跑")
class LiveSearchTest(unittest.TestCase):
    def test_real_duckduckgo_search(self):
        service = SearchService(DuckDuckGoProvider(timeout=20), fetch_pages=0)
        outcome = service.search("蔚蓝档案 爱丽丝")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(outcome.results)
        for result in outcome.results:
            self.assertTrue(is_safe_url(result.url))


if __name__ == "__main__":
    unittest.main(verbosity=2)
