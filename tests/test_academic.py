"""学术检索与多源回退链的测试（全部离线，用假 provider）。"""
from __future__ import annotations

import unittest

from aliceqq.academic import (AcademicOutcome, AcademicService, ArxivProvider, CrossRefProvider,
                              SemanticScholarProvider, format_academic, is_academic_query)
from aliceqq.search import ChainedProvider, SearchOutcome, SearchResult


def make_result(title: str, url: str = "https://example.com/a") -> SearchResult:
    return SearchResult(title=title, url=url, snippet="摘要", source="example.com", trusted=False)


class FakeProvider:
    def __init__(self, name, results=None, ok=True, error="", rate_limited=False):
        self.name = name
        self._results = results if results is not None else [make_result(f"{name} 的结果")]
        self._ok = ok
        self._error = error
        self._rate_limited = rate_limited
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        return AcademicOutcome(self._ok, query, self.name, results=self._results,
                               error=self._error, rate_limited=self._rate_limited)


class FakeWebProvider:
    """假网页检索源。

    注意：默认结果的标题里**必须包含查询词**，否则会被 ChainedProvider 的
    相关度门禁判为低相关——这也是实测中 Bing 返回「基（汉字）」时的真实情形。
    """

    def __init__(self, name, results=None, ok=True, error=""):
        self.name = name
        # 默认造一条"标题包含查询词"的结果，贴近真实可用源的行为
        self._results = results if results is not None else None
        self._ok = ok
        self._error = error
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        results = self._results
        if results is None:
            results = [make_result(f"{query} - {self.name} 的结果")]
        return SearchOutcome(self._ok, query, self.name, results=results, error=self._error)


class AcademicIntentTest(unittest.TestCase):
    def test_detects_academic_queries(self):
        for text in ("帮我找几篇关于深度学习的论文", "10.1038/nature12373 是什么",
                     "arXiv 上有什么相关研究", "有这个文献的 DOI 吗"):
            self.assertTrue(is_academic_query(text), text)

    def test_ignores_non_academic(self):
        for text in ("蔚蓝档案最近有什么活动", "推荐个游戏", "今天好累", "爱丽丝是谁"):
            self.assertFalse(is_academic_query(text), text)


class AcademicChainTest(unittest.TestCase):
    def test_uses_first_successful_provider(self):
        first = FakeProvider("arxiv")
        second = FakeProvider("crossref")
        service = AcademicService([first, second])
        outcome = service.search("deep learning")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "arxiv")
        self.assertEqual(second.calls, [], "第一个源成功时不该继续调用后面的")

    def test_falls_back_when_rate_limited(self):
        # 这正是 Semantic Scholar 的实测行为：无 key 时 429
        s2 = FakeProvider("semantic_scholar", ok=False, error="HTTP 429 被限流",
                          rate_limited=True)
        arxiv = FakeProvider("arxiv")
        service = AcademicService([s2, arxiv])
        outcome = service.search("deep learning")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "arxiv")
        self.assertTrue(s2.calls)

    def test_reports_rate_limit_when_all_fail(self):
        s2 = FakeProvider("semantic_scholar", ok=False, error="HTTP 429 被限流", rate_limited=True)
        crossref = FakeProvider("crossref", ok=False, error="没有匹配的文献")
        service = AcademicService([s2, crossref])
        outcome = service.search("x")
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.rate_limited)
        text = format_academic(outcome)
        self.assertIn("限流", text)

    def test_empty_service_is_unavailable(self):
        self.assertFalse(AcademicService([]).available)

    def test_doi_query_prefers_crossref(self):
        # 带 DOI 的查询应优先走 CrossRef 的精确接口，而不是让 arXiv 把它当文本搜
        arxiv = FakeProvider("arxiv")
        crossref = FakeProvider("crossref")
        service = AcademicService([arxiv, crossref])
        outcome = service.search("这篇 10.1103/PhysRevLett.116.061102 讲什么")
        self.assertEqual(outcome.provider, "crossref")
        self.assertEqual(arxiv.calls, [], "带 DOI 时不应先调用 arXiv")

    def test_non_doi_query_keeps_original_order(self):
        arxiv = FakeProvider("arxiv")
        crossref = FakeProvider("crossref")
        service = AcademicService([arxiv, crossref])
        outcome = service.search("深度学习可解释性")
        self.assertEqual(outcome.provider, "arxiv")

    def test_caches_repeated_queries(self):
        provider = FakeProvider("arxiv")
        service = AcademicService([provider])
        service.search("q")
        service.search("q")
        self.assertEqual(len(provider.calls), 1, "同一查询应命中缓存")


class FormatAcademicTest(unittest.TestCase):
    def test_format_includes_title_and_link(self):
        outcome = AcademicOutcome(True, "q", "arxiv", results=[
            SearchResult("A Paper", "https://arxiv.org/abs/1234.5678", "abstract here",
                         "arxiv.org", True)])
        text = format_academic(outcome)
        self.assertIn("A Paper", text)
        self.assertIn("https://arxiv.org/abs/1234.5678", text)
        self.assertIn("abstract here", text)
        # 明确禁止编造文献细节
        self.assertIn("不要编造", text)

    def test_format_failure_is_honest(self):
        outcome = AcademicOutcome(False, "q", "arxiv", error="连接超时")
        text = format_academic(outcome)
        self.assertIn("学术检索失败", text)
        self.assertIn("连接超时", text)


class QuestionCoreTest(unittest.TestCase):
    """口语提问要把疑问套话剥掉，否则逐条过滤会误删好结果。"""

    def test_strips_question_prefix(self):
        from aliceqq.search import question_core
        self.assertEqual(question_core("什么是量子纠缠"), "量子纠缠")
        self.assertEqual(question_core("请问什么是深度学习"), "深度学习")

    def test_terms_include_core_phrase(self):
        from aliceqq.search import _query_terms
        terms = _query_terms("什么是量子纠缠")
        self.assertIn("量子纠缠", terms, f"未取到核心短语：{terms}")

    def test_core_not_over_stripped(self):
        from aliceqq.search import question_core
        # 不该把词干也吃掉
        self.assertEqual(question_core("量子纠缠"), "量子纠缠")


class RelevanceForSpacelessChineseTest(unittest.TestCase):
    """中文提问常不带空格——这是相关度判定最容易失效的地方。"""

    def test_query_terms_extracts_topic_from_spaceless_sentence(self):
        from aliceqq.search import _query_terms
        terms = _query_terms("蔚蓝档案最近有什么活动？")
        self.assertIn("蔚蓝档案", terms, f"未能从无空格句中抽出专名：{terms}")

    def test_redundant_superstring_terms_are_dropped(self):
        # 整句词永远匹配不到标题，留着只会稀释相关度
        from aliceqq.search import _query_terms
        self.assertEqual(_query_terms("蔚蓝档案最近有什么活动？"), ["蔚蓝档案"])
        self.assertEqual(_query_terms("什么是量子纠缠"), ["量子纠缠"])

    def test_relevance_not_zero_for_good_results(self):
        from aliceqq.search import ChainedProvider
        outcome = SearchOutcome(True, "蔚蓝档案最近有什么活动？", "bocha", results=[
            make_result("蔚蓝档案二周年活动内容有哪些"),
            make_result("《蔚蓝档案》三周年演出回顾"),
        ])
        score = ChainedProvider.relevance(outcome, "蔚蓝档案最近有什么活动？")
        self.assertGreaterEqual(score, 0.34,
                                "标题确实相关的无空格查询不该被判定为低相关")

    def test_ungradable_query_is_not_falsely_flagged(self):
        # "今天的科技新闻"既无空格也无专名，只剩整句 -> 不判，
        # 否则会把当天新闻误标成"相关度较低"
        from aliceqq.search import ChainedProvider, relevance_terms
        self.assertEqual(relevance_terms("今天的科技新闻"), [])
        outcome = SearchOutcome(True, "今天的科技新闻", "bilibili", results=[
            make_result("2026.10.07科技前沿新闻，未来已来"),
        ])
        self.assertEqual(ChainedProvider.relevance(outcome, "今天的科技新闻"), 1.0)

    def test_gradable_queries_still_use_terms(self):
        from aliceqq.search import relevance_terms
        # 顺序按 (长度, 码点) 排定，保证可复现
        self.assertEqual(sorted(relevance_terms("蔚蓝档案 最新活动")),
                         sorted(["蔚蓝档案", "最新活动"]))
        self.assertEqual(relevance_terms("什么是量子纠缠"), ["量子纠缠"])
        # 同样输入必须给出同样顺序
        self.assertEqual(relevance_terms("蔚蓝档案 最新活动"),
                         relevance_terms("蔚蓝档案 最新活动"))

    def test_relevance_still_rejects_junk(self):
        from aliceqq.search import ChainedProvider
        outcome = SearchOutcome(True, "基沃托斯 游戏开发部", "bing", results=[
            make_result("基（汉字）_百度百科"),
        ])
        self.assertLess(ChainedProvider.relevance(outcome, "基沃托斯 游戏开发部"), 0.34)


class SearchChainTest(unittest.TestCase):
    def test_first_provider_wins(self):
        first = FakeWebProvider("so360")
        second = FakeWebProvider("bing")
        chain = ChainedProvider([first, second])
        outcome = chain.search("蔚蓝档案")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "so360")
        self.assertEqual(second.calls, [])

    def test_falls_back_to_second_when_first_fails(self):
        first = FakeWebProvider("so360", ok=False, error="页面结构已变化")
        second = FakeWebProvider("bing")
        chain = ChainedProvider([first, second])
        outcome = chain.search("蔚蓝档案")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "bing")
        self.assertTrue(first.calls and second.calls)

    def test_reports_all_failures(self):
        first = FakeWebProvider("so360", ok=False, error="HTTP 202 反爬")
        second = FakeWebProvider("bing", ok=False, error="HTTP 200 但解析失败")
        chain = ChainedProvider([first, second])
        outcome = chain.search("x")
        self.assertFalse(outcome.ok)
        self.assertIn("so360", outcome.error)
        self.assertIn("bing", outcome.error)

    def test_empty_results_count_as_failure(self):
        # 有 200 但零结果，也要继续尝试下一个源
        first = FakeWebProvider("so360", results=[])
        second = FakeWebProvider("bing")
        chain = ChainedProvider([first, second])
        outcome = chain.search("x")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "bing")

    def test_individual_junk_results_are_dropped(self):
        # 实测：查"什么是量子纠缠"时返回里混进了完全不相关的内容。
        # 批次级相关度挡不住它，必须逐条过滤。
        provider = FakeWebProvider("bilibili", results=[
            make_result("一口气看懂量子力学，量子叠加，量子纠缠"),
            make_result("女人的第一次，到底是什麼感覺呢？真實經歷！"),
            make_result("简单讲一下什么是量子纠缠"),
        ])
        chain = ChainedProvider([provider])
        outcome = chain.search("什么是量子纠缠")
        titles = [r.title for r in outcome.results]
        self.assertIn("一口气看懂量子力学，量子叠加，量子纠缠", titles)
        self.assertNotIn("女人的第一次，到底是什麼感覺呢？真實經歷！", titles)

    def test_filter_keeps_everything_when_no_terms(self):
        from aliceqq.search import ChainedProvider as CP
        outcome = SearchOutcome(True, "嗯", "x", results=[make_result("任意标题")])
        filtered = CP.filter_relevant(outcome, "嗯")
        self.assertEqual(len(filtered.results), 1, "没有可判定词时不应误删")

    def test_filter_keeps_original_if_all_would_be_dropped(self):
        # 全部不命中说明判定过严，此时保留原样，交给批次级相关度处理
        from aliceqq.search import ChainedProvider as CP
        outcome = SearchOutcome(True, "q", "x", results=[make_result("完全不搭边的标题")])
        filtered = CP.filter_relevant(outcome, "蔚蓝档案 最新活动")
        self.assertEqual(len(filtered.results), 1)

    def test_low_relevance_results_are_not_accepted_silently(self):
        # 实测情形：查"基沃托斯 游戏开发部"，某源返回「基（汉字）_百度百科」。
        # 这类结果不能让链直接采用。
        junk = FakeWebProvider("bing", results=[make_result("基（汉字）_百度百科"),
                                                make_result("基 jī - 汉典")])
        good = FakeWebProvider("so360")
        chain = ChainedProvider([junk, good])
        outcome = chain.search("基沃托斯 游戏开发部")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.provider, "so360",
                         "低相关的结果应被跳过，继续尝试下一个源")

    def test_all_low_relevance_is_flagged(self):
        # 全部源都低相关时，仍然给结果，但要标注出来让模型知道可能没对准
        junk_a = FakeWebProvider("bing", results=[make_result("基（汉字）_百度百科")])
        junk_b = FakeWebProvider("so360", results=[make_result("与查询无关的内容")])
        chain = ChainedProvider([junk_a, junk_b])
        outcome = chain.search("基沃托斯 游戏开发部")
        self.assertTrue(outcome.ok)
        self.assertIn("相关度较低", outcome.provider)

    def test_blocked_sources_are_named_in_annotation(self):
        # 实测情形：360 被风控后链退到 Bing，注释里应写明"so360 被风控拦截"，
        # 这样模型能如实告诉用户"检索源暂时不可用"，而不是让人以为"网上没这条信息"
        blocked = FakeWebProvider("so360", ok=False,
                                  error="被 360 判定为异常访问（返回「访问异常页面」），进入 30 秒冷却")
        junk = FakeWebProvider("bing", results=[make_result("基（汉字）_百度百科")])
        chain = ChainedProvider([blocked, junk])
        outcome = chain.search("基沃托斯 游戏开发部")
        self.assertTrue(outcome.ok)
        self.assertIn("so360", outcome.provider)
        self.assertIn("风控", outcome.provider)

    def test_quota_failure_is_named_in_annotation(self):
        # 博查无额度（403）也要写进注释，否则用户只看到结果差、看不到原因
        no_quota = FakeWebProvider(
            "bocha", ok=False,
            error="HTTP 403（Key 有效，但账户没有余额/套餐额度——需要到控制台充值或领取免费额度）")
        junk = FakeWebProvider("bing", results=[make_result("基（汉字）_百度百科")])
        chain = ChainedProvider([no_quota, junk])
        outcome = chain.search("基沃托斯 游戏开发部")
        self.assertIn("bocha", outcome.provider)
        self.assertIn("无额度", outcome.provider)

    def test_caches_fallthrough_result(self):
        junk = FakeWebProvider("bing", results=[make_result("基（汉字）_百度百科")])
        good = FakeWebProvider("so360")
        chain = ChainedProvider([junk, good])
        chain.search("基沃托斯 游戏开发部")
        calls_after_first = len(good.calls)
        chain.search("基沃托斯 游戏开发部")
        self.assertEqual(len(good.calls), calls_after_first, "同查询应命中缓存")

    def test_provider_names_readable(self):
        chain = ChainedProvider([FakeWebProvider("so360"), FakeWebProvider("bing")])
        self.assertEqual(chain.provider_names, "so360 -> bing")


class ProviderUnitTest(unittest.TestCase):
    def test_arxiv_parses_atom(self):
        # 用真实抓取到的 Atom 结构片段做解析验证（不发网络请求）
        import re
        sample = ("<entry><title>Modern Mathematics of Deep Learning</title>"
                  "<id>http://arxiv.org/abs/2105.04026v2</id>"
                  "<published>2021-05-09T00:00:00Z</published>"
                  "<summary>We describe the new field.</summary>"
                  "<author><name>Julius Berner</name></author></entry>")
        title = re.search(r"<title>(.*?)</title>", sample, re.S).group(1)
        self.assertIn("Deep Learning", title)

    def test_crossref_handles_missing_abstract(self):
        provider = CrossRefProvider()
        result = provider._item_to_result({
            "title": ["A Study"], "DOI": "10.1/x", "URL": "https://doi.org/10.1/x",
            "issued": {"date-parts": [[2024]]}, "container-title": ["Some Journal"],
            "author": [{"given": "Jane", "family": "Doe"}],
        })
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "crossref.org · DOI:10.1/x")
        self.assertIn("未提供摘要", result.snippet)
        self.assertIn("2024", result.snippet)

    def test_crossref_rejects_unsafe_url(self):
        provider = CrossRefProvider()
        self.assertIsNone(provider._item_to_result(
            {"title": ["X"], "URL": "http://127.0.0.1/x"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
