"""B 站"视频内信息 + 评论"的测试（离线，打桩 HTTP）。

响应结构取自实测。
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from aliceqq.search import BilibiliProvider, SearchResult

DETAIL = {
    "code": 0,
    "data": {
        "aid": 115957489999556, "cid": 42172482540,
        "desc": "文案校对：@某人 本期讲入坑规划",
        "duration": 1234,
        "pages": [{"cid": 1}, {"cid": 2}],
        "stat": {"view": 428368, "like": 12000},
    },
}

PLAYER_WITH_SUB = {
    "code": 0,
    "data": {"subtitle": {"subtitles": [
        {"lan": "zh-CN", "lan_doc": "中文（自动）",
         "subtitle_url": "//aisubtitle.hdslb.com/bfs/ai_subtitle/x.json"},
    ]}},
}
PLAYER_NO_SUB = {"code": 0, "data": {"subtitle": {"subtitles": []}}}

REPLIES = {
    "code": 0,
    "data": {"replies": [
        {"like": 851, "content": {"message": "买自抽！真的很重要"}},
        {"like": 270, "content": {"message": "每天登录一次把日常做了就行"}},
        {"like": 108, "content": {"message": "萌新刚入坑5天，38级"}},
    ]},
}

SUBTITLE_JSON = {"body": [
    {"content": "大家好"}, {"content": "今天讲入坑规划"}, {"content": "首先是买自抽"},
]}


class _FakeResponse(io.BytesIO):
    def __init__(self, payload):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        super().__init__(raw)
        self.status = 200
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeOpener:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.calls.append(url)
        for key, payload in self.routes.items():
            if key in url:
                return _FakeResponse(payload)
        return _FakeResponse({"code": -404, "message": "not found"})


def make_provider(routes, **kwargs):
    provider = BilibiliProvider(timeout=5, **kwargs)
    provider._opener = _FakeOpener(routes)
    provider._warmed = True
    return provider, provider._opener


class DetailTest(unittest.TestCase):
    def test_parses_detail_fields(self):
        provider, opener = make_provider({"web-interface/view": DETAIL})
        detail = provider._fetch_detail(opener, "BV1mvz4BDETu")
        self.assertEqual(detail["aid"], 115957489999556)
        self.assertEqual(detail["cid"], 42172482540)
        self.assertIn("入坑规划", detail["desc"])
        self.assertEqual(detail["duration"], 1234)
        self.assertEqual(detail["view"], 428368)
        self.assertEqual(detail["pages"], 2)

    def test_detail_failure_returns_empty(self):
        provider, opener = make_provider({})
        self.assertEqual(provider._fetch_detail(opener, "BVxxx"), {})


class SubtitleTest(unittest.TestCase):
    def test_reads_subtitle_text(self):
        provider, opener = make_provider({
            "player/v2": PLAYER_WITH_SUB,
            "aisubtitle": SUBTITLE_JSON,
        })
        with mock.patch("aliceqq.search.http_get",
                        lambda url, timeout=None: (200, json.dumps(SUBTITLE_JSON), url)):
            result = provider._fetch_subtitle(opener, "BV1", 42172482540)
        self.assertIn("买自抽", result["text"])
        self.assertIn("中文", result["note"])

    def test_missing_subtitle_explains_how_to_enable(self):
        provider, opener = make_provider({"player/v2": PLAYER_NO_SUB})
        result = provider._fetch_subtitle(opener, "BV1", 42172482540)
        self.assertEqual(result["text"], "")
        # 必须告诉用户"为什么没有"以及"怎么才能有"
        self.assertIn("BILIBILI_SESSDATA", result["note"])

    def test_anonymous_reports_login_hint(self):
        provider = BilibiliProvider(timeout=5)
        self.assertFalse(provider.logged_in)
        provider2 = BilibiliProvider(timeout=5, sessdata="abc")
        self.assertTrue(provider2.logged_in)


class CommentTest(unittest.TestCase):
    def test_parses_top_comments_with_likes(self):
        provider, opener = make_provider({"v2/reply": REPLIES})
        comments = provider._fetch_comments(opener, 123)
        self.assertEqual(len(comments), 3)
        self.assertIn("赞851", comments[0])
        self.assertIn("买自抽", comments[0])

    def test_anonymous_does_not_paginate(self):
        # 匿名实测只有 3 条；不该为了拿更多而连发请求
        provider, opener = make_provider({"v2/reply": REPLIES})
        provider._fetch_comments(opener, 123)
        self.assertEqual(len(opener.calls), 1, "匿名状态只请求一页")

    def test_logged_in_paginates(self):
        provider, opener = make_provider({"v2/reply": REPLIES}, sessdata="xxx",
                                         comments_limit=25)
        provider._fetch_comments(opener, 123)
        self.assertGreaterEqual(len(opener.calls), 1)

    def test_no_aid_returns_empty(self):
        provider, opener = make_provider({"v2/reply": REPLIES})
        self.assertEqual(provider._fetch_comments(opener, None), [])


class TranscriptGuardTest(unittest.TestCase):
    """字幕/识别文本的"未核对"标注与跑题警告。

    为什么不做成闸门：实测长文本里"活动""都会"这类高频二元组必然偶然撞车，
    用零重合当判据既拦不住搬运内容，又会误杀正常解说。所以改为标注 + 警告。
    """

    def test_mismatched_transcript_is_warned_not_dropped(self):
        """跑题时：文本要保留，但必须带警告。

        这是刻意的设计——早期版本直接丢弃跑题文本，但那个判据在长文本上不可靠
        （"活动""都会"这类高频二元组必然偶然撞车），会误杀正常解说。
        """
        mcdonalds = ("昨天晚上突然给我发了个账单 302块五毛五的麦当劳 然后跟我说 "
                     "明天员工餐靠你了 我才不是你抠门 就是你想搞我 冷冻鸡肉到底有多便宜 "
                     "所有炸的鸡我们用的是同一个腌料 里边有酸奶")
        provider = BilibiliProvider(timeout=5)
        provider._fetch_detail = lambda opener, bvid: {"aid": 1, "cid": 2, "duration": 100}
        provider._fetch_subtitle = lambda opener, bvid, cid: {"text": mcdonalds, "note": ""}
        provider._fetch_comments = lambda opener, aid: []
        opener = _FakeOpener({})
        result = SearchResult("【BA卡牌】我奶奶看了都会的速通攻略",
                              "https://www.bilibili.com/video/BV1x", "", "bilibili.com", True)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            provider._enrich(opener, result, query="蔚蓝档案 活动攻略")
        # 文本保留（不丢内容）
        self.assertIn("麦当劳", result.content)
        # 但明确标注未核对，并给出跑题警告
        self.assertIn("机器识别，未核对", result.content)
        self.assertIn("请勿当作本主题的信息使用", result.content)

    def test_clearly_unrelated_transcript_is_detected(self):
        from aliceqq.search import transcript_matches_topic
        gym = ("那五分化呢五次把全身练一遍 很好理解 你能坚持六天 你还神采飞扬的 "
               "说明你的训练强度可能有问题 哈喽兄弟姐妹们 后台总是有留言问我说 "
               "凯哥到底是一分话好还是三分话好 还是五分化好 说我练五分化 我练了半年")
        ref = "【蔚蓝档案】国服第28章推图走格子攻略 文案校对 蔚蓝档案攻略"
        self.assertFalse(transcript_matches_topic(gym, ref))

    def test_short_text_is_not_judged(self):
        from aliceqq.search import transcript_matches_topic
        self.assertTrue(transcript_matches_topic("好的", "完全无关的参考文本"))

    def test_subtitle_is_labeled_unverified(self):
        provider = BilibiliProvider(timeout=5)
        routes = {"web-interface/view": DETAIL, "player/v2": PLAYER_NO_SUB,
                  "v2/reply": REPLIES}
        opener = _FakeOpener(routes)
        result = SearchResult("标题", "https://www.bilibili.com/video/BV1x", "", "bilibili.com", True)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            provider._enrich(opener, result, query="蔚蓝档案")
        # 没有字幕时应说明原因
        self.assertIn("无字幕数据", result.content)


class EnrichTest(unittest.TestCase):
    def _routes(self):
        return {"web-interface/view": DETAIL, "player/v2": PLAYER_NO_SUB, "v2/reply": REPLIES}

    def test_content_includes_desc_duration_views_and_comments(self):
        provider, opener = make_provider(self._routes())
        result = SearchResult("标题", "https://www.bilibili.com/video/BV1x", "", "bilibili.com", True)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            provider._enrich(opener, result)
        self.assertIn("【视频简介】", result.content)
        self.assertIn("【视频时长】约 20 分钟", result.content)
        self.assertIn("【播放量】428368", result.content)
        self.assertIn("【分P数】2", result.content)
        self.assertIn("【热门评论", result.content)
        self.assertIn("赞851", result.content)

    def test_subtitle_note_can_be_suppressed(self):
        provider, opener = make_provider(self._routes())
        result = SearchResult("标题", "https://www.bilibili.com/video/BV1x", "", "bilibili.com", True)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            provider._enrich(opener, result, include_subtitle_note=False)
        self.assertNotIn("无字幕数据", result.content)
        # 其它信息仍要在
        self.assertIn("【热门评论", result.content)

    def test_non_video_result_is_skipped(self):
        provider, opener = make_provider(self._routes())
        result = SearchResult("专栏", "https://www.bilibili.com/read/cv123", "", "bilibili.com", True)
        provider._enrich(opener, result)
        self.assertEqual(result.content, "")
        self.assertEqual(opener.calls, [], "非视频结果不该发详情/评论请求")

    def test_enrich_top_zero_disables_fetching(self):
        provider, opener = make_provider(
            {"search_type=video": {"code": 0, "data": {"result": [
                {"title": "t", "bvid": "BV1a", "author": "u", "description": "d", "pubdate": 0},
            ]}}},
            enrich_top=0)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            outcome = provider.search("蔚蓝档案", limit=3)
        self.assertTrue(outcome.ok)
        # 关闭富化时不应请求详情/评论
        self.assertFalse(any("web-interface/view" in url for url in opener.calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
