"""B 站搜索测试（离线，用打桩的 HTTP）。

响应结构取自实测（search/type 与 search/all/v2 的真实字段）。
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from aliceqq.search import BilibiliProvider, _DOMAIN_GUARD

# 实测到的字段结构
TYPE_RESPONSE = {
    "code": 0,
    "message": "OK",
    "data": {
        "page": 1,
        "numResults": 1000,
        "result": [
            {
                "type": "video",
                "author": "-冬阳同学-",
                "bvid": "BV1ofan6NEPY",
                "title": '这次乐队终于不再三缺一了！【<em class="keyword">蔚蓝档案</em>】乐队活动复刻攻略',
                "description": "本期活动从 10 月 9 日开始，持续到 10 月 29 日。",
                "pubdate": 1791345600,
                "arcurl": "http://www.bilibili.com/video/BV1ofan6NEPY",
            },
            {
                "type": "video",
                "author": "某人",
                "bvid": "BV1xxxxxxxxx",
                "title": "无关视频",
                "description": "",
                "pubdate": 0,
            },
        ],
    },
}

ARTICLE_RESPONSE = {
    "code": 0,
    "data": {
        "result": [
            {
                "id": 23337585,
                "author": "攻略组",
                "title": "【<em class=\"keyword\">蔚蓝档案</em>】活动公告汇总",
                "desc": "本期活动公告与卡池时间表。",
                "pubdate": 1791345600,
            }
        ]
    },
}


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.status = 200
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeOpener:
    def __init__(self, responses):
        # responses: {关键字: payload}
        self.responses = responses
        self.calls = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.calls.append(url)
        for key, payload in self.responses.items():
            if key in url:
                return _FakeResponse(json.dumps(payload).encode("utf-8"))
        return _FakeResponse(json.dumps({"code": -412, "message": "request was banned"}).encode())


class BilibiliProviderTest(unittest.TestCase):
    def _provider(self, responses):
        provider = BilibiliProvider(timeout=5)
        opener = _FakeOpener(responses)
        provider._opener = opener
        provider._warmed = True   # 跳过首页预热，避免测试里发真实请求
        return provider, opener

    def test_parses_video_results(self):
        provider, opener = self._provider({"search_type=video": TYPE_RESPONSE})
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            outcome = provider.search("蔚蓝档案 最新活动", limit=5)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(len(outcome.results), 2)
        first = outcome.results[0]
        # 标题里的 <em> 高亮标签必须清掉
        self.assertNotIn("<em", first.title)
        self.assertIn("蔚蓝档案", first.title)
        # bvid -> 可访问链接
        self.assertEqual(first.url, "https://www.bilibili.com/video/BV1ofan6NEPY")
        self.assertIn("2026-", first.snippet)      # pubdate 格式化后出现在摘要里
        self.assertIn("UP：-冬阳同学-", first.snippet)

    def test_article_type_is_queried_and_parsed(self):
        provider, opener = self._provider({
            "search_type=video": {"code": 0, "data": {"result": []}},
            "search_type=article": ARTICLE_RESPONSE,
        })
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            outcome = provider.search("蔚蓝档案 活动公告", limit=5)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.results[0].url, "https://www.bilibili.com/read/cv23337585")
        self.assertIn("专栏", outcome.results[0].source)

    def test_falls_back_to_all_v2(self):
        all_response = {
            "code": 0,
            "data": {"result": [{"result_type": "video", "data": TYPE_RESPONSE["data"]["result"]}]},
        }
        provider, opener = self._provider({
            "search_type=video": {"code": -412, "message": "request was banned"},
            "search_type=article": {"code": -412, "message": "request was banned"},
            "search/all/v2": all_response,
        })
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            outcome = provider.search("蔚蓝档案", limit=5)
        self.assertTrue(outcome.ok, outcome.error)
        self.assertTrue(any("all/v2" in url for url in opener.calls))

    def test_risk_control_is_reported_and_enters_cooldown(self):
        provider, _ = self._provider({})   # 全部返回 -412
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            outcome = provider.search("蔚蓝档案", limit=5)
        self.assertFalse(outcome.ok)
        self.assertIn("412", outcome.error)
        self.assertGreater(_DOMAIN_GUARD.blocked_for("bilibili.com"), 0)
        _DOMAIN_GUARD._blocked_until.pop("bilibili.com", None)  # 清理，避免影响其它测试

    def test_pubdate_formatting(self):
        self.assertEqual(BilibiliProvider._format_pubdate(1791345600), "2026-10-07")
        # 0 与非法值都返回空串，不能报成 1970
        self.assertEqual(BilibiliProvider._format_pubdate(0), "")
        self.assertEqual(BilibiliProvider._format_pubdate(None), "")

    def test_title_cleanup(self):
        raw = '【<em class="keyword">蔚蓝档案</em>】攻略 &amp; 心得'
        cleaned = BilibiliProvider._clean_title(raw)
        self.assertEqual(cleaned, "【蔚蓝档案】攻略 & 心得")


if __name__ == "__main__":
    unittest.main(verbosity=2)
