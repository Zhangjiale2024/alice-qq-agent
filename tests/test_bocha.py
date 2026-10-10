"""博查 API 与风控防抖的测试（离线，用打桩的 HTTP）。

响应结构取自官方文档，测试保证字段映射与请求体形状正确。
"""
from __future__ import annotations

import io
import json
import time
import unittest
from unittest import mock

from aliceqq.search import (BochaProvider, SearchOutcome, _DOMAIN_GUARD, _DomainGuard,
                            looks_blocked)

# 官方文档给出的响应样例（截取必要字段）
DOC_RESPONSE = {
    "code": 200,
    "log_id": "d71841ad20095f61",
    "msg": None,
    "data": {
        "_type": "SearchResponse",
        "queryContext": {"originalQuery": "蔚蓝档案最新活动"},
        "webPages": {
            "totalEstimatedMatches": 8912791,
            "value": [
                {
                    "name": "蔚蓝档案国服活动公告",
                    "url": "https://www.taptap.cn/moment/123",
                    "snippet": "简短描述",
                    "summary": "本期活动从 10 月 9 日开始，持续到 10 月 29 日。",
                    "siteName": "TapTap",
                    "datePublished": "2026-10-07T00:00:00+08:00",
                    "dateLastCrawled": "2026-10-07T00:00:00Z",
                },
                {
                    "name": "内网地址不该出现",
                    "url": "http://127.0.0.1/admin",
                    "summary": "danger",
                },
            ],
        },
    },
}


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200):
        super().__init__(payload)
        self.status = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class BochaProviderTest(unittest.TestCase):
    def test_parses_documented_response(self):
        provider = BochaProvider("test-key")
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(json.dumps(DOC_RESPONSE).encode("utf-8"))

        with mock.patch("aliceqq.search.net.open", fake_urlopen):
            outcome = provider.search("蔚蓝档案最新活动", limit=5)

        self.assertTrue(outcome.ok, outcome.error)
        # 内网地址必须被过滤
        self.assertEqual(len(outcome.results), 1)
        result = outcome.results[0]
        self.assertEqual(result.title, "蔚蓝档案国服活动公告")
        self.assertEqual(result.url, "https://www.taptap.cn/moment/123")
        # summary 优先于 snippet，且带上发布时间与站点名
        self.assertIn("10 月 9 日", result.snippet)
        self.assertIn("2026-10-07", result.snippet)
        self.assertIn("TapTap", result.snippet)
        # 请求体形状
        self.assertTrue(captured["body"]["summary"])
        self.assertEqual(captured["body"]["query"], "蔚蓝档案最新活动")
        self.assertIn("freshness", captured["body"])
        self.assertTrue(captured["url"].startswith("https://api.bocha.cn/v1/web-search"))
        # 鉴权头
        self.assertTrue(any("Bearer" in str(v) for v in captured["headers"].values())
                        or any("bearer" in str(k).lower() for k in captured["headers"]))

    def test_freshness_mapping(self):
        cases = {
            "今天有什么活动": "oneDay",
            "这周更新了吗": "oneWeek",
            "最近有什么活动": "oneMonth",
            "蔚蓝档案是什么": "noLimit",
        }
        for query, expected in cases.items():
            self.assertEqual(BochaProvider._freshness_for(query), expected, query)

    def test_without_key_reports_clearly(self):
        outcome = BochaProvider("").search("x")
        self.assertFalse(outcome.ok)
        self.assertIn("BOCHA_API_KEY", outcome.error)

    def test_quota_error_is_distinguished_from_invalid_key(self):
        # 实测：Key 有效但账户没额度时也返回 403，文案必须区分开，
        # 否则用户会以为 Key 错了、去反复换 Key。
        import urllib.error
        quota_body = json.dumps({"code": "403",
                                 "message": "You do not have enough money or package quota"}).encode()

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(quota_body))

        with mock.patch("aliceqq.search.net.open", fake_urlopen):
            outcome = BochaProvider("valid-but-broke").search("x")
        self.assertFalse(outcome.ok)
        self.assertIn("没有余额", outcome.error)
        self.assertIn("免费额度", outcome.error)

    def test_auth_error_mentions_key(self):
        import urllib.error

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

        with mock.patch("aliceqq.search.net.open", fake_urlopen):
            outcome = BochaProvider("wrong").search("x")
        self.assertFalse(outcome.ok)
        self.assertIn("401", outcome.error)
        self.assertIn("API Key", outcome.error)

    def test_api_level_error_is_reported(self):
        payload = {"code": 403, "msg": "quota exceeded"}

        def fake_urlopen(request, timeout=None):
            return _FakeResponse(json.dumps(payload).encode("utf-8"))

        with mock.patch("aliceqq.search.net.open", fake_urlopen):
            outcome = BochaProvider("k").search("x")
        self.assertFalse(outcome.ok)
        self.assertIn("quota exceeded", outcome.error)


class BlockDetectionTest(unittest.TestCase):
    def test_detects_360_access_page(self):
        page = "<html><head><title>访问异常页面</title></head><body>请输入验证码 captcha</body></html>"
        self.assertTrue(looks_blocked(page))

    def test_detects_baidu_verify(self):
        page = "<html><head><title>百度安全验证</title></head><body>wappass.baidu.com 请完成验证</body></html>"
        self.assertTrue(looks_blocked(page))

    def test_normal_page_is_not_blocked(self):
        page = ("<html><head><title>蔚蓝档案 - 萌娘百科</title></head><body>" + "内容" * 6000
                + "</body></html>")
        self.assertFalse(looks_blocked(page))

    def test_short_generic_page_not_falsely_blocked(self):
        # 只有单个关键词不足以判定拦截，避免误伤正常短页面
        self.assertFalse(looks_blocked("<html><title>Article</title><body>captcha theory</body></html>"))


class DomainGuardTest(unittest.TestCase):
    def test_escalating_cooldown(self):
        guard = _DomainGuard()
        first = guard.note_block("x.com")
        second = guard.note_block("x.com")
        third = guard.note_block("x.com")
        self.assertLess(first, second)
        self.assertLess(second, third)
        self.assertGreater(guard.blocked_for("x.com"), 0)

    def test_success_resets_escalation(self):
        guard = _DomainGuard()
        guard.note_block("y.com")
        guard.note_block("y.com")
        guard.note_success("y.com")
        # 重置后再被拦，冷却应回到第一档
        self.assertLessEqual(guard.note_block("y.com"), _DomainGuard.COOLDOWN_LADDER[1])

    def test_min_interval_enforced(self):
        guard = _DomainGuard(min_interval=0.2)
        guard.acquire("z.com")
        started = time.time()
        guard.acquire("z.com")
        self.assertGreaterEqual(time.time() - started, 0.15)

    def test_cooldown_expires(self):
        guard = _DomainGuard()
        guard._blocked_until["w.com"] = time.time() - 1  # 已过期
        self.assertEqual(guard.blocked_for("w.com"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
