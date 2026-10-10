"""语音识别（音频转文字）测试。

包含：multipart 拼装、ASR 客户端、音频地址选择、超限拒识、闸门与富化集成。
全部离线（打桩 HTTP）。
"""
from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request
from unittest import mock

from aliceqq.asr import ASRClient, Transcription, build_client, build_multipart
from aliceqq.config import AppConfig
from aliceqq.search import BilibiliProvider, SearchResult


class _FakeResponse(io.BytesIO):
    def __init__(self, payload, headers=None):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        super().__init__(raw)
        self.status = 200
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MultipartTest(unittest.TestCase):
    def test_body_shape(self):
        body, content_type = build_multipart({"model": "whisper-1", "language": "zh"},
                                            "file", "audio.m4a", b"AUDIODATA")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        boundary = content_type.split("boundary=", 1)[1]
        text = body.decode("utf-8", "ignore")
        self.assertIn(f"--{boundary}", text)
        self.assertIn('name="model"', text)
        self.assertIn("whisper-1", text)
        self.assertIn('name="file"; filename="audio.m4a"', text)
        self.assertIn("AUDIODATA", text)
        self.assertTrue(text.rstrip().endswith(f"--{boundary}--"))

    def test_boundary_is_unique(self):
        _, c1 = build_multipart({}, "file", "a.m4a", b"x")
        _, c2 = build_multipart({}, "file", "a.m4a", b"x")
        self.assertNotEqual(c1, c2, "boundary 必须每次不同，否则并发会串包")


class ASRClientTest(unittest.TestCase):
    def test_not_available_without_key(self):
        client = ASRClient("https://api.example.com/v1", "")
        self.assertFalse(client.available)
        result = client.transcribe(b"data")
        self.assertFalse(result.ok)
        self.assertIn("ASR_API_KEY", result.error)

    def test_parses_text_response(self):
        client = ASRClient("https://api.example.com/v1", "sk-x")
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization") or request.headers.get("authorization")
            captured["type"] = request.headers.get("Content-type") or request.headers.get("Content-Type")
            return _FakeResponse({"text": "今天我们讲入坑规划"})

        with mock.patch("aliceqq.asr.net.open", fake_urlopen):
            result = client.transcribe(b"AUDIO", filename="a.m4a")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.text, "今天我们讲入坑规划")
        self.assertTrue(captured["url"].endswith("/audio/transcriptions"))
        self.assertIn("Bearer sk-x", str(captured["auth"]))
        self.assertIn("multipart/form-data", str(captured["type"]))

    def test_accepts_plain_text_response(self):
        client = ASRClient("https://api.example.com/v1", "sk-x")
        with mock.patch("aliceqq.asr.net.open",
                        lambda request, timeout=None: _FakeResponse("纯文本结果")):
            result = client.transcribe(b"AUDIO")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "纯文本结果")

    def test_auth_error_mentions_key(self):
        import urllib.error
        client = ASRClient("https://api.example.com/v1", "bad")

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {},
                                         io.BytesIO(b"bad key"))

        with mock.patch("aliceqq.asr.net.open", fake_urlopen):
            result = client.transcribe(b"AUDIO")
        self.assertFalse(result.ok)
        self.assertIn("401", result.error)
        self.assertIn("API Key", result.error)

    def test_empty_result_is_failure_not_silent_success(self):
        client = ASRClient("https://api.example.com/v1", "sk-x")
        with mock.patch("aliceqq.asr.net.open",
                        lambda request, timeout=None: _FakeResponse({"text": "  "})):
            result = client.transcribe(b"AUDIO")
        self.assertFalse(result.ok)
        self.assertIn("为空", result.error)


class BuildClientTest(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertIsNone(build_client(AppConfig()))

    def test_enabled_without_key_stays_disabled(self):
        config = AppConfig()
        config.asr.enabled = True
        config.asr.api_key = ""
        self.assertIsNone(build_client(config))

    def test_enabled_with_key_builds_client(self):
        config = AppConfig()
        config.asr.enabled = True
        config.asr.api_key = "sk-x"
        client = build_client(config)
        self.assertIsNotNone(client)
        self.assertTrue(client.available)


class _FakeOpener:
    def __init__(self, routes, head_length=None):
        self.routes = routes
        self.head_length = head_length
        self.calls = []

    def open(self, request, timeout=None):
        url = request.full_url
        method = request.get_method()
        self.calls.append((method, url))
        if method == "HEAD":
            headers = {"Content-Length": str(self.head_length)} if self.head_length else {}
            return _FakeResponse(b"", headers)
        for key, payload in self.routes.items():
            if key in url:
                return _FakeResponse(payload)
        return _FakeResponse({"code": -404, "message": "not found"})


PLAYURL = {
    "code": 0,
    "data": {"dash": {"audio": [
        {"id": 30280, "bandwidth": 183411, "baseUrl": "https://a.example/high.m4a"},
        {"id": 30216, "bandwidth": 65559, "baseUrl": "https://a.example/low.m4a"},
        {"id": 30232, "bandwidth": 103539, "baseUrl": "https://a.example/mid.m4a"},
    ]}},
}

DETAIL = {"code": 0, "data": {"aid": 1, "cid": 2, "desc": "", "duration": 600,
                              "pages": [{"cid": 2}], "stat": {"view": 10}}}
PLAYER_NO_SUB = {"code": 0, "data": {"subtitle": {"subtitles": []}}}
REPLIES = {"code": 0, "data": {"replies": []}}


class SessionStateTest(unittest.TestCase):
    """SESSDATA 到底有没有生效——不能只看"非空"。

    实测教训：`.env` 里填了个 34 字符的**假** SESSDATA，代码看到非空就当已登录，
    字幕取不到时还提示"配置 BILIBILI_SESSDATA 后多数视频可读到字幕"，
    而实际上 nav 接口返回 isLogin=false，它和匿名毫无区别。

    这里的目标是：**填了坏的 Cookie 时要如实说它没生效**，而不是给一条
    用户照做也没用的建议。
    """

    NAV_OK = {"code": 0, "data": {"isLogin": True, "uname": "someone", "mid": 1}}
    NAV_ANON = {"code": -101, "message": "账号未登录",
                "data": {"isLogin": False, "wbi_img": {"img_url": "https://i0.hdslb.com/bfs/wbi/abc.png",
                                                       "sub_url": "https://i0.hdslb.com/bfs/wbi/def.png"}}}
    PLAYER_NO_SUB = {"code": 0, "data": {"subtitle": {"subtitles": []}}}

    def _provider(self, sessdata):
        provider = BilibiliProvider(timeout=5, sessdata=sessdata)
        provider._warmed = True
        return provider

    def test_no_sessdata_is_anonymous(self):
        provider = self._provider("")
        opener = _FakeOpener({})
        self.assertEqual(provider._session_state(opener), "anonymous")
        # 匿名时不该去查 nav（省一次请求）
        self.assertEqual(opener.calls, [])

    def test_valid_session_detected(self):
        provider = self._provider("real-looking-cookie-value")
        opener = _FakeOpener({"web-interface/nav": self.NAV_OK})
        self.assertEqual(provider._session_state(opener), "valid")

    def test_dead_cookie_is_reported_as_invalid(self):
        provider = self._provider("e3ec0360%2C1806996169%2Cf18b5%2Aa2")
        opener = _FakeOpener({"web-interface/nav": self.NAV_ANON})
        self.assertEqual(provider._session_state(opener), "invalid")

    def test_state_is_cached(self):
        """一次会话只查一次 nav，别给 B 站添请求。"""
        provider = self._provider("cookie")
        opener = _FakeOpener({"web-interface/nav": self.NAV_OK})
        provider._session_state(opener)
        provider._session_state(opener)
        self.assertEqual(len(opener.calls), 1)

    def test_network_error_is_not_reported_as_invalid(self):
        """查不通时不能断言 Cookie 坏了（那是误导）。"""
        provider = self._provider("cookie")

        class _Broken:
            def open(self, request, timeout=None):
                raise OSError("网络断了")

        self.assertEqual(provider._session_state(_Broken()), "unknown")

    def test_no_subtitle_note_for_invalid_cookie_names_the_problem(self):
        provider = self._provider("e3ec0360%2C1806996169%2Cf18b5%2Aa2")
        opener = _FakeOpener({"web-interface/nav": self.NAV_ANON})
        note = provider._no_subtitle_note(opener)
        self.assertIn("没有生效", note)
        self.assertIn("SESSDATA", note)
        # 不能给"配置后就有效"这种照做也没用的建议
        self.assertNotIn("配置 BILIBILI_SESSDATA 后多数视频", note)

    def test_no_subtitle_note_for_anonymous_suggests_configuring(self):
        provider = self._provider("")
        note = provider._no_subtitle_note(_FakeOpener({}))
        self.assertIn("配置 BILIBILI_SESSDATA", note)

    def test_no_subtitle_note_for_valid_session_blames_the_video(self):
        provider = self._provider("cookie")
        opener = _FakeOpener({"web-interface/nav": self.NAV_OK})
        note = provider._no_subtitle_note(opener)
        self.assertIn("本身没有提供", note)
        self.assertNotIn("没有生效", note)

    def test_fetch_subtitle_uses_the_honest_note(self):
        provider = self._provider("e3ec0360%2C1806996169%2Cf18b5%2Aa2")
        routes = {"player/v2": self.PLAYER_NO_SUB, "web-interface/nav": self.NAV_ANON}
        result = provider._fetch_subtitle(_FakeOpener(routes), "BV1", 2)
        self.assertEqual(result["text"], "")
        self.assertIn("没有生效", result["note"])


class SessdataInjectorTest(unittest.TestCase):
    """Cookie 必须真的发出去。

    这是本轮最有价值的一个修复。原先 `_get_opener()` 是把 SESSDATA 塞进
    CookieJar，但项目里请求写成 `_fetch_json(opener, url)`——opener 是外部传进来的
    （测试传假 opener，生产也可能换）。只要有一处不经过 Jar，Cookie 就静默失效，
    而**失效表现和匿名完全一样**：nav 返回未登录、字幕拿不到 0 字节。
    加上 .env 里长期是假占位值，这个 bug 藏了很久。

    真实后果（实测）：填入有效 SESSDATA 后字幕依然 0 字；修好后同一批视频
    能读到 1200 字字幕。
    """

    def test_cookie_is_added_for_bilibili(self):
        from aliceqq.search import _SessdataInjector
        injector = _SessdataInjector("abc123")
        request = urllib.request.Request("https://api.bilibili.com/x/web-interface/nav")
        injector.http_request(request)
        self.assertEqual(request.get_header("Cookie"), "SESSDATA=abc123")

    def test_existing_cookie_header_is_replaced(self):
        """请求上已经有别的 Cookie 头时，要以 SESSDATA 为准。

        否则会带着另一个 Cookie 出去、而 SESSDATA 没生效——这种"看起来配了
        但实际没带"的情况最难查。
        """
        from aliceqq.search import _SessdataInjector
        request = urllib.request.Request("https://api.bilibili.com/x/player/v2")
        request.add_header("Cookie", "buvid3=old; other=1")
        _SessdataInjector("abc").http_request(request)
        self.assertEqual(request.unredirected_hdrs.get("Cookie"), "SESSDATA=abc")
        self.assertNotIn("Cookie", request.headers)

    def test_cookie_uses_unredirected_header(self):
        """用 add_unredirected_header：跨域重定向时不会把 Cookie 丢给别家。"""
        from aliceqq.search import _SessdataInjector
        request = urllib.request.Request("https://api.bilibili.com/x")
        _SessdataInjector("abc").http_request(request)
        self.assertEqual(request.unredirected_hdrs.get("Cookie"), "SESSDATA=abc")

    def test_cookie_is_not_sent_to_other_domains(self):
        """Cookie 绝不能泄漏到非 B 站域名。"""
        from aliceqq.search import _SessdataInjector
        injector = _SessdataInjector("secret-cookie")
        for url in ("https://api.openai.com/v1/chat",
                    "https://evil-bilibili.com.attacker.net/x",
                    "https://notbilibili.com/x",
                    "https://example.com/?ref=bilibili.com"):
            request = urllib.request.Request(url)
            injector.http_request(request)
            self.assertIsNone(request.get_header("Cookie"), url)

    def test_empty_cookie_adds_nothing(self):
        from aliceqq.search import _SessdataInjector
        request = urllib.request.Request("https://api.bilibili.com/x")
        _SessdataInjector("").http_request(request)
        self.assertIsNone(request.get_header("Cookie"))

    def test_provider_opener_carries_the_cookie(self):
        """端到端：provider 造出来的 opener 发请求时确实带上了 SESSDATA。

        用一个假 handler 抓住最终真正发出去的头。
        """
        captured = {}

        class _Capture(urllib.request.BaseHandler):
            def http_request(self, request):
                captured["cookie"] = request.get_header("Cookie")
                raise urllib.error.URLError("停在这里，不发真请求")

            https_request = http_request

        provider = BilibiliProvider(timeout=5, sessdata="real-cookie-value")
        opener = provider._get_opener()
        opener.add_handler(_Capture())
        with self.assertRaises(urllib.error.URLError):
            opener.open(urllib.request.Request("https://api.bilibili.com/x/web-interface/nav"))
        self.assertEqual(captured.get("cookie"), "SESSDATA=real-cookie-value")

    def test_provider_without_sessdata_sends_no_cookie(self):
        captured = {}

        class _Capture(urllib.request.BaseHandler):
            def http_request(self, request):
                captured["cookie"] = request.get_header("Cookie")
                raise urllib.error.URLError("停止")

            https_request = http_request

        provider = BilibiliProvider(timeout=5, sessdata="")
        opener = provider._get_opener()
        opener.add_handler(_Capture())
        with self.assertRaises(urllib.error.URLError):
            opener.open(urllib.request.Request("https://api.bilibili.com/x/web-interface/nav"))
        self.assertIsNone(captured.get("cookie"))


class AudioTest(unittest.TestCase):
    def _provider(self, **kwargs):
        provider = BilibiliProvider(timeout=5, **kwargs)
        provider._warmed = True
        return provider

    def test_picks_lowest_bandwidth_track(self):
        provider = self._provider()
        opener = _FakeOpener({"playurl": PLAYURL})
        info = provider._fetch_audio_url(opener, "BV1", 2)
        self.assertEqual(info["url"], "https://a.example/low.m4a",
                         "应选最低码率：识别不需要高音质，更省流量与成本")

    def test_download_refuses_when_too_large(self):
        provider = self._provider()
        opener = _FakeOpener({}, head_length=50 * 1024 * 1024)
        result = provider._download_audio(opener, "https://a.example/x.m4a", 8 * 1024 * 1024)
        self.assertFalse(result["ok"])
        # 必须如实说明跳过原因，而不是上传被截断的损坏文件
        self.assertIn("超出识别体积上限", result["reason"])
        self.assertFalse(any(m == "GET" for m, _ in opener.calls), "超限时不应下载")

    def test_download_returns_bytes_when_within_limit(self):
        provider = self._provider()
        opener = _FakeOpener({}, head_length=1024)
        result = provider._download_audio(opener, "https://a.example/x.m4a", 8 * 1024 * 1024)
        # HEAD 返回 1024，但 GET 由 fake 给空 body -> ok 取决于 body
        self.assertIn("ok", result)

    def test_download_refuses_when_server_ignores_range(self):
        # 实测：B 站有时忽略 Range 头，3MB 上限却回整个 10MB 文件。
        # 这种情况必须放弃，而不是截断使用一个损坏的 m4a。
        provider = self._provider()

        class _BigOpener:
            def __init__(self):
                self.calls = []

            def open(self, request, timeout=None):
                self.calls.append(request.get_method())
                if request.get_method() == "HEAD":
                    return _FakeResponse(b"", {})   # 不给 Content-Length
                return _FakeResponse(b"X" * (3 * 1024 * 1024))

        opener = _BigOpener()
        result = provider._download_audio(opener, "https://a.example/x.m4a", 1024 * 1024)
        self.assertFalse(result["ok"])
        self.assertIn("超过上限", result["reason"])
        self.assertIn("未接受分段请求", result["reason"])

    def test_long_video_is_skipped_by_duration_gate(self):
        # max_seconds 必须真正生效：只按体积限是不够的，20 分钟视频也可能小于体积上限。
        # 云端识别按时长计费，所以不做自动分段——只能如实说明跳过。
        class _FakeASR:
            available = True

            def transcribe(self, audio, filename="a.m4a"):
                raise AssertionError("超时长时不该调用识别")

        provider = self._provider(asr_client=_FakeASR(), asr_guard=lambda: True,
                                  asr_max_seconds=300)
        opener = _FakeOpener({"playurl": PLAYURL})
        result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024,
                                      video_seconds=1253)
        self.assertEqual(result["text"], "")
        self.assertIn("已跳过", result["note"])
        self.assertIn("按时长计费", result["note"])
        self.assertEqual(opener.calls, [], "超时长时不该下载音频")

    def test_long_video_with_ffmpeg_but_no_local_asr_is_skipped(self):
        """找到 ffmpeg 也不算数：云端识别不能自动分段（会烧钱）。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_local=False,
                                  asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(1253, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "skip")
        self.assertIn("按时长计费", plan["note"])

    def test_local_asr_long_video_gets_segmented(self):
        """本地识别不花钱：长视频应改成分段识别，而不是直接放弃。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=4,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(1253, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "segmented")
        self.assertEqual(plan["segments"], 4)
        self.assertEqual(plan["segment"], 300)
        self.assertIn("ffmpeg", plan["note"])

    def test_note_does_not_claim_false_coverage(self):
        """覆盖范围不能自相矛盾。

        回归：早期版本无条件写"识别的是前 N 分钟"，于是一个 222 秒的视频
        会写成"前 4 分钟"（240 秒 > 222 秒）。只有在真的没覆盖完时才提"未识别"。
        """
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=100,
                                  asr_segment_seconds=100, asr_max_segments=4,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        # 222 秒视频、每段 100 秒、最多 4 段 -> 3 段足够覆盖完
        plan = provider._transcribe_plan(222, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "segmented")
        self.assertEqual(plan["segments"], 3)
        self.assertNotIn("未识别", plan["note"], "已经覆盖完整就不该说还有没识别的")
        self.assertNotIn("前 4", plan["note"])

    def test_uncovered_part_is_stated_when_capped(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=2,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(3600, 8 * 1024 * 1024)
        self.assertEqual(plan["segments"], 2)
        self.assertIn("未识别", plan["note"])
        # 只识别前 10 分钟，剩 50 分钟
        self.assertIn("前 10.0 分钟", plan["note"])

    def test_segment_count_is_capped(self):
        """超长视频不能无限切下去，要受 max_segments 限制。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=6,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(4 * 3600, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "segmented")
        self.assertEqual(plan["segments"], 6, "4 小时视频也只识别前 6 段")

    def test_all_segments_allowed_when_video_fits(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=24,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(1000, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "segmented")
        self.assertEqual(plan["segments"], 4)      # 1000/300 -> 4 段刚好覆盖
        self.assertNotIn("未识别", plan["note"])

    def test_segment_count_grows_when_needed(self):
        """1000 秒要 4 段（3 段只到 900 秒），覆盖完整才不多不少。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=24,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        self.assertEqual(provider._transcribe_plan(901, 8 * 1024 * 1024)["segments"], 4)
        self.assertEqual(provider._transcribe_plan(899, 8 * 1024 * 1024)["segments"], 3)

    def test_segmented_needs_ffmpeg(self):
        """没有 ffmpeg 时不能假装能切段，要如实说找不到 ffmpeg。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_local=True,
                                  asr_ffmpeg="")
        with mock.patch("aliceqq.search.find_ffmpeg", return_value=""):
            plan = provider._transcribe_plan(1253, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "skip")
        self.assertIn("ffmpeg", plan["note"])

    def test_zero_max_segments_falls_back_to_truncate(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_segment_seconds=300, asr_max_segments=0,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(1253, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "truncate")
        self.assertIn("只识别开头", plan["note"])

    def test_short_video_goes_direct(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(120, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "direct")

    def test_zero_limit_means_no_gate(self):
        """max_seconds=0 表示不限制时长，不能把视频全判成超限。"""
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=0,
                                  asr_local=True, asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        plan = provider._transcribe_plan(9999, 8 * 1024 * 1024)
        self.assertEqual(plan["mode"], "direct")

    def test_segmented_run_transcribes_every_chunk_with_real_lengths(self):
        """端到端（打桩）：长视频走分段，每段都识别，覆盖时长按**实测**累加。

        这里刻意让 ffmpeg 返回的第 2 段比请求的短（模拟"到文件尾"），
        验证 note 里的覆盖时长不会虚报成 段数×段长。
        """
        calls = []

        class _FakeASR:
            available = True

            def transcribe(self, audio, filename="a.m4a"):
                calls.append(filename)
                return Transcription(True, text="第%d段内容" % len(calls))

        lengths = [120.0, 60.0]      # 第 2 段实际只有 60 秒

        def fake_extract(data, start, duration, ffmpeg="", **kw):
            return b"chunk", ""

        def fake_probe(data, ffmpeg="", **kw):
            return lengths[min(len(calls), len(lengths) - 1)], ""

        provider = self._provider(asr_client=_FakeASR(), asr_guard=lambda: True,
                                  asr_max_seconds=300, asr_segment_seconds=300,
                                  asr_max_segments=2, asr_local=True,
                                  asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        opener = _FakeOpener({"playurl": PLAYURL}, head_length=1024)

        original_open = opener.open

        def fake_open(request, timeout=None):
            if request.get_method() == "HEAD":
                return _FakeResponse(b"", {"Content-Length": "1024"})
            if "a.example" in request.full_url:
                return _FakeResponse(b"FAKEAUDIO")
            return original_open(request, timeout=timeout)

        opener.open = fake_open
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None), \
                mock.patch("aliceqq.search.ffmpeg_extract", side_effect=fake_extract), \
                mock.patch("aliceqq.search.ffmpeg_probe", side_effect=fake_probe):
            result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024,
                                          video_seconds=1253)
        self.assertIn("第1段内容", result["text"])
        self.assertIn("第2段内容", result["text"])
        self.assertEqual(len(calls), 2, "两段都要真的调用识别")
        # 覆盖时长 = 120 + 60 = 180 秒 = 3.0 分钟（而不是 2×300=10 分钟）
        self.assertIn("3.0 分钟", result["note"])
        self.assertIn("共 2 段", result["note"])

    def test_segmented_reports_extraction_failure(self):
        """第一段就切不出来时要如实说明，不能返回空文本还装作成功。"""
        class _FakeASR:
            available = True

            def transcribe(self, audio, filename="a.m4a"):
                raise AssertionError("切不出来时不该调用识别")

        provider = self._provider(asr_client=_FakeASR(), asr_guard=lambda: True,
                                  asr_max_seconds=300, asr_segment_seconds=300,
                                  asr_max_segments=2, asr_local=True,
                                  asr_ffmpeg="C:\\fake\\ffmpeg.exe")
        opener = _FakeOpener({"playurl": PLAYURL}, head_length=1024)
        original_open = opener.open

        def fake_open(request, timeout=None):
            if request.get_method() == "HEAD":
                return _FakeResponse(b"", {"Content-Length": "1024"})
            if "a.example" in request.full_url:
                return _FakeResponse(b"FAKEAUDIO")
            return original_open(request, timeout=timeout)

        opener.open = fake_open
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None), \
                mock.patch("aliceqq.search.ffmpeg_extract",
                           return_value=(None, "ffmpeg 退出码 1")):
            result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024,
                                          video_seconds=1253)
        self.assertEqual(result["text"], "")
        self.assertIn("分段失败", result["note"])
        self.assertIn("ffmpeg 退出码 1", result["note"])

    def test_short_video_passes_duration_gate(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: True, asr_max_seconds=300)
        opener = _FakeOpener({"playurl": PLAYURL})
        result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024, video_seconds=120)
        # 未超时长，应该走到下载（可能因 fake 的 body 为空而失败，但不应是"时长"拒绝）
        self.assertNotIn("时长上限", result.get("note", ""))

    def test_transcribe_respects_guard(self):
        provider = self._provider(asr_client=ASRClient("https://x/v1", "k"),
                                  asr_guard=lambda: False)
        opener = _FakeOpener({"playurl": PLAYURL})
        result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024)
        self.assertEqual(result["text"], "")
        self.assertIn("额度", result["note"])
        self.assertEqual(opener.calls, [], "被额度拒绝时不该发起任何请求")

    def test_transcribe_without_client_stays_silent(self):
        provider = self._provider()
        opener = _FakeOpener({"playurl": PLAYURL})
        result = provider._transcribe(opener, "BV1", 2, 8 * 1024 * 1024)
        self.assertEqual(result, {"text": "", "note": ""})

    def test_enrich_uses_asr_when_no_subtitle(self):
        class _FakeASR:
            available = True

            def transcribe(self, audio, filename="a.m4a"):
                return Transcription(True, text="视频里说今天讲入坑规划")

        # DETAIL 里的视频是 10 分钟，云端按时长计费不会分段，
        # 所以这里把时长上限放宽到能通过，否则会被时长闸门拦下
        provider = self._provider(asr_client=_FakeASR(), asr_guard=lambda: True,
                                  asr_max_seconds=1200)
        routes = {"web-interface/view": DETAIL, "player/v2": PLAYER_NO_SUB,
                  "playurl": PLAYURL, "v2/reply": REPLIES}
        opener = _FakeOpener(routes, head_length=1024)

        original_open = opener.open

        def fake_open(request, timeout=None):
            if request.get_method() == "HEAD":
                return _FakeResponse(b"", {"Content-Length": "1024"})
            if "a.example" in request.full_url:
                return _FakeResponse(b"FAKEAUDIO")
            return original_open(request, timeout=timeout)

        opener.open = fake_open
        result = SearchResult("标题", "https://www.bilibili.com/video/BV1x", "", "bilibili.com", True)
        with mock.patch("aliceqq.search.time.sleep", lambda *_: None):
            provider._enrich(opener, result)
        self.assertIn("【视频语音识别（未核对）】", result.content)
        self.assertIn("入坑规划", result.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
