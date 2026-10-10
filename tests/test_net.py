"""网络出口自动降级的测试。

背景：这台机器的系统代理反复掉线，而 urllib 默认走它，代理一挂
LLM/检索/语音全部 WinError 10061。net 层负责在出口之间自动切换。
"""
from __future__ import annotations

import unittest
import urllib.error
from unittest import mock

from aliceqq import net


class _FakeResponse:
    def __init__(self, body=b"ok"):
        self.body = body

    def read(self, *args):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(result):
    """造一个 opener：result 是异常就抛，否则返回它。"""
    opener = mock.MagicMock()
    if isinstance(result, Exception):
        opener.open.side_effect = result
    else:
        opener.open.return_value = result
    return opener


class RouteFallbackTest(unittest.TestCase):
    def setUp(self):
        net.reset_route()
        self.saved = net._ROUTES

    def tearDown(self):
        net._ROUTES = self.saved
        net.reset_route()

    def _use(self, routes):
        net._ROUTES = list(routes)

    def test_uses_first_working_route(self):
        good = _opener(_FakeResponse())
        self._use([("system", good), ("direct", _opener(OSError("不该走到这")))])
        with net.open("http://example.com") as resp:
            self.assertEqual(resp.read(), b"ok")
        self.assertEqual(net.current_route(), "system")

    def test_falls_back_when_system_proxy_refused(self):
        """核心场景：系统代理拒绝连接（WinError 10061），应自动改走直连。"""
        refused = ConnectionRefusedError(10061, "由于目标计算机积极拒绝，无法连接。")
        direct = _opener(_FakeResponse(b"direct"))
        self._use([("system", _opener(refused)), ("direct", direct)])
        with net.open("http://example.com") as resp:
            self.assertEqual(resp.read(), b"direct")
        self.assertEqual(net.current_route(), "direct")

    def test_remembers_working_route(self):
        """记住可用出口，下次不再先撞坏的那个。"""
        refused = _opener(ConnectionRefusedError(10061, "refused"))
        direct = _opener(_FakeResponse())
        self._use([("system", refused), ("direct", direct)])
        net.open("http://a")
        net.open("http://b")
        # 第二次应该直接用 direct，system 不该被再试
        self.assertEqual(refused.open.call_count, 1)
        self.assertEqual(direct.open.call_count, 2)

    def test_http_error_does_not_switch_route(self):
        """服务器有响应（403/404）说明网络通，不该换出口重试。"""
        err = urllib.error.HTTPError("http://x", 403, "Forbidden", {}, None)
        system = _opener(err)
        direct = _opener(_FakeResponse())
        self._use([("system", system), ("direct", direct)])
        with self.assertRaises(urllib.error.HTTPError):
            net.open("http://x")
        self.assertEqual(direct.open.call_count, 0)

    def test_all_routes_fail_raises_last_error(self):
        first = ConnectionRefusedError(10061, "refused")
        second = OSError("也断了")
        self._use([("system", _opener(first)), ("direct", _opener(second))])
        with self.assertRaises(OSError) as ctx:
            net.open("http://x")
        self.assertEqual(str(ctx.exception), "也断了")

    def test_reset_route_forgets_preference(self):
        refused = _opener(ConnectionRefusedError(10061, "refused"))
        direct = _opener(_FakeResponse())
        self._use([("system", refused), ("direct", direct)])
        net.open("http://a")
        net.reset_route()
        self.assertIsNone(net.current_route())

    def test_stale_preference_is_reprobed(self):
        """记住的出口过了一段时间就不再优先，允许重新探测。"""
        good = _opener(_FakeResponse())
        self._use([("system", good), ("direct", _opener(_FakeResponse()))])
        net.open("http://a")
        self.assertEqual(net.current_route(), "system")
        with mock.patch.object(net, "REPROBE_SECONDS", -1):
            order = net._order()
        self.assertEqual(order[0][0], "system")   # 顺序退回默认，而不是空列表

    def test_route_names_available(self):
        names = net.route_names()
        self.assertIn("system", names)
        self.assertIn("direct", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
