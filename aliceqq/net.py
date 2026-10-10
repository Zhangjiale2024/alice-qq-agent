"""统一的网络出口：系统代理不通时自动直连。

## 为什么需要这一层（都是实测）

这台机器的 Windows 系统代理（127.0.0.1:11808）**反复掉线**。
urllib 默认会去读注册表里的这个代理，代理一挂，所有请求立刻报
WinError 10061（由于目标计算机积极拒绝）：

    LLM（DeepSeek）  -> 连不上，机器人变哑巴
    联网检索（Bocha）-> 连不上
    语音合成（百度）  -> 连不上
    而国内站点直连其实是通的

**一条代理配置，拖垮全部能力。**

## 这一层做什么

1. 按顺序尝试多个出口：系统代理、直连（以及可选的本地代理）；
2. 记住最近能用的那个，短时间内优先用它 —— 避免每次请求都先撞一遍坏代理；
3. 用过的出口失败时自动切换，不需要任何配置改动。

## 它做不到什么（不要误解）

它**不能**让境外站点变得可用。那些站点确实需要能上外网的代理；
代理真挂了，DeepSeek 依然连不上。这一层保证的是
**代理挂掉时国内能力（B站检索、百度语音等）不受影响**，
而不是"绕过网络限制"。
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

# 同一个出口连续失败多久之后，允许再次把它排到前面（秒）
REPROBE_SECONDS = 60.0
_LOCK = threading.Lock()
_preferred: Optional[str] = None
_preferred_at = 0.0

# 本地直连代理（可选，见 GPT-SoVITS-setup/local_proxy.py）
LOCAL_PROXY = "http://127.0.0.1:8899"


def _build_routes() -> List[Tuple[str, urllib.request.OpenerDirector]]:
    """构造候选出口。顺序即默认优先级。"""
    routes: List[Tuple[str, urllib.request.OpenerDirector]] = []

    # ① 系统代理：境外站点（DeepSeek/Bocha）只有它能通
    routes.append(("system", urllib.request.build_opener()))

    # ② 本地直连代理：起了就用，它本身是直连的，能过国内站点
    proxy_handler = urllib.request.ProxyHandler({"http": LOCAL_PROXY,
                                                 "https": LOCAL_PROXY})
    routes.append(("local_proxy", urllib.request.build_opener(proxy_handler)))

    # ③ 强制直连：国内站点通常都能通，且不受系统代理影响
    routes.append(("direct",
                   urllib.request.build_opener(urllib.request.ProxyHandler({}))))
    return routes


_ROUTES = _build_routes()
_ROUTE_MAP = {name: opener for name, opener in _ROUTES}


def route_names() -> List[str]:
    return [name for name, _ in _ROUTES]


def _order() -> List[Tuple[str, urllib.request.OpenerDirector]]:
    """把上次成功的出口排到最前面。

    没有这一条的话，每次请求都要先撞一遍坏代理，白白多等一个超时。
    """
    with _LOCK:
        preferred = _preferred
        fresh = (time.time() - _preferred_at) < REPROBE_SECONDS
    if not preferred or not fresh:
        return list(_ROUTES)
    head = [(n, o) for n, o in _ROUTES if n == preferred]
    tail = [(n, o) for n, o in _ROUTES if n != preferred]
    return head + tail


def _remember(name: str) -> None:
    global _preferred, _preferred_at
    with _LOCK:
        _preferred = name
        _preferred_at = time.time()


def current_route() -> Optional[str]:
    """最近一次成功的出口名，没成功过返回 None。"""
    with _LOCK:
        return _preferred


def reset_route() -> None:
    """清掉记住的出口（测试用，或想让下次重新探测）。"""
    global _preferred, _preferred_at
    with _LOCK:
        _preferred = None
        _preferred_at = 0.0


def open(request, timeout: float = 30, routes: Optional[List[str]] = None):
    """按优先级尝试各出口，返回第一个成功的响应。

    失败时抛最后一个异常，保持和 urllib.request.urlopen 一致的行为，
    调用方现有的 except 分支不用改。
    """
    if isinstance(request, str):
        request = urllib.request.Request(request)
    candidates = _order()
    if routes:
        wanted = set(routes)
        candidates = [(n, o) for n, o in candidates if n in wanted] or candidates

    last_error: Optional[Exception] = None
    for name, opener in candidates:
        try:
            response = opener.open(request, timeout=timeout)
            _remember(name)
            return response
        except urllib.error.HTTPError:
            # 服务器**有响应**（403/404 等）说明网络是通的，
            # 这不是出口问题，不该换出口重试，直接抛给调用方。
            _remember(name)
            raise
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise urllib.error.URLError("没有可用的网络出口")


def urlopen(url, data=None, timeout: float = 30, headers: Optional[dict] = None):
    """urlopen 的替代品，签名兼容常见用法。"""
    request = urllib.request.Request(url, data=data, headers=headers or {})
    return open(request, timeout=timeout)
