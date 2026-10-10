"""联网检索：真实可用的搜索 + 网页正文提取 + 来源标注。

设计要点：
1. 零配置可用：默认走 DuckDuckGo 的 HTML 端点（无需 API Key 与注册）；
   也支持 SearXNG JSON（自建或公共实例）作为更稳定的选择。
2. 纯标准库实现：urllib 发请求，html.parser 解析，不引入 bs4/requests。
3. 域名策略：拦截内网/危险地址，标注高可信来源。
4. 结果永远带 URL：模型必须能给出出处，否则无法自证不是编的。
5. 缓存：同一 query 在 TTL 内不重复请求，省时间也省对方服务器。
"""
from __future__ import annotations

from . import net
from .asr import quality_warning
from .ffmpeg import extract_segment as ffmpeg_extract
from .ffmpeg import find_ffmpeg
from .ffmpeg import probe_duration as ffmpeg_probe
import base64
import gzip
import hashlib
import html
import http.cookiejar
import ipaddress
import json
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 可信来源（给出的结果会标注来源类型，便于模型区分权威性）
TRUSTED_DOMAINS = (
    "wikipedia.org", "baike.baidu.com", "zhihu.com", "github.com", "arxiv.org",
    "bluearchive.wiki", "bluearchive.fandom.com", "moegirl.org.cn", "bilibili.com",
    "nyato.com", "game8.jp", "appmedia.jp", "gov.cn", "edu.cn",
)
# 不值得抓正文的来源
LOW_VALUE_DOMAINS = (
    "pinterest.", "facebook.com", "instagram.com", "tiktok.com",
    "doubleclick.net", "googleadservices.com", "googlesyndication.com",
    # 搜索引擎自家的 AI 聚合页：没有可读正文，且会占掉结果位
    "ai.so.com", "so.com/link",
    # 图片/短视频/排行榜聚合站：内容与问题无关，只是把关键词拼进标题
    "tv.360kan.com", "ranks.hao.360.com", "image.so.com", "so.com/image",
    "hao.360.com", "v.qq.com/x/page", "kuaishou.com",
)


def looks_like_aggregator(url: str) -> bool:
    """图片/视频/排行聚合页：标题里有关键词但内容不是答案。"""
    low = (url or "").lower()
    markers = ("360kan.com", "ranks.hao", "image.so", "/image?", "baike.so.com/doc",
               "so.com/s?q=", "sogou.com/web?query=")
    return any(marker in low for marker in markers)
# 内网/本机地址一律拒绝，避免被用来探测内网
BLOCKED_HOST_RE = re.compile(r"^(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?$)",
                             re.IGNORECASE)

MAX_SNIPPET_CHARS = 300
# 正文给足：实测 TapTap 这类页面导航占掉前 3000 字，给少了只剩菜单、
# 拿不到真正包含"今天/本月"日期的帖子标题与正文。
MAX_PAGE_CHARS = 6000
MAX_RESULTS = 5
# 链式检索的最低相关度：低于它就会继续尝试下一个源。
# 0.34 约等于"至少覆盖 1/3 的查询词"，能挡掉"基（汉字）"这类无关结果。
CHAIN_RELEVANCE_MIN = 0.34
# 单次检索最多抓几页正文、总共给模型多少正文
MAX_FETCH_PAGES = 3
MAX_TOTAL_CONTENT_CHARS = 14000
# 页面 JSON 里常见的时间戳（lofter/TapTap 等前端框架会内嵌结构化数据）
JSON_DATE_RE = re.compile(r'"(?:createTime|created_at|published_at|postTime|time)"\s*:\s*"?'
                          r'(20\d{2}[-/]\d{1,2}[-/]\d{1,2})')
JSON_TITLE_RE = re.compile(r'"(?:title|subject|name|postTitle)"\s*:\s*"([^"]{6,80})"')


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    trusted: bool = False
    content: str = ""          # 抓取到的正文（可选）

    def to_dict(self) -> Dict[str, object]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet,
                "source": self.source, "trusted": self.trusted}


@dataclass
class SearchOutcome:
    ok: bool
    query: str
    provider: str = ""
    results: List[SearchResult] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0

    @property
    def empty(self) -> bool:
        return not self.results


class _DuckDuckGoParser(HTMLParser):
    """解析 DuckDuckGo HTML 版的结果块。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._current: Dict[str, str] = {}
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "") or ""
        if tag == "a" and "result__a" in classes:
            self._in_link = True
            self._buffer = []
            self._current = {"url": attrs_dict.get("href", ""), "title": "", "snippet": ""}
        elif tag == "a" and "result__snippet" in classes:
            self._in_snippet = True
            self._buffer = []
        elif self._in_snippet and tag in ("b", "span"):
            pass

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            self._current["title"] = "".join(self._buffer).strip()
            self._buffer = []
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            if self._current:
                self._current["snippet"] = "".join(self._buffer).strip()
                self.results.append(self._current)
                self._current = {}
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_snippet:
            self._buffer.append(data)


class _TextExtractor(HTMLParser):
    """从网页里抽可见正文，丢掉脚本/样式/导航。"""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}
    BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: List[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK_TAGS and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)
                self._chunks.append(" ")

    def text(self) -> str:
        joined = "".join(self._chunks)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n\s*\n+", "\n", joined)
        return joined.strip()


# ---------------------------------------------------------------------------
# 安全与工具函数
# ---------------------------------------------------------------------------

def is_safe_url(url: str) -> bool:
    """拒绝非 http(s) 与内网/本机地址。"""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip()
    if not host or BLOCKED_HOST_RE.match(host):
        return False
    # 纯 IP 也要拦内网段
    try:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return False
    except ValueError:
        pass
    return True


def domain_of(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_trusted(url: str) -> bool:
    domain = domain_of(url)
    return any(domain.endswith(d) or d in domain for d in TRUSTED_DOMAINS)


def is_low_value(url: str) -> bool:
    domain = domain_of(url)
    return any(word in domain for word in LOW_VALUE_DOMAINS)


# 检索消歧：实测查"蔚蓝档案 最新活动"时，Bing 会把"蔚蓝"理解成独立游戏《Celeste》，
# 返回一堆无关结果。补上手游语境即可命中正确目标。
DISAMBIGUATION = (
    (re.compile(r"蔚蓝档案|碧蓝档案"), "蔚蓝档案 手游"),
    (re.compile(r"^(蔚蓝|碧蓝)(?!档案)"), "蔚蓝档案 手游"),
    (re.compile(r"\bba\b", re.IGNORECASE), "蔚蓝档案"),
)


# 是否需要消歧取决于检索源：360 能正确理解"蔚蓝档案"，
# 而 cn.bing.com 会把它拆成"蔚蓝"并命中《Celeste》。
# 所以这里保留能力但默认不启用（由 SEARCH_DISAMBIGUATE 控制）。
_DISAMBIGUATE_ENABLED = False


def set_disambiguation(enabled: bool) -> None:
    global _DISAMBIGUATE_ENABLED
    _DISAMBIGUATE_ENABLED = bool(enabled)


def disambiguate(query: str) -> str:
    if not _DISAMBIGUATE_ENABLED:
        return query
    cleaned = query
    for pattern, replacement in DISAMBIGUATION:
        if pattern.search(cleaned):
            cleaned = pattern.sub(replacement, cleaned, count=1)
            break
    return cleaned


def clean_query(text: str, max_len: int = 120) -> str:
    """把口语问题压成搜索词。"""
    cleaned = re.sub(r"\[CQ:[^\]]*\]", " ", text or "")
    cleaned = re.sub(r"^(爱丽丝|alice|助手|请|帮我|麻烦|你能不能|可以|能不能)+\s*", "", cleaned.strip(),
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"(帮我)?(搜一下|搜索一下|查一下|查查|上网查|百度一下|google一下)", " ", cleaned)
    cleaned = re.sub(r"[？?！!。，,、；;：:~～]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return disambiguate(cleaned)[:max_len]


def _decode_body(raw: bytes, headers) -> str:
    if headers.get("Content-Encoding") == "gzip":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
    for encoding in ("utf-8", "gb18030", "big5", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def http_get(url: str, timeout: int = 15, max_bytes: int = 400_000) -> Tuple[int, str, str]:
    """返回 (状态码, 正文, 最终URL)。失败抛异常由调用方处理。"""
    if not is_safe_url(url):
        raise ValueError(f"拒绝请求不安全的地址：{url}")
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    })
    with net.open(request, timeout=timeout) as response:
        raw = response.read(max_bytes)
        return response.status, _decode_body(raw, response.headers), response.geturl()


def _unwrap_ddg_url(url: str) -> str:
    """DuckDuckGo 的结果链接常带跳转前缀，取出真实地址。"""
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    if "duckduckgo.com" in (parsed.hostname or ""):
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get("uddg")
        if target:
            return urllib.parse.unquote(target[0])
    return url


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

# 搜索引擎的"访问被拦截"页面特征。识别它有两个作用：
# 1. 如实报告（而不是误报成"页面结构已变化"，让人以为改改解析就行）；
# 2. 触发冷却，避免继续请求把封禁时间拉长。
BLOCK_PAGE_MARKERS = (
    "访问异常页面", "百度安全验证", "请输入验证码", "captcha",
    "wappass.baidu.com", "security-verify", "验证码", "人机验证",
    "请求过于频繁", "Access Denied", "unusual traffic",
)


def looks_blocked(body: str) -> bool:
    if not body:
        return False
    head = body[:8000]
    title_match = re.search(r"<title>(.*?)</title>", head, re.S | re.I)
    title = title_match.group(1) if title_match else ""
    if any(marker in title for marker in ("访问异常", "安全验证", "验证码", "Access Denied")):
        return True
    # 正文很短且出现验证类关键词，基本就是拦截页
    if len(body) < 20000 and sum(1 for m in BLOCK_PAGE_MARKERS if m in body) >= 2:
        return True
    return False


class _DomainGuard:
    """按域名做最小间隔 + 封禁冷却。

    为什么要它：实测中我因为短时间大量请求被 360 判定为异常，
    返回「访问异常页面」，之后所有查询都拿不到结果——
    表现就是"搜索结果更老了"（因为退到了更差的源）。
    所以必须自我限速，并在检测到封禁后停手一段时间。
    """

    # 递进式冷却：实测 360 的拦截是间歇性的（同一次测试里第 1 次成功、第 2 次被拦），
    # 一次被拦就罚 10 分钟会让检索长时间不可用。改为逐次加码，
    # 只有持续被拦才退到长冷却。
    COOLDOWN_LADDER = (30.0, 120.0, 300.0, 900.0)

    def __init__(self, min_interval: float = 1.5):
        self.min_interval = min_interval
        self._last_at: Dict[str, float] = {}
        self._blocked_until: Dict[str, float] = {}
        self._block_count: Dict[str, int] = {}

    def blocked_for(self, domain: str) -> float:
        until = self._blocked_until.get(domain, 0.0)
        remaining = until - time.time()
        return remaining if remaining > 0 else 0.0

    def note_block(self, domain: str) -> float:
        """记一次拦截并按次数加码冷却，返回本次冷却秒数。"""
        count = self._block_count.get(domain, 0)
        cooldown = self.COOLDOWN_LADDER[min(count, len(self.COOLDOWN_LADDER) - 1)]
        self._block_count[domain] = count + 1
        self._blocked_until[domain] = time.time() + cooldown
        return cooldown

    def note_success(self, domain: str) -> None:
        """成功一次就把加码重置，避免偶发拦截累积成长期封禁。"""
        self._block_count[domain] = 0

    def acquire(self, domain: str, *, max_wait: float = 3.0) -> None:
        """必要时等待，保证对同一域名的请求间隔不小于 min_interval。"""
        last = self._last_at.get(domain, 0.0)
        wait = self.min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(min(wait, max_wait))
        self._last_at[domain] = time.time()


_DOMAIN_GUARD = _DomainGuard()


class _Cache:
    def __init__(self, ttl: int = 600):
        self.ttl = ttl
        self._data: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str):
        item = self._data.get(key)
        if not item:
            return None
        stamp, value = item
        if time.time() - stamp > self.ttl:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value) -> None:
        self._data[key] = (time.time(), value)


class SearchProvider:
    name = "base"

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """零配置默认：DuckDuckGo 的 HTML 端点。"""

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        url = f"{self.ENDPOINT}?q={urllib.parse.quote(query)}"
        try:
            status, body, _ = http_get(url, timeout=self.timeout)
        except Exception as exc:
            return SearchOutcome(False, query, self.name, error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))
        if status != 200:
            return SearchOutcome(False, query, self.name, error=f"HTTP {status}",
                                 elapsed=round(time.time() - started, 2))

        parser = _DuckDuckGoParser()
        parser.feed(body)
        results: List[SearchResult] = []
        for item in parser.results:
            target = _unwrap_ddg_url(item.get("url", ""))
            title = html.unescape(item.get("title", "")).strip()
            if not target or not title or not is_safe_url(target):
                continue
            results.append(SearchResult(
                title=title,
                url=target,
                snippet=html.unescape(item.get("snippet", ""))[:MAX_SNIPPET_CHARS],
                source=domain_of(target),
                trusted=is_trusted(target),
            ))
            if len(results) >= limit:
                break
        # 可信来源优先排在前面
        _DOMAIN_GUARD.note_success("so.com")
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results=results,
                             elapsed=round(time.time() - started, 2))


class So360Provider(SearchProvider):
    """默认检索源：360 搜索（www.so.com）。

    为什么选它：实测对比了几个源之后，
      - DuckDuckGo HTML：返回反爬页（HTTP 202，零结果），已不可用；
      - www.bing.com：把中文查询按 gb2312 误解码；cn.bing.com 虽可用，
        但当查询是系列名（如"蔚蓝档案 最新活动"）时会被分词成"蔚蓝"，
        命中独立游戏《Celeste》，结果与目标无关；
      - 360：能正确命中"蔚蓝档案社区活动""九游蔚蓝档案活动"等目标页面。
    结果块带 data-mdurl，可直接取到真实地址，无需破解跳转链接。
    """

    name = "so360"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        remaining = _DOMAIN_GUARD.blocked_for("so.com")
        if remaining > 0:
            return SearchOutcome(False, query, self.name,
                                 error=f"处于封禁冷却中（还需 {int(remaining)} 秒）",
                                 elapsed=round(time.time() - started, 2))
        _DOMAIN_GUARD.acquire("so.com")
        url = f"https://www.so.com/s?q={urllib.parse.quote(query, encoding='utf-8')}"
        try:
            status, body, _ = http_get(url, timeout=self.timeout)
        except Exception as exc:
            return SearchOutcome(False, query, self.name, error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))
        if looks_blocked(body):
            cooldown = _DOMAIN_GUARD.note_block("so.com")
            return SearchOutcome(False, query, self.name,
                                 error=f"被 360 判定为异常访问（返回「访问异常页面」），"
                                       f"进入 {int(cooldown)} 秒冷却。"
                                       "这通常是因为短时间内请求过多。",
                                 elapsed=round(time.time() - started, 2))
        if status != 200:
            return SearchOutcome(False, query, self.name, error=f"HTTP {status}",
                                 elapsed=round(time.time() - started, 2))

        results: List[SearchResult] = []
        for block in _split_360_blocks(body):
            title = _360_title(block)
            target = _360_url(block)
            if not title or not target or not is_safe_url(target):
                continue
            if is_low_value(target) or looks_like_aggregator(target):
                continue
            if _too_off_topic(title, query):
                continue
            results.append(SearchResult(
                title=title, url=target, snippet=_360_snippet(block),
                source=domain_of(target), trusted=is_trusted(target)))
            if len(results) >= limit:
                break
        if not results:
            return SearchOutcome(False, query, self.name,
                                 error="未能从 www.so.com 解析出结果（页面结构可能已变化）",
                                 elapsed=round(time.time() - started, 2))
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results=results,
                             elapsed=round(time.time() - started, 2))


# 上文话题挖掘用的已知专名。用户追问"现在有什么活动？"时，
# 靠这张表把上文的"蔚蓝档案"捞出来补进检索词。
# 之所以不用分词：中文原文没有空格，按空白切词会把整句当成一个词。
KNOWN_TOPICS = (
    "蔚蓝档案", "碧蓝档案", "blue archive", "爱丽丝", "アリス",
    "明日方舟", "原神", "崩坏", "星穹铁道", "绝区零", "少女前线", "碧蓝航线",
    "公主连结", "赛马娘", "FGO", "阴阳师", "鸣潮", "尘白禁区", "无期迷途",
    "千年科学学园", "游戏开发部", "基沃托斯",
    "Steam", "Switch", "PS5", "Xbox", "王者荣耀", "英雄联盟", "CS2", "瓦罗兰特",
)

# 中文复合词：连续汉字且以"档案/游戏/学园/铁道/方舟"等结尾，多为作品名/机构名
# 前缀取 2-3 字：{3,3} 会漏掉"蔚蓝档案"这类 2+2 的作品名，
# 而固定 {2} 又会把"封印者档案"截成"印者档案"。
_COMPOUND_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,3}(?:档案|游戏|学园|铁道|方舟|航线|连结|少女|学院|计划|世界|幻想|物语))")


def mine_topic(text: str, exclude: str = "") -> str:
    """从一段文本里挖出可检索的话题词。

    优先用已知专名表（准确），其次用复合词模式（覆盖未登记的作品名）。
    """
    low = (text or "").lower()
    for topic in KNOWN_TOPICS:
        if topic.lower() in low and topic not in (exclude or ""):
            return topic
    match = _COMPOUND_RE.search(text or "")
    if match:
        candidate = match.group(1)
        if candidate not in (exclude or ""):
            return candidate
    return ""


# 疑问/请求性套话：去掉它们剩下的才是"话题核心"。
# 没有这一步，"什么是量子纠缠"会变成一个整句词，
# 逐条过滤时要求整句出现，好结果反而被删掉。
_QUESTION_PREFIXES = (
    "请问", "请帮我", "帮我", "我想知道", "我想问", "你知道", "能不能", "可以",
    "解释一下", "介绍一下", "讲一下", "说一下", "告诉我",
    "什么是", "什么是叫做", "啥是", "什么叫",
)
_QUESTION_SUFFIXES = (
    "是什么", "有什么", "有哪些", "是什么东西", "是什么呀", "怎么做", "怎么办",
    "怎么样", "什么样", "为什么", "吗", "呢", "啊", "呀", "吧",
)


def char_bigrams(text: str) -> set:
    """取字符二元组（去掉标点空白）。中文没有分词也能用它衡量文本相似度。"""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text or "")
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def text_overlap(text: str, reference: str) -> float:
    """text 与 reference 的字符二元组重合度（0~1）。"""
    ref = char_bigrams(reference)
    if not ref:
        return 1.0
    return len(ref & char_bigrams(text)) / len(ref)


def transcript_matches_topic(text: str, reference: str, min_chars: int = 40) -> bool:
    """粗略判断一段字幕/识别文本是否与本视频主题有关。

    **这只是提示信号，不是闸门。** 踩过的坑：

    B 站搬运/标题党很常见——实测标题写着"蔚蓝档案攻略"的视频，
    其 AI 字幕讲的是**健身分化训练**、**电动车电池**、**麦当劳员工餐成本**。

    一开始我用"字符二元组零重合"当闸门直接丢弃。但那不可靠：
    长文本里"活动""都会"这类高频二元组必然偶然撞车，
    于是麦当劳的字幕也能"通过"，而纯游戏解说的正确字幕反而可能被误杀。
    阈值调来调去都站不住，所以改成：**文本照样给，但明确标注未核对**，
    发现用词与主题毫无重合时再额外加一句警告，交给模型判断。
    """
    body = (text or "").strip()
    if len(body) < min_chars:
        return True   # 太短不判定，避免误杀
    return bool(char_bigrams(body) & char_bigrams(reference))


def question_core(text: str) -> str:
    """从口语提问里取出核心短语。

    "什么是量子纠缠" -> "量子纠缠"
    "蔚蓝档案最近有什么活动" -> "蔚蓝档案最近活动"（再由其它规则细分）
    """
    core = (text or "").strip()
    for _ in range(3):
        changed = False
        for prefix in _QUESTION_PREFIXES:
            if core.startswith(prefix):
                core = core[len(prefix):]
                changed = True
        for suffix in _QUESTION_SUFFIXES:
            if core.endswith(suffix) and len(core) > len(suffix):
                core = core[: -len(suffix)]
                changed = True
        if not changed:
            break
    return core.strip("的了吗呢啊呀吧，,。？！?! ")


def relevance_terms(query: str) -> List[str]:
    """用于相关度判定的词；判定不了时返回空列表表示"这次不判"。

    有些口语查询（如"今天的科技新闻"）既无空格、又匹配不到专名、
    也剥不掉疑问套话，此时只剩整句一个"词"——拿它去比对标题必然全不命中，
    会把当天新闻误标成"相关度较低"。这种情况宁可不判，也不要误判。
    """
    terms = _query_terms(query)
    if not terms:
        return []
    if len(terms) == 1:
        term = terms[0]
        # 单个词几乎等于整句 -> 粒度太粗，判不了
        cleaned = re.sub(r"[\s，,。？！?!、]+", "", query or "")
        if len(term) >= 6 and len(term) >= len(cleaned) * 0.8:
            return []
    return terms


def _query_terms(query: str) -> List[str]:
    """把查询拆成有区分度的词，用于判断结果是否跑偏。

    关键：中文提问常常**不带空格**（"蔚蓝档案最近有什么活动？"），
    只按空白切会得到一整句，导致相关度恒为 0、被误判成"低相关"。
    所以对每个过长片段再用 mine_topic 抽出专名。
    """
    text = query or ""
    terms = [t for t in re.split(r"[\s,，、]+", text) if len(t) >= 2]
    # 对没空格的整句，补上两种"可判定的词"：
    #   1) 专名（如"蔚蓝档案"）
    #   2) 去掉疑问套话后的核心短语（如"什么是量子纠缠" -> "量子纠缠"）
    for term in list(terms):
        if len(term) >= 4:
            topic = mine_topic(term)
            if topic and topic not in terms:
                terms.append(topic)
            core = question_core(term)
            if len(core) >= 2 and core not in terms:
                terms.append(core)
    # 去掉过于泛化的连接词
    stop = {"最新", "活动", "什么", "怎么", "哪些", "今天", "最近", "现在", "目前", "帮我",
            "有什么", "是什么", "哪些活动", "什么活动"}
    filtered = [t for t in terms if t not in stop]

    # 去掉"冗余超串"：如果某个词包含另一个词，只保留更短的那个。
    # 否则整句词（永远不会出现在标题里）会稀释相关度——
    # 实测："蔚蓝档案最近有什么活动？" 保留整句时相关度被拉到 0.33（低于阈值），
    # 去掉后只剩 "蔚蓝档案" -> 1.0。
    minimal: List[str] = []
    # 排序要不依赖 set 的迭代顺序，否则同样输入可能给出不同顺序，
    # 日志/测试都难比对（实测就因此出现过断言不稳定）。
    for term in sorted(set(filtered), key=lambda t: (len(t), t)):
        if any(term != other and other in term for other in minimal):
            continue
        minimal.append(term)
    return minimal or sorted(set(terms), key=lambda t: (len(t), t))



def _too_off_topic(title: str, query: str) -> bool:
    """结果标题与查询毫无重叠时，判为跑偏。

    实测：360 对"现在游戏里正在开的活动是什么"这类缺少主语的模糊查询，
    会返回《王者荣耀》、图片站、短视频聚合页——如果不过滤，
    "信息太少"就会变成"信息是错的"，比太少更糟。
    """
    # 只在查询本身有话题词时才做跑偏判定。
    # 反例（实测）：查询"现在有什么活动？"没有话题词，结果标题里自然也不会有，
    # 于是 6 条正确结果被全部过滤掉，引擎再把它报成"检索失败"。
    if not has_topic(query):
        return False
    terms = _query_terms(query)
    if not terms:
        return False
    return not any(term.lower() in (title or "").lower() for term in terms)


def _split_360_blocks(body: str) -> List[str]:
    return re.split(r'<li class="res-list', body)[1:]


def _360_title(block: str) -> str:
    match = re.search(r'<h3[^>]*class="[^"]*res-title[^"]*"[^>]*>(.*?)</h3>', block, re.S)
    if not match:
        match = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.S)
    if not match:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()


def _360_url(block: str) -> str:
    # 首选 data-mdurl：360 直接给出了目标地址
    match = re.search(r'data-mdurl="([^"]+)"', block)
    if match:
        return html.unescape(match.group(1)).strip()
    # 退路：从 h3 里的链接取，so.com/link 形式无法还原时放弃该条
    match = re.search(r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"', block, re.S)
    if not match:
        return ""
    href = html.unescape(match.group(1))
    if "so.com/link" in href:
        return ""
    return href


def _360_snippet(block: str) -> str:
    match = re.search(r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>', block, re.S)
    if not match:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
    return re.sub(r"\s+", " ", text).strip()[:MAX_SNIPPET_CHARS]


class BingProvider(SearchProvider):
    """默认检索源：Bing 的中国站 HTML。

    为什么不是 DuckDuckGo：实测 html.duckduckgo.com 现在对脚本请求
    返回 HTTP 202 的反爬页面（正文 14KB 但零结果块），已不可用。
    cn.bing.com 能正确处理 UTF-8 中文查询并返回结构化结果；
    www.bing.com 会把查询按 gb2312 误解码，所以必须用 cn 站点。
    """

    name = "bing"

    def __init__(self, host: str = "cn.bing.com", timeout: int = 15):
        self.host = host or "cn.bing.com"
        self.timeout = timeout

    def _endpoint(self, query: str) -> str:
        encoded = urllib.parse.quote(query, encoding="utf-8")
        return f"https://{self.host}/search?q={encoded}&setlang=zh-CN"

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        domain = self.host
        remaining = _DOMAIN_GUARD.blocked_for(domain)
        if remaining > 0:
            return SearchOutcome(False, query, self.name,
                                 error=f"处于封禁冷却中（还需 {int(remaining)} 秒）",
                                 elapsed=round(time.time() - started, 2))
        _DOMAIN_GUARD.acquire(domain)
        try:
            status, body, _ = http_get(self._endpoint(query), timeout=self.timeout)
        except Exception as exc:
            return SearchOutcome(False, query, self.name, error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))
        if looks_blocked(body):
            _DOMAIN_GUARD.note_block(domain)
            return SearchOutcome(False, query, self.name,
                                 error=f"被 {self.host} 判定为异常访问，已进入冷却",
                                 elapsed=round(time.time() - started, 2))
        if status != 200:
            return SearchOutcome(False, query, self.name, error=f"HTTP {status}",
                                 elapsed=round(time.time() - started, 2))

        results: List[SearchResult] = []
        for block in _split_bing_blocks(body):
            title = _bing_title(block)
            target = _bing_url(block)
            if not title or not target or not is_safe_url(target):
                continue
            results.append(SearchResult(
                title=title, url=target, snippet=_bing_snippet(block),
                source=domain_of(target), trusted=is_trusted(target)))
            if len(results) >= limit:
                break
        if not results:
            # 结构变了或触发反爬：如实报告，绝不返回空结果冒充"没找到"
            return SearchOutcome(False, query, self.name,
                                 error=f"未能从 {self.host} 解析出结果（页面结构可能已变化）",
                                 elapsed=round(time.time() - started, 2))
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results=results,
                             elapsed=round(time.time() - started, 2))


def _split_bing_blocks(body: str) -> List[str]:
    return re.split(r'<li class="b_algo"', body)[1:]


def _bing_url(block: str) -> str:
    """从结果块里取出真实目标地址。"""
    match = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"', block)
    href = ""
    if match:
        href = html.unescape(match.group(1))
    else:
        cite = re.search(r"<cite[^>]*>(.*?)</cite>", block, re.S)
        if cite:
            text = re.sub(r"<[^>]+>", "", cite.group(1))
            href = html.unescape(text).replace(" ", "").strip()
    if "bing.com/ck/a" in href:
        # Bing 的跳转链接把目标放在 u=a1<base64url> 里
        parsed = urllib.parse.urlparse(href)
        token = urllib.parse.parse_qs(parsed.query).get("u", [""])[0]
        if token.startswith("a1"):
            padded = token[2:] + "=" * (-len(token[2:]) % 4)
            try:
                return base64.urlsafe_b64decode(padded).decode("utf-8", "ignore")
            except Exception:
                return ""
        return ""
    return href


def _bing_title(block: str) -> str:
    match = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
    if not match:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()


def _bing_snippet(block: str) -> str:
    match = re.search(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', block, re.S)
    if not match:
        match = re.search(r'<div class="b_caption"[^>]*>(.*?)</div>', block, re.S)
    if not match:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
    return re.sub(r"\s+", " ", text).strip()[:MAX_SNIPPET_CHARS]


class SearXNGProvider(SearchProvider):
    """自建/公共 SearXNG 的 JSON 接口，比抓 HTML 稳定，推荐有条件时使用。"""

    name = "searxng"

    def __init__(self, base_url: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        url = f"{self.base_url}/search?q={urllib.parse.quote(query)}&format=json"
        try:
            status, body, _ = http_get(url, timeout=self.timeout)
            payload = json.loads(body)
        except Exception as exc:
            return SearchOutcome(False, query, self.name, error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))
        results: List[SearchResult] = []
        for item in (payload.get("results") or [])[: limit * 2]:
            target = str(item.get("url", "") or "")
            title = str(item.get("title", "") or "").strip()
            if not target or not title or not is_safe_url(target):
                continue
            results.append(SearchResult(
                title=title, url=target,
                snippet=str(item.get("content", "") or "")[:MAX_SNIPPET_CHARS],
                source=domain_of(target), trusted=is_trusted(target)))
            if len(results) >= limit:
                break
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results=results,
                             elapsed=round(time.time() - started, 2))


def _nav_noise_ratio(text: str) -> float:
    """粗略估计文本里"导航味"的比例：短行（菜单项）占比越高越像导航。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 1.0
    short = sum(1 for l in lines if len(l) <= 8)
    return short / len(lines)


def _json_hints(body: str) -> str:
    """从页面内嵌 JSON 里抽标题+日期。

    TapTap / B 站动态这类前端渲染页面，真正的内容（帖子标题、发布时间）
    常常只存在于 __NEXT_DATA__ 之类的 JSON 里，纯文本抽取会漏掉，
    结果只能拿到导航菜单——这是"检索到的信息又少又老"的主要原因之一。
    """
    try:
        entries = []
        for match in JSON_DATE_RE.finditer(body):
            date = match.group(1)
            window = body[max(0, match.start() - 400): match.start()]
            titles = JSON_TITLE_RE.findall(window)
            if titles:
                title = html.unescape(titles[-1]).strip()
                if 6 <= len(title) <= 80:
                    entries.append(f"[{date}] {title}")
        # 去重、保留最近（字符串排序对 YYYY-MM-DD 足够）
        unique = sorted(set(entries), reverse=True)
        if not unique:
            return ""
        return "页面内嵌内容（含时间）：\n" + "\n".join(unique[:12])
    except Exception:
        return ""


# 各类站点的时间写法：2026-10-07 / 2026年10月7日 / 2026/10/07 / 07/27（当年）
_ANY_DATE_RE = re.compile(
    r"(20\d{2}\s?[-年/\.]\s?\d{1,2}(?:\s?[-月/\.]\s?\d{1,2}日?)?)"
    r"|(\b\d{1,2}/\d{1,2}\b)")


def staleness_days(text: str, now: Optional[float] = None) -> int:
    """内容里最新日期距今天数；无法判断时返回 -1。"""
    latest = _latest_date_in(text)
    if not latest:
        return -1
    try:
        year, month, day = (int(part) for part in latest.split("-"))
        stamp = time.mktime((year, month, day, 0, 0, 0, 0, 0, -1))
    except (ValueError, OverflowError):
        return -1
    return max(0, int(((time.time() if now is None else now) - stamp) / 86400))


def freshest_content_age(outcome: SearchOutcome) -> int:
    """这批结果里"最新内容"距今多少天；全都判断不出时返回 -1。

    用途：把"信息有多旧"明确告诉模型。实测中模型能看到正文里的日期，
    但只有提示词点名要求它比较时，它才会主动提醒用户"这是几个月前的信息"。
    """
    ages = [staleness_days(r.content or "") for r in outcome.results]
    ages = [a for a in ages if a >= 0]
    return min(ages) if ages else -1


def _latest_date_in(text: str) -> str:
    """找出文本里出现的最新日期，用于提示模型"这批信息有多新"。"""
    if not text:
        return ""
    found = []
    for match in re.finditer(r"(20\d{2})\s?[-年/\.]\s?(\d{1,2})(?:\s?[-月/\.]\s?(\d{1,2}))?", text):
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        day = int(match.group(3) or 1)
        if 1 <= month <= 12 and 1 <= day <= 31:
            found.append((year, month, day))
    if not found:
        return ""
    year, month, day = max(found)
    return f"{year}-{month:02d}-{day:02d}"


def _date_anchored(text: str, max_lines: int = 25) -> str:
    """挑出"带日期的行"。

    社区/论坛类页面上，真正有用的信息是"某天发布了某条公告"，
    而这类行的特征就是带日期。导航菜单几乎不带日期，
    所以日期锚定能有效把信噪比拉起来。
    """
    if not text:
        return ""
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 220:
            continue
        if _ANY_DATE_RE.search(line):
            lines.append(line)
        if len(lines) >= max_lines:
            break
    if not lines:
        return ""
    # 去重但保持原顺序
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "含时间的片段：\n" + "\n".join(unique)


def _markup_time_hints(body: str, max_lines: int = 12) -> str:
    """从 HTML 属性/时间标签里抽"带时间的条目"。

    很多列表页把发布时间放在 title= 或 <time> 里，纯文本抽取会丢掉。
    """
    try:
        found: List[str] = []
        for match in re.finditer(r"<time[^>]*>([^<]{4,40})</time>", body, re.I):
            found.append(match.group(1).strip())
        for match in re.finditer(r'title="([^"]{6,120})"', body):
            value = html.unescape(match.group(1)).strip()
            if _ANY_DATE_RE.search(value) or re.search(r"(公告|活动|更新|维护)", value):
                found.append(value)
        unique: List[str] = []
        seen = set()
        for item in found:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
            if len(unique) >= max_lines:
                break
        if not unique:
            return ""
        return "标记中的标题/时间：\n" + "\n".join(unique)
    except Exception:
        return ""


def fetch_page(url: str, timeout: int = 15, max_chars: int = MAX_PAGE_CHARS) -> str:
    """抓取网页正文（用于给模型提供足够上下文）。

    做法：先抽可见文本，再补上内嵌 JSON 里的标题+日期，
    并且**丢掉开头的导航带**——实测导航会占掉前 3000 字，
    把真正的内容挤出上下文窗口。
    """
    if is_low_value(url):
        return ""
    try:
        status, body, _ = http_get(url, timeout=timeout)
    except Exception:
        return ""
    if status != 200:
        return ""

    hints = _json_hints(body)
    markup = _markup_time_hints(body)

    parser = _TextExtractor()
    try:
        parser.feed(body)
    except Exception:
        return hints[:max_chars]
    text = parser.text()

    # 逐段评估：短行密集的段落多半是导航/页脚，丢掉
    paragraphs = [p.strip() for p in re.split(r"\n", text) if p.strip()]
    kept: List[str] = []
    for index, para in enumerate(paragraphs):
        # 开头 12 段且明显是短行菜单 -> 判为导航
        if index < 12 and len(para) <= 12 and _nav_noise_ratio(para) > 0.5:
            continue
        kept.append(para)
    cleaned = "\n".join(kept)

    anchored = _date_anchored(text)
    # 优先级：内嵌 JSON 的时间线索 > 标记里的时间行 > 日期锚定行 > 正文
    prefix = "\n\n".join(p for p in (hints, markup, anchored) if p.strip())
    budget = max_chars - len(prefix)
    if budget < 500:
        budget = 500
    if len(cleaned) > budget:
        cleaned = cleaned[:budget].rstrip() + "…"

    parts = [p for p in (prefix, cleaned) if p.strip()]
    return "\n\n".join(parts)


class _SessdataInjector(urllib.request.BaseHandler):
    """无条件给 B 站域名的请求加上 `Cookie: SESSDATA=...`。

    为什么只用 CookieJar 不够：CookieJar 只对"经过它的请求"生效，而本项目里
    请求是 `_fetch_json(opener, url)` 这样把 opener 传进去的——测试传假 opener，
    生产也可能换 opener。只要有一处不经过 Jar，SESSDATA 就静默失效，
    而**失效的表现和匿名完全一样**（nav 返回未登录、字幕拿不到），极难发现。
    这里改成在请求层强制注入：只要域名是 B 站就一定带上。

    安全边界：只对 `bilibili.com` / `*.bilibili.com` 注入，别的域名一个字节都不发。
    """

    def __init__(self, sessdata: str):
        self.sessdata = sessdata or ""

    @staticmethod
    def _is_bilibili(url: str) -> bool:
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError:
            return False
        return host == "bilibili.com" or host.endswith(".bilibili.com")

    def http_request(self, request):
        if self.sessdata and self._is_bilibili(request.full_url):
            # 先清掉可能存在的旧 Cookie 头：否则"已带 Cookie"的请求会带着
            # 另一个 Cookie 出去，而 SESSDATA 反而没生效（更难查）。
            for store in (request.headers, request.unredirected_hdrs):
                for key in [k for k in store if k.lower() == "cookie"]:
                    store.pop(key, None)
            request.add_unredirected_header("Cookie", "SESSDATA=" + self.sessdata)
        return request

    https_request = http_request


class BilibiliProvider(SearchProvider):
    """B 站搜索（视频 + 专栏）。

    实测结论：
    - 直接请求 api.bilibili.com/.../search/type 会返回 **HTTP 412 "request was banned"**；
    - **先访问一次 bilibili.com 拿到 buvid3 cookie 后，同一接口返回 200**；
    - search/all/v2 即使不带 cookie 也能用（作为兜底）。
    所以这里复用了与百度相同的"cookie 会话"手法——区别是这次真的有效。

    相对抓 HTML 的优势：
    - 字段结构化：bvid / title / author / description / pubdate；
    - pubdate 是真实时间戳，时效性判断比从正文里猜日期可靠得多；
    - B 站是中文游戏/动漫信息的主要聚集地（官方动态、活动攻略都在这里）。
    """

    name = "bilibili"
    HOME = "https://www.bilibili.com/"
    TYPE_SEARCH = "https://api.bilibili.com/x/web-interface/search/type"
    ALL_SEARCH = "https://api.bilibili.com/x/web-interface/search/all/v2"

    DETAIL = "https://api.bilibili.com/x/web-interface/view"
    PLAYER = "https://api.bilibili.com/x/player/v2"
    REPLIES = "https://api.bilibili.com/x/v2/reply"
    PLAYURL = "https://api.bilibili.com/x/player/playurl"

    def __init__(self, timeout: int = 15, include_article: bool = True,
                 sessdata: str = "", enrich_top: int = 2, comments_limit: int = 8,
                 subtitle_chars: int = 1200, comments_chars: int = 600,
                 enrich_delay: float = 0.35, asr_client=None, asr_guard=None,
                 asr_max_bytes: int = 8 * 1024 * 1024, asr_max_seconds: int = 300,
                 asr_segment_seconds: int = 300, asr_max_segments: int = 24,
                 asr_ffmpeg: str = "", asr_local: bool = False):
        self.timeout = timeout
        self.include_article = include_article
        # SESSDATA：用户自己的 B 站登录 Cookie。
        # 匿名实测：评论只返回 3 条、字幕完全不返回；
        # 带上它才能翻页拿更多评论、并读取 AI/CC 字幕。
        self.sessdata = sessdata or ""
        self.enrich_top = max(0, enrich_top)
        self.comments_limit = max(0, comments_limit)
        self.subtitle_chars = max(0, subtitle_chars)
        self.comments_chars = max(0, comments_chars)
        self.enrich_delay = enrich_delay
        # 语音识别兜底：B 站没给字幕时，下载音频自己做识别。
        # asr_guard 是可调用对象，返回 False 表示"本次不许用"（额度/开关由上层决定）——
        # 这样 provider 不需要接触存储层。
        self.asr_client = asr_client
        self.asr_guard = asr_guard
        self.asr_max_bytes = max(0, asr_max_bytes)
        self.asr_max_seconds = max(0, asr_max_seconds)
        # 长视频分段：本地识别不花钱，所以超时长的视频不再直接放弃，
        # 而是用 ffmpeg 切段**顺序**识别（切段同时顺带解决了体积上限）。
        self.asr_segment_seconds = max(0, asr_segment_seconds)
        self.asr_max_segments = max(0, asr_max_segments)
        self.asr_ffmpeg = asr_ffmpeg or ""
        self.asr_local = bool(asr_local)
        self._opener = None
        self._warmed = False

    @property
    def asr_ready(self) -> bool:
        return bool(self.asr_client is not None and getattr(self.asr_client, "available", False))

    @property
    def logged_in(self) -> bool:
        return bool(self.sessdata)

    def _get_opener(self):
        if self._opener is None:
            # 为什么不能只靠 CookieJar：CookieJar 只在**请求真的经过它**时才生效，
            # 而 _fetch_json / _download_audio 这些方法都接收外部传入的 opener
            # （测试会传假 opener，也方便复用连接）。实测过后果：
            # 配了 SESSDATA，nav 依然返回"账号未登录"，字幕/高码率全拿不到——
            # 而且**表现和没配一模一样**，所以长期没被发现（之前填的是假值，
            # 更看不出来）。这里用一个无条件注入 Cookie 头的 processor，
            # 不管 opener 是谁，只要走 B 站域名就一定带上 SESSDATA。
            processors = []
            if self.sessdata:
                processors.append(_SessdataInjector(self.sessdata))
            processors.append(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            self._opener = urllib.request.build_opener(*processors)
        return self._opener

    def _headers(self, referer: str = HOME) -> Dict[str, str]:
        return {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Referer": referer,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    NAV = "https://api.bilibili.com/x/web-interface/nav"

    def verify_login(self) -> Dict[str, Any]:
        """校验 SESSDATA 是否有效。

        配置了 Cookie 却无效（过期/复制错）时，表现和匿名一样——
        用户会以为"字幕功能没用"。所以提供一个显式检查，供启动时打印。
        """
        if not self.sessdata:
            return {"configured": False, "valid": False, "user": "",
                    "note": "未配置 BILIBILI_SESSDATA：评论最多 3 条，且读不到字幕"}
        try:
            opener = self._get_opener()
            payload = self._fetch_json(opener, self.NAV)
        except Exception as exc:
            return {"configured": True, "valid": False, "user": "",
                    "note": f"校验失败：{type(exc).__name__}: {exc}"}
        data = payload.get("data") or {}
        valid = bool(data.get("isLogin"))
        return {"configured": True, "valid": valid,
                "user": str(data.get("uname") or ""),
                "note": "登录态有效" if valid else "SESSDATA 无效或已过期，行为等同匿名"}

    # -- 视频内信息与评论 -------------------------------------------------
    def _fetch_detail(self, opener, bvid: str) -> Dict:
        try:
            payload = self._fetch_json(opener, f"{self.DETAIL}?bvid={bvid}")
        except Exception:
            return {}
        if payload.get("code") != 0:
            return {}
        data = payload.get("data") or {}
        stat = data.get("stat") or {}
        return {
            "aid": data.get("aid"),
            "cid": data.get("cid"),
            "desc": re.sub(r"\s+", " ", str(data.get("desc") or "")).strip(),
            "duration": data.get("duration"),
            "view": stat.get("view"),
            "pages": len(data.get("pages") or []),
        }

    # WBI 签名用的重排表（B 站前端公开算法）
    _MIXIN_KEY_ENC_TAB = (
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
        33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
        61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
        36, 20, 34, 44, 52,
    )

    def _mixin_key(self, opener) -> str:
        """取 WBI mixin key（进程内缓存）。"""
        if getattr(self, "_mixin_cache", None):
            return self._mixin_cache
        try:
            payload = self._fetch_json(opener, "https://api.bilibili.com/x/web-interface/nav")
            wbi = (payload.get("data") or {}).get("wbi_img") or {}
            img_key = str(wbi.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
            sub_key = str(wbi.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
            raw = img_key + sub_key
            if len(raw) < 64:
                return ""
            mixin = "".join(raw[i] for i in self._MIXIN_KEY_ENC_TAB)[:32]
            self._mixin_cache = mixin
            return mixin
        except Exception:
            return ""

    def _wbi_signed_query(self, params: Dict[str, Any], mixin: str) -> str:
        signed = {k: str(v) for k, v in params.items()}
        signed["wts"] = str(int(time.time()))
        query = urllib.parse.urlencode(sorted(signed.items()))
        signed["w_rid"] = hashlib.md5((query + mixin).encode()).hexdigest()
        return urllib.parse.urlencode(sorted(signed.items()))

    def _session_state(self, opener) -> str:
        """判断配的 SESSDATA 到底有没有生效：'valid' / 'invalid' / 'anonymous'。

        为什么要查这个：`subtitle.subtitles` 为空时无法区分三种情况——
        视频真的没有字幕、匿名访问被限制、或者**配的 Cookie 是坏的/过期的**。
        实测过最难受的一种：`.env` 里填的是个假值（34 字符的占位串），
        代码看到"非空"就按"已登录"处理并提示"配置 SESSDATA 后多数视频可读"，
        而实际上它和匿名完全没区别——用户按提示配置了，却怎么都不生效。

        `nav` 接口返回 `data.isLogin`，是唯一能区分"登录了"和"没登录"的地方。
        结果缓存在实例上：一次会话只查一次，不给 B 站添请求。
        """
        if not self.sessdata:
            return "anonymous"
        cached = getattr(self, "_session_state_cache", None)
        if cached:
            return cached
        state = "invalid"
        try:
            payload = self._fetch_json(opener, self.NAV)
            data = payload.get("data") or {}
            if payload.get("code") == 0 and data.get("isLogin"):
                state = "valid"
        except Exception:
            # 网络异常时不能断言 Cookie 坏了，按"未验证"处理成 invalid 会误导，
            # 所以这里保留 anonymous 的措辞（不确定就少说）。
            state = "unknown"
        self._session_state_cache = state
        return state

    def _no_subtitle_note(self, opener) -> str:
        """字幕取不到时，给出**符合实情**的说明。"""
        state = self._session_state(opener)
        if state == "valid":
            return "（无字幕数据：该视频本身没有提供 AI/CC 字幕）"
        if state == "invalid":
            return ("（无字幕数据：另外检测到 .env 里的 BILIBILI_SESSDATA **没有生效**"
                    "（nav 接口返回未登录），所以连「登录才可见」的字幕也拿不到。"
                    "请换成浏览器里当前有效的 SESSDATA（60~80 字符）后重试）")
        if state == "unknown":
            return ("（无字幕数据：该视频没提供字幕，或访问受限；"
                    "BILIBILI_SESSDATA 的状态这次没能验证）")
        return ("（无字幕数据：该视频没提供字幕，或匿名访问拿不到。"
                "配置 BILIBILI_SESSDATA 后多数视频可读到 AI/CC 字幕）")

    def _fetch_subtitle(self, opener, bvid: str, cid) -> Dict[str, str]:
        """取视频字幕文本。

        匿名状态下 B 站**不返回字幕轨道**（实测 0/5 个视频），
        所以无字幕时给出可执行的说明，而不是假装"这个视频没内容"。
        """
        if not cid:
            return {"text": "", "note": ""}
        try:
            payload = self._fetch_json(opener, f"{self.PLAYER}?bvid={bvid}&cid={cid}")
        except Exception:
            return {"text": "", "note": ""}
        if payload.get("code") != 0:
            return {"text": "", "note": ""}
        tracks = ((payload.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
        if not tracks:
            return {"text": "", "note": self._no_subtitle_note(opener)}
        track = next((t for t in tracks if str(t.get("lan", "")).startswith("zh")), tracks[0])
        url = str(track.get("subtitle_url") or "")

        # 实测：普通 player 接口有时对 ai-zh 轨道返回**空地址**，
        # 而 WBI 签名的同名接口会给出完整地址（含 auth_key）。所以补一次兜底。
        if not url:
            mixin = self._mixin_key(opener)
            if mixin:
                try:
                    query = self._wbi_signed_query({"bvid": bvid, "cid": cid}, mixin)
                    fallback = self._fetch_json(
                        opener, f"https://api.bilibili.com/x/player/wbi/v2?{query}")
                    tracks2 = ((fallback.get("data") or {}).get("subtitle")
                               or {}).get("subtitles") or []
                    track2 = next((t for t in tracks2
                                   if str(t.get("lan", "")).startswith("zh")), None)
                    if track2 and track2.get("subtitle_url"):
                        track, url = track2, str(track2["subtitle_url"])
                except Exception:
                    pass
        if url.startswith("//"):
            url = "https:" + url
        if not url:
            return {"text": "", "note": "（字幕轨道存在但未给出地址，可能需要登录）"}
        try:
            status, body, _ = http_get(url, timeout=self.timeout)
            items = json.loads(body).get("body") or []
        except Exception as exc:
            return {"text": "", "note": f"（字幕读取失败：{type(exc).__name__}）"}
        text = " ".join(str(i.get("content") or "") for i in items).strip()
        if not text:
            return {"text": "", "note": ""}
        return {"text": text[:self.subtitle_chars],
                "note": f"（字幕语言：{track.get('lan_doc') or track.get('lan') or '未知'}）"}

    def _fetch_audio_url(self, opener, bvid: str, cid) -> Dict[str, Any]:
        """取音频流地址。

        实测：playurl 接口匿名可用，返回 3 条 AAC 轨道（64k/103k/183k）。
        选**最低码率**那条——语音识别不需要高音质，小体积上传更快也更省成本。
        """
        if not cid:
            return {}
        try:
            payload = self._fetch_json(
                opener, f"{self.PLAYURL}?bvid={bvid}&cid={cid}&fnval=16&fnver=0&fourk=1")
        except Exception:
            return {}
        if payload.get("code") != 0:
            return {}
        audio = ((payload.get("data") or {}).get("dash") or {}).get("audio") or []
        if not audio:
            return {}
        track = min(audio, key=lambda a: a.get("bandwidth") or 1 << 30)
        return {"url": str(track.get("baseUrl") or ""), "bandwidth": track.get("bandwidth")}

    def _download_audio(self, opener, url: str, max_bytes: int) -> Dict[str, Any]:
        """下载音频（限制体积）。

        先看 Content-Length 再决定：
        **超过上限就直接放弃**，而不是截断上传——
        截断的 m4a 是损坏文件，识别服务多半会失败或给出错误结果。
        例外：本地识别走分段路径时，由调用方把上限放大，
        因为本地识别不收钱、切段由 ffmpeg 负责（见 _transcribe）。
        """
        if not url or max_bytes <= 0:
            return {"ok": False, "reason": "未取到音频地址"}
        try:
            head = urllib.request.Request(url, method="HEAD", headers=self._headers())
            with opener.open(head, timeout=self.timeout) as response:
                length = int(response.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        if length and length > max_bytes:
            return {"ok": False, "size": length,
                    "reason": f"音频约 {length // 1024 // 1024}MB，"
                              f"超出识别体积上限 {max_bytes // 1024 // 1024}MB，已跳过"}
        # 多读 1 字节用于判断"是否真的超限"
        request = urllib.request.Request(
            url, headers={**self._headers(), "Range": f"bytes=0-{max_bytes}"})
        try:
            with opener.open(request, timeout=self.timeout) as response:
                data = response.read(max_bytes + 1)
        except Exception as exc:
            return {"ok": False, "reason": f"音频下载失败：{type(exc).__name__}"}
        if not data:
            return {"ok": False, "reason": "音频为空"}
        # 实测 B 站有时会忽略 Range 头、直接返回整个文件（3MB 上限却收到 10MB）。
        # 这时**不能截断使用**——截断的 m4a 是损坏文件，识别会失败或给出错结果，
        # 所以如实放弃并说明原因。
        if len(data) > max_bytes:
            return {"ok": False, "size": len(data),
                    "reason": f"音频实际大小 {len(data) // 1024 // 1024}MB 超过上限 "
                              f"{max_bytes // 1024 // 1024}MB（服务端未接受分段请求），已跳过"}
        return {"ok": True, "data": data, "size": len(data)}

    def _transcribe_plan(self, video_seconds: float, max_bytes: int) -> Dict[str, Any]:
        """决定"这条视频怎么识别"。

        返回 {mode, note, ...}：
          skip       —— 不识别（有 note 说明原因）
          direct     —— 整段下载直接识别（短视频，最省事）
          segmented  —— 下载后用 ffmpeg 切段顺序识别（长视频）
          truncate   —— 只识别前 max_seconds 秒（有 ffmpeg 但禁用分段时的兜底）

        为什么长视频要分段：本地识别不花钱，唯一的问题是**体积和耗时**。
        切段同时解决两个问题（每段都能独立穿过体积上限），
        而且比"只识别前 5 分钟"诚实得多——后者会把 20 分钟视频的其他内容当成不存在。
        """
        limit = int(self.asr_max_seconds or 0)
        segment = int(self.asr_segment_seconds or 0)
        seconds = max(0.0, float(video_seconds or 0.0))
        ffmpeg = find_ffmpeg(explicit=self.asr_ffmpeg)
        if not limit or seconds <= limit:
            return {"mode": "direct", "note": ""}
        # 超时长：只有在本地识别（不计费）时才值得花时间分段
        if self.asr_local and segment > 0 and ffmpeg:
            needed = int(math.ceil(seconds / segment))
            segments = min(self.asr_max_segments or 0, needed)
            if segments <= 0:
                return {"mode": "truncate", "limit": limit,
                        "note": ("（分段段数上限为 0，改为只识别开头 "
                                 f"{round(limit / 60)} 分钟）")}
            covered = min(seconds, segments * segment)
            note = (f"（视频约 {round(seconds / 60)} 分钟，已用 ffmpeg 切成 "
                    f"{round(segment / 60)} 分钟一段顺序识别）")
            # 只在**真的没覆盖完**时才说"还有多少没识别"。
            # 早期版本这里无条件写"识别的是前 N 分钟"，于是 222 秒的视频
            # 会写成"前 4 分钟"（240 秒 > 222 秒，自相矛盾）。
            if covered < seconds - 1:
                note = note[:-1] + (f"；本次只识别前 {covered / 60:.1f} 分钟，"
                                    f"剩余 {(seconds - covered) / 60:.1f} 分钟未识别）")
            return {"mode": "segmented", "note": note, "segments": segments,
                    "segment": segment, "ffmpeg": ffmpeg}
        # 云端识别或没有 ffmpeg：不能悄悄送一段过去，只能如实说明
        if not ffmpeg:
            reason = "没能找到 ffmpeg，无法裁剪音频"
        else:
            reason = "当前是云端识别（按时长计费），不做自动分段"
        return {"mode": "skip", "limit": limit,
                "note": (f"（视频时长约 {round(seconds / 60)} 分钟，超过语音识别时长上限 "
                         f"{round(limit / 60)} 分钟，已跳过；{reason}）")}

    def _transcribe(self, opener, bvid: str, cid, max_bytes: int,
                    video_seconds: Any = None) -> Dict[str, str]:
        """没有字幕时，自己下载音频做语音识别。

        按 `_transcribe_plan` 的三条路径走：
          direct    —— 直接识别整段音频；
          segmented —— ffmpeg 切段顺序识别（本地识别，长视频）；
          skip      —— 如实说明为什么不识别。

        分段是为长视频加的（原先超时长一律放弃）。**宁可覆盖一部分并写清覆盖范围，
        也不假装整条视频都识别过了。**
        """
        if not self.asr_ready:
            return {"text": "", "note": ""}
        try:
            seconds = float(video_seconds) if video_seconds else 0.0
        except (TypeError, ValueError):
            seconds = 0.0
        plan = self._transcribe_plan(seconds, max_bytes)
        if plan["mode"] == "skip":
            return {"text": "", "note": plan["note"]}

        if self.asr_guard is not None and not self.asr_guard():
            return {"text": "", "note": "（已启用语音识别，但本次额度已用尽，跳过）"}
        audio = self._fetch_audio_url(opener, bvid, cid)
        if not audio.get("url"):
            return {"text": "", "note": "（未取到音频流地址，无法做语音识别）"}
        time.sleep(self.enrich_delay)
        # 分段模式下把体积上限放大：切段后每段很小，
        # 但原文件必须先整个下载下来才能切。
        want = max_bytes
        if plan["mode"] == "segmented":
            # 每段 5 分钟 × 最低码率 64kbps ≈ 2.4MB，留 4 倍余量
            per_segment = int(plan["segment"] * 64 * 1000 / 8 * 4)
            want = max(max_bytes, per_segment)
        downloaded = self._download_audio(opener, audio["url"], want)
        if not downloaded.get("ok"):
            reason = downloaded.get("reason", "音频获取失败")
            # 分段模式下体积超限不算致命：真超了就把上限收敛回原值再试一次
            if plan["mode"] == "segmented" and "超过上限" in reason:
                downloaded = self._download_audio(opener, audio["url"], max_bytes)
                if downloaded.get("ok"):
                    plan = dict(plan, mode="direct",
                                note="（音频体积受限，只识别整段前部分音频）")
                else:
                    return {"text": "", "note": f"（{downloaded.get('reason', reason)}）"}
            else:
                return {"text": "", "note": f"（{reason}）"}
        time.sleep(self.enrich_delay)

        if plan["mode"] == "direct":
            try:
                result = self.asr_client.transcribe(downloaded["data"], filename="audio.m4a")
            except Exception as exc:
                return {"text": "", "note": f"（语音识别异常：{type(exc).__name__}）"}
            if not result.ok:
                return {"text": "", "note": f"（语音识别失败：{result.error}）"}
            note = "（由音频识别生成，可能存在误差）"
            warning = quality_warning(result.text, seconds=seconds,
                                      avg_logprob=getattr(result, "avg_logprob", None),
                                      no_speech_prob=getattr(result, "no_speech_prob", None))
            return {"text": result.text, "note": note + warning}

        # ---------- 分段识别 ----------
        ffmpeg = plan["ffmpeg"]
        segment = float(plan["segment"])
        pieces: List[str] = []
        warnings: List[str] = []
        failed = ""
        done = 0
        covered = 0.0
        for index in range(int(plan["segments"])):
            start = index * segment
            if seconds and start >= seconds:
                break
            span = segment if not seconds else min(segment, max(0.0, seconds - start))
            if span < 1.0:
                break
            chunk, err = ffmpeg_extract(downloaded["data"], start, span, ffmpeg=ffmpeg)
            if chunk is None:
                failed = err or "未知原因"
                if not pieces:
                    return {"text": "", "note": (
                        f"（ffmpeg 分段失败，无法识别这条视频：{failed}）")}
                break
            # 记下这一段的**真实**时长：
            # 计划里的段长只是请求值，实际可能因为到文件尾而更短（见 ffmpeg.extract_segment），
            # 直接用"段数 × 段长"报覆盖范围会虚报。
            real_span = span
            measured, _ = ffmpeg_probe(chunk, ffmpeg=ffmpeg)
            if measured > 0:
                real_span = measured
            try:
                result = self.asr_client.transcribe(chunk, filename="chunk.mkv")
            except Exception as exc:
                failed = f"{type(exc).__name__}"
                break
            if not result.ok:
                failed = result.error or "识别失败"
                break
            done += 1
            covered += real_span
            if result.text.strip():
                pieces.append(result.text.strip())
                piece_warning = quality_warning(
                    result.text, seconds=real_span,
                    avg_logprob=getattr(result, "avg_logprob", None),
                    no_speech_prob=getattr(result, "no_speech_prob", None))
                if piece_warning and piece_warning not in warnings:
                    warnings.append(piece_warning)
        if not pieces:
            return {"text": "", "note": f"（分段识别没有拿到文本：{failed or '内容为空'}）"}
        note = (f"（由音频识别生成，可能存在误差；覆盖约 "
                f"{covered / 60:.1f} 分钟，共 {done} 段）")
        if plan.get("note"):
            note += plan["note"]
        if failed:
            note += f"（第 {done + 1} 段起中断：{failed}）"
        if warnings:
            note += "".join(warnings[:1])
        return {"text": "".join(pieces), "note": note}

    def _fetch_comments(self, opener, aid) -> List[str]:
        """取热门评论（sort=2 即按热度）。

        匿名实测**只返回 3 条**，但那是最高赞的几条，信息密度往往很高；
        带 SESSDATA 后可翻页取更多。
        """
        if not aid or self.comments_limit <= 0:
            return []
        comments: List[str] = []
        page = 1
        max_pages = 1 if not self.logged_in else max(1, (self.comments_limit + 19) // 20)
        while page <= max_pages and len(comments) < self.comments_limit:
            try:
                payload = self._fetch_json(
                    opener, f"{self.REPLIES}?type=1&oid={aid}&sort=2&ps=20&pn={page}")
            except Exception:
                break
            if payload.get("code") != 0:
                break
            replies = ((payload.get("data") or {}).get("replies")) or []
            if not replies:
                break
            for reply in replies:
                message = re.sub(r"\s+", " ",
                                 str((reply.get("content") or {}).get("message") or "")).strip()
                if not message:
                    continue
                comments.append(f"[赞{reply.get('like') or 0}] {message[:120]}")
                if len(comments) >= self.comments_limit:
                    break
            if len(replies) < 20:
                break
            page += 1
        return comments

    def _enrich(self, opener, result: SearchResult, *, include_subtitle_note: bool = True,
                query: str = "") -> None:
        """把"视频内的信息"（简介/字幕）与热门评论补进 result.content。

        include_subtitle_note：只在第一条结果里说明"字幕拿不到、如何开启"。
        实测每个视频都带这条说明会重复多遍，白占上下文。
        """
        bvid = result.url.rsplit("/", 1)[-1] if "/video/" in result.url else ""
        if not bvid:
            return
        detail = self._fetch_detail(opener, bvid)
        if not detail:
            return
        blocks: List[str] = []
        if detail.get("desc") and len(str(detail["desc"])) > 5:
            blocks.append("【视频简介】" + str(detail["desc"])[:300])
        try:
            if detail.get("duration"):
                blocks.append(f"【视频时长】约 {int(detail['duration']) // 60} 分钟")
        except (TypeError, ValueError):
            pass
        if detail.get("view"):
            blocks.append(f"【播放量】{detail['view']}")
        if detail.get("pages") and detail["pages"] > 1:
            blocks.append(f"【分P数】{detail['pages']}")

        if self.subtitle_chars > 0:
            time.sleep(self.enrich_delay)
            subtitle = self._fetch_subtitle(opener, bvid, detail.get("cid"))
            if subtitle.get("text"):
                # 搬运/标题党提示：不丢弃文本（丢弃会误杀正常解说），
                # 而是明确标注"机器识别、未核对"，用词偏离时再加一句警告。
                reference = f"{result.title} {result.snippet} {query}"
                warning = ""
                if not transcript_matches_topic(subtitle["text"], reference):
                    warning = ("\n注意：该字幕用词与标题/查询主题几乎没有重合，"
                               "可能是搬运或标题与内容不符的视频，请勿当作本主题的信息使用。")
                blocks.append("【视频字幕（机器识别，未核对）】"
                              + subtitle["text"] + subtitle.get("note", "") + warning)
            else:
                # B 站没给字幕（匿名必然如此）。若启用了自有 ASR，就下载音频自己识别——
                # 这才是"没有现成字幕时仍能拿到视频里说了什么"的唯一途径。
                transcript = self._transcribe(opener, bvid, detail.get("cid"),
                                              getattr(self, "asr_max_bytes", 0),
                                              video_seconds=detail.get("duration"))
                if transcript.get("text"):
                    reference = f"{result.title} {result.snippet} {query}"
                    warning = ""
                    if not transcript_matches_topic(transcript["text"], reference):
                        warning = ("\n注意：该识别文本用词与标题/查询主题几乎没有重合，"
                                   "可能是搬运或标题与内容不符的视频，请勿当作本主题的信息使用。")
                    blocks.append("【视频语音识别（未核对）】"
                                  + transcript["text"] + transcript["note"] + warning)
                elif transcript.get("note"):
                    blocks.append(transcript["note"])
                elif subtitle.get("note") and include_subtitle_note:
                    blocks.append(subtitle["note"])

        if self.comments_chars > 0:
            time.sleep(self.enrich_delay)
            comments = self._fetch_comments(opener, detail.get("aid"))
            if comments:
                scope = "匿名状态最多 3 条" if not self.logged_in else "按赞排序"
                blocks.append(f"【热门评论（{scope}）】"
                              + " / ".join(comments)[:self.comments_chars])

        if blocks:
            result.content = "\n".join(blocks)

    def _warm_up(self, opener) -> None:
        """访问首页取得 buvid3 等 cookie；没有它搜索接口会被判 412。"""
        if self._warmed:
            return
        try:
            request = urllib.request.Request(self.HOME, headers=self._headers())
            with opener.open(request, timeout=self.timeout) as response:
                response.read(120_000)
        except Exception:
            pass
        self._warmed = True

    def _fetch_json(self, opener, url: str) -> Dict:
        request = urllib.request.Request(url, headers=self._headers())
        with opener.open(request, timeout=self.timeout) as response:
            return json.loads(_decode_body(response.read(500_000), response.headers))

    @staticmethod
    def _clean_title(title: str) -> str:
        # 搜索结果的标题里有 <em class="keyword"> 高亮标签，要去掉
        text = re.sub(r"<[^>]+>", "", title or "")
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _format_pubdate(stamp) -> str:
        """把 pubdate 时间戳转成 YYYY-MM-DD。

        注意：B 站对"未知发布时间"会返回 0，直接格式化会得到 1970-01-01——
        那会被时效判定当成"极旧信息"，进而误报"内容很可能已过时"。
        所以这里把 0/None/非法值都视为"未知"，返回空串。
        """
        try:
            value = int(stamp)
        except (TypeError, ValueError):
            return ""
        if value <= 0:
            return ""
        try:
            return time.strftime("%Y-%m-%d", time.localtime(value))
        except (ValueError, OSError, OverflowError):
            return ""


    def _video_results(self, items: List[Dict], limit: int) -> List[SearchResult]:
        results: List[SearchResult] = []
        for item in items:
            title = self._clean_title(str(item.get("title") or ""))
            bvid = str(item.get("bvid") or "").strip()
            url = f"https://www.bilibili.com/video/{bvid}" if bvid else str(item.get("arcurl") or "")
            if not title or not url or not is_safe_url(url):
                continue
            published = self._format_pubdate(item.get("pubdate"))
            author = str(item.get("author") or "").strip()
            desc = re.sub(r"\s+", " ", str(item.get("description") or "")).strip()
            # B 站视频简介经常是 UP 主的闲聊（实测见过"今天睡过头了…"），
            # 截短一些，把有限的上文预算留给标题与时间。
            if len(desc) > 120:
                desc = desc[:120].rstrip() + "…"
            meta = "；".join(x for x in (published or "日期未知",
                                        f"UP：{author}" if author else "") if x)
            snippet = f"{desc}（{meta}）" if desc else f"（{meta}）"
            results.append(SearchResult(title=title, url=url, snippet=snippet,
                                        source="bilibili.com · 视频", trusted=True))
            if len(results) >= limit:
                break
        return results

    def _article_results(self, items: List[Dict], limit: int) -> List[SearchResult]:
        results: List[SearchResult] = []
        for item in items:
            title = self._clean_title(str(item.get("title") or ""))
            cvid = item.get("id") or item.get("cvid")
            url = f"https://www.bilibili.com/read/cv{cvid}" if cvid else ""
            if not title or not url or not is_safe_url(url):
                continue
            published = self._format_pubdate(item.get("pubdate") or item.get("ctime"))
            author = str(item.get("author") or "").strip()
            desc = re.sub(r"\s+", " ", str(item.get("desc") or item.get("description") or "")).strip()
            if len(desc) > 160:
                desc = desc[:160].rstrip() + "…"
            meta = "；".join(x for x in (published or "日期未知",
                                        f"UP：{author}" if author else "") if x)
            snippet = f"{desc}（{meta}）" if desc else f"（{meta}）"
            results.append(SearchResult(title=title, url=url, snippet=snippet,
                                        source="bilibili.com · 专栏", trusted=True))
            if len(results) >= limit:
                break
        return results

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        remaining = _DOMAIN_GUARD.blocked_for("bilibili.com")
        if remaining > 0:
            return SearchOutcome(False, query, self.name,
                                 error=f"处于冷却中（还需 {int(remaining)} 秒）",
                                 elapsed=round(time.time() - started, 2))
        _DOMAIN_GUARD.acquire("bilibili.com")
        opener = self._get_opener()
        self._warm_up(opener)

        keyword = urllib.parse.quote(query, encoding="utf-8")
        collected: List[SearchResult] = []
        errors: List[str] = []

        # 视频
        try:
            payload = self._fetch_json(
                opener, f"{self.TYPE_SEARCH}?search_type=video&keyword={keyword}&page=1")
            if payload.get("code") == 0:
                collected += self._video_results(
                    ((payload.get("data") or {}).get("result") or []), limit)
            else:
                errors.append(f"video code={payload.get('code')}")
        except Exception as exc:
            errors.append(f"video {type(exc).__name__}: {exc}")

        # 专栏（活动公告常以专栏形式发布）
        if self.include_article and len(collected) < limit:
            time.sleep(0.6)
            try:
                payload = self._fetch_json(
                    opener, f"{self.TYPE_SEARCH}?search_type=article&keyword={keyword}&page=1")
                if payload.get("code") == 0:
                    collected += self._article_results(
                        ((payload.get("data") or {}).get("result") or []), limit)
                else:
                    errors.append(f"article code={payload.get('code')}")
            except Exception as exc:
                errors.append(f"article {type(exc).__name__}: {exc}")

        # 兜底：all/v2（不需要 cookie 也能用）
        if not collected:
            try:
                payload = self._fetch_json(
                    opener, f"{self.ALL_SEARCH}?keyword={keyword}")
                if payload.get("code") == 0:
                    for entry in ((payload.get("data") or {}).get("result") or []):
                        if entry.get("result_type") == "video":
                            collected += self._video_results(entry.get("data") or [], limit)
                            break
                else:
                    errors.append(f"all/v2 code={payload.get('code')}")
            except Exception as exc:
                errors.append(f"all/v2 {type(exc).__name__}: {exc}")

        if not collected:
            hint = ""
            if any("412" in e or "banned" in e for e in errors):
                hint = "（可能被风控，已进入冷却）"
                _DOMAIN_GUARD.note_block("bilibili.com")
            return SearchOutcome(False, query, self.name,
                                 error=f"未取到结果{'；'.join(errors)}{hint}",
                                 elapsed=round(time.time() - started, 2))
        _DOMAIN_GUARD.note_success("bilibili.com")
        collected = collected[:limit]
        # 只对前几个视频做"视频内信息 + 评论"的富化：
        # 每个视频要额外 3 次请求，全量富化既慢又容易触发风控。
        # "字幕拿不到"这类说明只说一次，避免每条结果都重复一段
        subtitle_note_used = False
        for result in collected[:self.enrich_top]:
            if "/video/" not in result.url:
                continue
            _DOMAIN_GUARD.acquire("bilibili.com", max_wait=1.0)
            try:
                self._enrich(opener, result, include_subtitle_note=not subtitle_note_used,
                             query=query)
            except Exception:
                continue
            if result.content and "无字幕数据" in result.content:
                subtitle_note_used = True
        return SearchOutcome(True, query, self.name, collected,
                             elapsed=round(time.time() - started, 2))


class BochaProvider(SearchProvider):
    """博查（Bocha）AI 搜索 API —— 官方接口，推荐方案。

    接口契约（来自官方文档）：
        POST https://api.bocha.cn/v1/web-search
        Authorization: Bearer {API_KEY}
        Body: {query, freshness, summary, count, include, exclude}
        响应: data.webPages.value[] -> name / url / snippet / summary / siteName / datePublished
        freshness: noLimit | oneDay | oneWeek | oneMonth | oneYear | YYYY-MM-DD..YYYY-MM-DD

    为什么它比抓取好（每一项都是前面实测踩到的坑）：
    - 摘要完整（CrossRef 基本没有摘要）；
    - 带 datePublished（抓取只能从正文里猜日期）；
    - 官方接口，没有「访问异常页面」「百度安全验证」这类风控；
    - 可用 freshness 限定时间范围，时效性可控。

    注意：需要 API Key，因此默认不启用；配置 BOCHA_API_KEY 后会放到链首。
    """

    name = "bocha"
    ENDPOINT = "https://api.bocha.cn/v1/web-search"

    def __init__(self, api_key: str, timeout: int = 20, summary: bool = True,
                 default_freshness: str = "noLimit"):
        self.api_key = api_key
        self.timeout = timeout
        self.summary = summary
        self.default_freshness = default_freshness

    @staticmethod
    def _freshness_for(query: str) -> str:
        """按查询的时效紧迫度选时间范围。

        文档提醒「推荐 noLimit，指定范围可能导致范围内无结果」，
        所以只在查询明显在问"当下"时才收紧，其余交给它自己优化。
        """
        text = query or ""
        if re.search(r"(今天|今日|刚刚|刚才|实时)", text):
            return "oneDay"
        if re.search(r"(这周|本周|最近几天)", text):
            return "oneWeek"
        if re.search(r"(最近|近期|最新|现在|目前|当前|活动|更新|卡池|版本)", text):
            return "oneMonth"
        return "noLimit"

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        if not self.api_key:
            return SearchOutcome(False, query, self.name,
                                 error="未配置 BOCHA_API_KEY",
                                 elapsed=round(time.time() - started, 2))
        payload: Dict[str, Any] = {
            "query": query,
            "count": max(1, min(int(limit), 50)),
            "freshness": self._freshness_for(query) or self.default_freshness,
        }
        if self.summary:
            payload["summary"] = True
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.ENDPOINT, data=body_bytes, method="POST")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")
        try:
            with net.open(request, timeout=self.timeout) as response:
                raw = _decode_body(response.read(600_000), response.headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            hint = ""
            lowered = detail.lower()
            # 403 有两种完全不同的含义，实测踩过：Key 有效但账户没额度时也是 403。
            # 若一律说"Key 无效"，用户会去反复换 Key，方向完全错。
            if "quota" in lowered or "money" in lowered or "balance" in lowered:
                hint = "（Key 有效，但账户没有余额/套餐额度——需要到控制台充值或领取免费额度）"
            elif exc.code == 401:
                hint = "（API Key 无效或未启用）"
            elif exc.code == 403:
                hint = "（无权限，可能是 Key 未开通该接口）"
            elif exc.code == 429:
                hint = "（频率限制，稍后再试）"
            return SearchOutcome(False, query, self.name,
                                 error=f"HTTP {exc.code}{hint}: {detail}",
                                 elapsed=round(time.time() - started, 2))
        except Exception as exc:
            return SearchOutcome(False, query, self.name,
                                 error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return SearchOutcome(False, query, self.name, error="返回内容不是合法 JSON",
                                 elapsed=round(time.time() - started, 2))
        if data.get("code") not in (None, 200):
            return SearchOutcome(False, query, self.name,
                                 error=f"接口返回 code={data.get('code')} msg={data.get('msg')}",
                                 elapsed=round(time.time() - started, 2))

        pages = (((data.get("data") or {}).get("webPages") or {}).get("value") or [])
        results: List[SearchResult] = []
        for item in pages:
            title = str(item.get("name") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url or not is_safe_url(url):
                continue
            snippet = str(item.get("summary") or item.get("snippet") or "").strip()
            published = str(item.get("datePublished") or "")[:10]
            site = str(item.get("siteName") or domain_of(url))
            extra = "；".join(x for x in (published, site) if x)
            if extra:
                snippet = f"{snippet[:MAX_SNIPPET_CHARS]}（{extra}）"
            else:
                snippet = snippet[:MAX_SNIPPET_CHARS]
            results.append(SearchResult(title=title, url=url, snippet=snippet,
                                        source=domain_of(url) or site,
                                        trusted=is_trusted(url)))
            if len(results) >= limit:
                break
        if not results:
            return SearchOutcome(False, query, self.name, error="接口未返回结果",
                                 elapsed=round(time.time() - started, 2))
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results=results,
                             elapsed=round(time.time() - started, 2))


class BaiduProvider(SearchProvider):
    """百度网页搜索（带 cookie 会话）。

    手法参照 python-baidusearch（PyPI: baidusearch 1.0.3）：
    先访问 www.baidu.com 建立 cookie（BAIDUID 等），再带 Referer 请求 /s。
    这是**正确的做法**——不带会话时百度必然返回「百度安全验证」页。

    **实测结论（本机 IP）**：即使照做，放行率仍是 0/6。
    百度对本机出口 IP 的风控是概率性且持续性的，因此：
    - 这个 provider 默认**不放进链里**，需要时用 SEARCH_CHAIN 显式加入；
    - 被拦截时明确报告"安全验证"，而不是静默返回空结果
      （后者会让人误以为"百度没收录"，其实是根本没访问到）。

    另外注意：百度结果链接是 /link?url=... 跳转形式，无法还原真实地址，
    因此这里会丢弃这类条目，只保留能直接访问的。

    想要稳定的百度结果，只能用官方 API（BAIDU_API_KEY），
    配了 key 时这里会明确提示"官方端点尚未接入"，避免假装能用。
    """

    name = "baidu"
    BLOCK_MARKERS = ("百度安全验证", "wappass.baidu.com", "security-verify")
    HOME = "https://www.baidu.com/"
    SEARCH = "https://www.baidu.com/s?ie=utf-8&tn=baidu&wd="
    # 与 baidusearch 一致的 UA（实测用新 UA 并不会更容易放行）
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36")

    def __init__(self, timeout: int = 15, api_key: str = ""):
        self.timeout = timeout
        self.api_key = api_key
        self._opener = None
        self._warmed = False

    def _get_opener(self):
        if self._opener is None:
            jar = http.cookiejar.CookieJar()
            self._opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(jar))
        return self._opener

    def _headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": self.UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _warm_up(self, opener) -> None:
        """先访问首页拿 cookie；失败也不致命，搜索时会被拦并如实报告。"""
        if self._warmed:
            return
        try:
            request = urllib.request.Request(self.HOME, headers=self._headers())
            with opener.open(request, timeout=self.timeout) as response:
                response.read(200_000)
        except Exception:
            pass
        self._warmed = True

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        started = time.time()
        if self.api_key:
            return SearchOutcome(False, query, self.name,
                                 error="已配置 BAIDU_API_KEY，但官方搜索端点尚未接入"
                                       "（需要确认具体接口文档后实现）",
                                 elapsed=round(time.time() - started, 2))
        remaining = _DOMAIN_GUARD.blocked_for("baidu.com")
        if remaining > 0:
            return SearchOutcome(False, query, self.name,
                                 error=f"处于封禁冷却中（还需 {int(remaining)} 秒）",
                                 elapsed=round(time.time() - started, 2))
        _DOMAIN_GUARD.acquire("baidu.com")
        opener = self._get_opener()
        self._warm_up(opener)

        url = self.SEARCH + urllib.parse.quote(query, encoding="utf-8")
        try:
            request = urllib.request.Request(url, headers=self._headers(referer=self.HOME))
            with opener.open(request, timeout=self.timeout) as response:
                status = response.status
                body = _decode_body(response.read(800_000), response.headers)
        except urllib.error.HTTPError as exc:
            return SearchOutcome(False, query, self.name, error=f"HTTP {exc.code}",
                                 elapsed=round(time.time() - started, 2))
        except Exception as exc:
            return SearchOutcome(False, query, self.name,
                                 error=f"{type(exc).__name__}: {exc}",
                                 elapsed=round(time.time() - started, 2))

        if any(marker in body for marker in self.BLOCK_MARKERS) or looks_blocked(body):
            _DOMAIN_GUARD.note_block("baidu.com")
            return SearchOutcome(
                False, query, self.name,
                error="被百度的安全验证拦截（即使带 cookie 会话也一样；"
                      "本机实测放行率 0/6。需要官方 API 才能稳定使用）",
                elapsed=round(time.time() - started, 2))
        if status != 200:
            return SearchOutcome(False, query, self.name, error=f"HTTP {status}",
                                 elapsed=round(time.time() - started, 2))

        # 结果块：百度结构不稳定，h3 与 c-container 都试
        results: List[SearchResult] = []
        blocks = re.split(r'<div[^>]+class="[^"]*(?:result[^"]*c-container|c-container[^"]*result)',
                          body)[1:]
        if not blocks:
            blocks = re.split(r"<h3[^>]*>", body)[1:]
        for block in blocks:
            title_match = re.search(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.{4,160}?)</a>',
                                    block, re.S)
            if not title_match:
                continue
            title = html.unescape(re.sub(r"<[^>]+>", "", title_match.group(2))).strip()
            title = re.sub(r"\s+", " ", title)
            target = html.unescape(title_match.group(1))
            # 百度跳转链接（/link?url=）还原不了真实地址，丢弃而不是给出无效链接
            if "baidu.com/link" in target or not is_safe_url(target) or is_low_value(target):
                continue
            if _too_off_topic(title, query):
                continue
            snippet_match = re.search(r'class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>', block, re.S)
            snippet = ""
            if snippet_match:
                snippet = html.unescape(re.sub(r"<[^>]+>", " ", snippet_match.group(1)))
                snippet = re.sub(r"\s+", " ", snippet).strip()[:MAX_SNIPPET_CHARS]
            results.append(SearchResult(title=title, url=target, snippet=snippet,
                                        source=domain_of(target), trusted=is_trusted(target)))
            if len(results) >= limit:
                break
        if not results:
            return SearchOutcome(False, query, self.name,
                                 error="未解析出可用结果（百度结构变化，或结果只有跳转链接）",
                                 elapsed=round(time.time() - started, 2))
        results.sort(key=lambda r: (not r.trusted,))
        return SearchOutcome(True, query, self.name, results,
                             elapsed=round(time.time() - started, 2))


class ChainedProvider(SearchProvider):
    """多源串行回退。

    为什么需要它：实测没有哪个中文检索源是单独可靠的——
    - 360：能出结果，但部分内容滞后；
    - cn.bing.com：能出结果，但会把系列名拆词（蔚蓝档案 -> Celeste）；
    - DuckDuckGo：返回 202 反爬页；
    - 百度：PC/移动/tn=json 全部撞"百度安全验证"。
    所以按顺序尝试，用第一个真正返回结果的源，并把"实际用了哪个"记下来。
    """

    name = "chain"

    def __init__(self, providers: List[SearchProvider], cache_ttl: int = 300):
        self.providers = [p for p in providers if p is not None]
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, SearchOutcome] = {}
        self._cache_at: Dict[str, float] = {}

    @property
    def provider_names(self) -> str:
        return " -> ".join(p.name for p in self.providers)

    @staticmethod
    def filter_relevant(outcome: SearchOutcome, query: str) -> SearchOutcome:
        """逐条过滤掉与查询无关的结果。

        为什么需要：批次级相关度只能保证"这批里有些是相关的"，
        垃圾结果会搭着好结果混进来——实测查询"什么是量子纠缠"时，
        返回里混进了《女人的第一次，到底是什麼感覺呢？》。
        所以对每条结果单独判定：标题或摘要里至少命中一个查询词才保留。
        """
        if not outcome.results:
            return outcome
        terms = relevance_terms(query)
        if not terms:
            return outcome   # 判不了就不判，避免误删
        lowered = [t.lower() for t in terms]
        kept: List[SearchResult] = []
        for result in outcome.results:
            haystack = f"{result.title} {result.snippet}".lower()
            if any(term in haystack for term in lowered):
                kept.append(result)
        if not kept:
            return outcome   # 全被滤掉说明判定过严，保留原样交给批次级相关度处理
        outcome.results = kept
        return outcome

    @staticmethod
    def relevance(outcome: SearchOutcome, query: str) -> float:
        """结果与查询的相关度：标题里覆盖了多少个查询词。

        为什么需要：实测踩过——查"基沃托斯 游戏开发部"时 360 失败，
        链里下一个源（Bing）返回了「基（汉字）_百度百科」这种完全无关的结果。
        只看"有没有结果"就会把垃圾当答案发出去。
        """
        terms = relevance_terms(query)
        if not terms:
            # 判不了就不判：按"通过"处理，避免把好结果误标成低相关。
            return 1.0
        covered = 0
        for term in terms:
            if any(term.lower() in (r.title or "").lower() for r in outcome.results):
                covered += 1
        return covered / len(terms)

    def search(self, query: str, limit: int = MAX_RESULTS) -> SearchOutcome:
        key = f"{query}|{limit}"
        cached = self._cache.get(key)
        if cached is not None and time.time() - self._cache_at.get(key, 0) < self._cache_ttl:
            return cached

        errors: List[str] = []
        started = time.time()
        best: Optional[SearchOutcome] = None
        best_score = 0.0

        for provider in self.providers:
            try:
                outcome = provider.search(query, limit=limit)
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue
            if not (outcome.ok and outcome.results):
                errors.append(f"{provider.name}: {outcome.error or '无结果'}")
                continue

            outcome = self.filter_relevant(outcome, query)
            score = self.relevance(outcome, query)
            outcome.provider = provider.name
            # 相关度达标就直接采用，不再为"可能更好"付额外延迟
            if score >= CHAIN_RELEVANCE_MIN:
                self._cache[key] = outcome
                self._cache_at[key] = time.time()
                return outcome
            # 不达标：先留着当兜底，继续问下一个源
            errors.append(f"{provider.name}: 相关度低（{score:.2f}）")
            # 注意条件里的 best is None：相关度可能全为 0，
            # 若只写 score > best_score（初始 0.0），当所有源都是 0 分时
            # best 永远不会被赋值，最后会错误地报"全部失败"。
            if best is None or score > best_score:
                best, best_score = outcome, score

        if best is not None:
            # 所有源都只是"低相关"：仍然返回最好的那份，但把情况说清楚，
            # 让模型/用户知道这批结果可能没对准问题，以及前面为什么被跳过。
            # 把"前面的源为什么被跳过"一并写出来，否则用户只看到结果变差、看不到原因。
            reasons: List[str] = []
            for entry in errors:
                name = entry.split(":")[0].strip()
                if "余额" in entry or "quota" in entry.lower():
                    reasons.append(f"{name} 无额度")
                elif "Key" in entry or "401" in entry:
                    reasons.append(f"{name} 鉴权失败")
                elif "异常访问" in entry or "安全验证" in entry or "冷却" in entry:
                    reasons.append(f"{name} 被风控拦截")
            note = "相关度较低，建议换个说法再试"
            if reasons:
                note += "；" + "、".join(reasons)
            best.provider = f"{best.provider}（{note}）"
            self._cache[key] = best
            self._cache_at[key] = time.time()
            return best

        failed = SearchOutcome(False, query, self.provider_names,
                               error="；".join(errors) or "所有检索源都没有结果",
                               elapsed=round(time.time() - started, 2))
        self._cache[key] = failed
        self._cache_at[key] = time.time()
        return failed


class SearchService:
    """对上层暴露的统一搜索入口：查询清洗 + 缓存 + 正文抓取。"""

    def __init__(self, provider: Optional[SearchProvider] = None, *, cache_ttl: int = 600,
                 fetch_pages: int = MAX_FETCH_PAGES, timeout: int = 15):
        self.provider = provider or DuckDuckGoProvider(timeout=timeout)
        self.cache = _Cache(cache_ttl)
        self.fetch_pages = max(0, fetch_pages)
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.provider is not None

    def search(self, query: str, limit: int = MAX_RESULTS, *, with_content: bool = True) -> SearchOutcome:
        cleaned = clean_query(query)
        if not cleaned:
            return SearchOutcome(False, query, self.provider.name, error="查询为空")
        cached = self.cache.get(cleaned)
        if isinstance(cached, SearchOutcome):
            return cached
        outcome = self.provider.search(cleaned, limit=limit)
        if outcome.ok and with_content and self.fetch_pages:
            total = 0
            for index, result in enumerate(outcome.results):
                if index >= self.fetch_pages:
                    break
                if total >= MAX_TOTAL_CONTENT_CHARS:
                    break
                content = fetch_page(result.url, timeout=self.timeout)
                if content:
                    result.content = content
                    total += len(content)
        self.cache.set(cleaned, outcome)
        return outcome

    def format_for_model(self, outcome: SearchOutcome, max_results: int = 4) -> str:
        """把结果整理成给模型看的文本块。"""
        return _format_results(outcome, max_results)


# 需要用最新信息回答的信号。命中这些就值得先检索，而不是拿旧知识硬答。
def _has(pattern: str) -> bool:
    return False


TIME_SENSITIVE_PATTERNS = (
    # 时间词必须和"发生了什么/有没有"绑定，否则"今天好累"也会去联网。
    # 早期把 (今天|最近|现在) 单独列为一条，测试用例 test_needs_search_intent 抓到了这个过宽。
    re.compile(r"(今天|今日|昨天|明天|最近|近期|现在|目前|当前|这几天)[^。！？!?]{0,8}"
               r"(有(什么|没有|啥)|发生|出了?什么|更新|上线|开服|结束|开始|公布|宣布|新闻)"),
    re.compile(r"(最新|最新的|新版|新活动|新版本|新卡池|新皮肤|前瞻|预告|开服|上线|停服|维护)"),
    re.compile(r"(什么时候|几号|几月几号|几点|日期|时间表|日程|赛程|截止)"),
    re.compile(r"(多少钱|价格|售价|涨价|降价|汇率|费用)"),
    re.compile(r"(新闻|资讯|怎么了|进展|结果|比分|排名)"),
    re.compile(r"(搜索|搜一下|查一下|查查|上网查|联网|百度|google|谷歌)", re.IGNORECASE),
    re.compile(r"(活动|限定|复刻|联动|卡池|版本更新)"),
    re.compile(r"(是不是真的|到底|准确吗|确认一下|真的假的)"),
)


# 助词/虚词：出现它们不代表有"话题"，只说明这是句问句
_FUNCTION_WORDS = set(
    "现在 目前 当前 今天 昨天 明天 最近 近期 最新 活动 什么 是什么 有什么 怎么 怎样 哪些 哪个 "
    "游戏 里 的 了 吗 呢 啊 吧 请 帮我 查 一下 搜索 最近 开 正在 还有 多少 时候 情况 信息 内容 "
    "有 没有 是 在 和 与 也 都 就 很 太 可以 能否 应该".split())


def has_topic(text: str) -> bool:
    """查询里是否含有"话题词"（能让搜索引擎定位到具体对象的词）。

    实测：把"现在游戏里正在开的活动是什么"直接丢给搜索引擎，
    会返回《王者荣耀》官网、图片站、短视频聚合页——缺少话题词的查询
    不但搜不到答案，还会引入错误信息。

    判定方式：只要存在一个"不是虚词、长度>=2"的片段就算有话题。
    "蔚蓝档案"这种专名会通过；"现在游戏里正在开的活动是什么"会被拒。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    for term in _query_terms(cleaned):
        if term in _FUNCTION_WORDS:
            continue
        # 去掉纯虚词组合（如"最新的情况"）
        stripped = term
        for word in sorted(_FUNCTION_WORDS, key=len, reverse=True):
            stripped = stripped.replace(word, "")
        # 阈值取 3：实测"最新消息是什么"剥掉虚词后剩"消息"（2 字），
        # 仍然没有可检索的对象，搜了只会返回无关内容。
        if len(stripped) >= 3:
            return True
    return False


def needs_search(text: str) -> bool:
    """判断这条消息是否应该先检索。

    刻意保守：只在明显的时效性/事实性问题上返回 True，
    避免把普通闲聊也送去联网（既慢又浪费）。
    """
    cleaned = (text or "").strip()
    if len(cleaned) < 4:
        return False
    # 纯问候、情绪表达不值得检索
    if re.fullmatch(r"[sS]{0,12}", cleaned) and not re.search(r"[?？]", cleaned):
        pass
    return any(pattern.search(cleaned) for pattern in TIME_SENSITIVE_PATTERNS)


def format_for_model(outcome: SearchOutcome, max_results: int = 4) -> str:
    """模块级版本，方便引擎在自动检索时复用。"""
    return SearchService().format_for_model(outcome, max_results) if False else _format_results(outcome, max_results)


def _format_results(outcome: SearchOutcome, max_results: int = 4) -> str:
    if not outcome.ok:
        return f"（检索失败：{outcome.error}）"
    if not outcome.results:
        return "（没有检索到相关结果）"
    lines: List[str] = []

    # 时效警示放在最前面：这是最容易被忽略、也最影响可信度的信息。
    age = freshest_content_age(outcome)
    if age < 0:
        lines.append("【时效提示】无法从结果中判断发布时间，回答时不要声称这是"
                     "「最新」消息，必要时说明无法确认时效。")
    elif age <= 3:
        lines.append(f"【时效提示】结果里有 {age} 天内的内容，可以作为较新的信息使用。")
    elif age <= 30:
        lines.append(f"【时效提示】结果里最新的内容距今约 {age} 天，"
                     "回答时请说明这是约几周前的信息。")
    else:
        months = max(1, round(age / 30))
        lines.append(f"【时效提示】结果里最新的内容距今约 {age} 天（约 {months} 个月），"
                     "**很可能已经过时**。必须明确告诉用户这是旧信息、"
                     "不要当作「现在正在进行的」内容来回答，并建议以游戏内公告为准。")
    lines.append("")
    for index, result in enumerate(outcome.results[:max_results], 1):
        tag = "可信来源" if result.trusted else "普通来源"
        lines.append(f"[{index}] {result.title}")
        lines.append(f"    来源：{result.source}（{tag}）")
        lines.append(f"    链接：{result.url}")
        if result.snippet:
            lines.append(f"    摘要：{result.snippet}")
        else:
            # 摘要为空不能让模型空手推断；标题本身往往就带时间和事件
            lines.append(f"    摘要：（搜索页未提供摘要，标题为：{result.title}）")
        if result.content:
            freshness = _latest_date_in(result.content)
            if freshness:
                lines.append(f"    正文节选（其中最新日期：{freshness}；请优先采信这部分，"
                             f"并据此判断信息是否过时）：")
            else:
                lines.append("    正文节选：")
            lines.append(f"      {result.content}")
        lines.append("")
    return "\n".join(lines).strip()


def build_service(config, asr_client=None, asr_guard=None) -> Optional[SearchService]:
    """按配置构建检索服务；未启用或配置不合法时返回 None。"""
    features = getattr(config, "features", None)
    settings = getattr(config, "search", None)
    if features is None or settings is None:
        return None
    if not (features.web_search and settings.enabled):
        return None
    set_disambiguation(bool(getattr(settings, "disambiguate", False)))
    provider_name = (settings.provider or "so360").lower()
    timeout = int(settings.timeout_seconds or 15)
    if provider_name in ("", "none", "off", "disabled"):
        return None
    def _one(name: str):
        if name == "searxng" and settings.searxng_url:
            return SearXNGProvider(settings.searxng_url, timeout=timeout)
        if name == "duckduckgo":
            return DuckDuckGoProvider(timeout=timeout)
        if name == "bing":
            return BingProvider(timeout=timeout)
        if name == "baidu":
            return BaiduProvider(timeout=timeout,
                                 api_key=getattr(settings, "baidu_api_key", "") or "")
        if name == "bocha":
            key = getattr(settings, "bocha_api_key", "") or ""
            if not key:
                return None
            return BochaProvider(key, timeout=timeout,
                                 summary=bool(getattr(settings, "bocha_summary", True)))
        if name == "bilibili":
            asr_settings = getattr(config, "asr", None)
            provider_kind = str(getattr(asr_settings, "provider", "") or "").lower()
            is_local = provider_kind in ("local", "faster_whisper", "faster-whisper",
                                        "whisper_local")
            return BilibiliProvider(
                timeout=timeout,
                sessdata=getattr(settings, "bilibili_sessdata", "") or "",
                enrich_top=int(getattr(settings, "bilibili_enrich_top", 2) or 0),
                comments_limit=int(getattr(settings, "bilibili_comments", 8) or 0),
                subtitle_chars=int(getattr(settings, "bilibili_subtitle_chars", 1200) or 0),
                comments_chars=int(getattr(settings, "bilibili_comments_chars", 600) or 0),
                asr_client=asr_client, asr_guard=asr_guard,
                asr_max_bytes=int(getattr(asr_settings, "max_bytes", 8 * 1024 * 1024) or 0),
                asr_max_seconds=int(getattr(asr_settings, "max_seconds", 300) or 0),
                asr_segment_seconds=int(getattr(asr_settings, "segment_seconds", 300) or 0),
                asr_max_segments=int(getattr(asr_settings, "max_segments", 24) or 0),
                asr_local=is_local,
                asr_ffmpeg=find_ffmpeg(config))
        if name == "so360":
            return So360Provider(timeout=timeout)
        return None

    if provider_name in ("chain", "auto", ""):
        order = list(settings.chain or ["bilibili", "bocha", "so360", "bing"])
        # 顺序完全尊重 SEARCH_CHAIN 的配置，不再自动重排——
        # 早期"有 key 就把博查提到链首"会覆盖用户明确指定的优先级。
        # 只在没有 key 时把 bocha 摘掉，避免每次白跑一次鉴权失败。
        if not getattr(settings, "bocha_api_key", ""):
            order = [n for n in order if n != "bocha"]
        providers = [p for p in (_one(str(n).strip().lower()) for n in order) if p]
        if not providers:
            return None
        provider = ChainedProvider(providers, cache_ttl=int(settings.cache_ttl_seconds or 300))
    else:
        provider = _one(provider_name)
        if provider is None:
            return None
    return SearchService(provider, cache_ttl=int(settings.cache_ttl_seconds or 600),
                         fetch_pages=int(settings.fetch_pages or 0), timeout=timeout)
