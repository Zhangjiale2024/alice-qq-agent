"""学术文献检索：CrossRef / Semantic Scholar / arXiv。

为什么要三个源一起接（而不是只接前两个）——实测结论：

| 源 | 摘要 | 实测可用性 |
| --- | --- | --- |
| CrossRef | 基本没有（DOI 详情接口也不行） | 稳定，适合按标题/DOI 定位文献元数据 |
| Semantic Scholar | 有 | **无 API Key 时基本 429**；有 key 才是主力 |
| arXiv | 完整 | 免费稳定，预印本全文摘要 |

所以策略是：
- 学术意图 + 有 S2 key -> S2 优先（摘要质量最好），失败退到 arXiv/CrossRef；
- 学术意图 + 无 key   -> arXiv + CrossRef（都能用，不必依赖限流接口）；
- 非学术意图          -> 完全不进这里，走网页检索。

设计成独立模块，是因为学术检索与网页搜索的"好坏标准"不同：
学术要的是摘要、作者、年份、DOI；网页要的是时效与正文。
"""
from __future__ import annotations

from . import net
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .search import (MAX_SNIPPET_CHARS, SearchOutcome, SearchProvider, SearchResult,
                     _decode_body, is_safe_url)

# 学术检索必须给出可联系的 UA，CrossRef 的 polite pool 依赖它
ACADEMIC_UA = "AliceQQAgent/2.0 (academic literature lookup; mailto:alice@example.com)"

# 学术意图信号：命中这些词才走文献检索
ACADEMIC_PATTERNS = (
    re.compile(r"(论文|文献|期刊|综述|预印本|学术|科研|引用|参考文献|被引)", re.IGNORECASE),
    re.compile(r"(paper|papers|literature|journal|preprint|arxiv|citation|doi|abstract)", re.IGNORECASE),
    re.compile(r"(研究|实验|理论|算法|模型|数据集).{0,6}(证明|表明|显示|结果|论文)", re.IGNORECASE),
    re.compile(r"(有没有|找一下|帮我找|查一下).{0,8}(论文|文献|研究)", re.IGNORECASE),
    re.compile(r"10\.\d{4,9}/\S+"),  # 直接给了 DOI
)


def is_academic_query(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in ACADEMIC_PATTERNS)


def _strip_jats(text: str) -> str:
    """CrossRef 的摘要常带 JATS XML 标签，去干净再给模型。"""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _academic_get(url: str, timeout: int = 20, headers: Optional[Dict[str, str]] = None):
    request = urllib.request.Request(url, headers={
        "User-Agent": ACADEMIC_UA,
        "Accept": "application/json",
        **(headers or {}),
    })
    with net.open(request, timeout=timeout) as response:
        return response.status, _decode_body(response.read(600_000), response.headers)


@dataclass
class AcademicOutcome:
    """学术检索的统一返回。"""
    ok: bool
    query: str
    provider: str = ""
    results: List[SearchResult] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0
    rate_limited: bool = False


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------

class CrossRefProvider:
    """CrossRef REST API：元数据权威，但摘要多为空。"""

    name = "crossref"
    ENDPOINT = "https://api.crossref.org/works"

    def __init__(self, timeout: int = 20, rows: int = 5, mailto: str = "alice@example.com"):
        self.timeout = timeout
        self.rows = rows
        self.mailto = mailto

    def _item_to_result(self, item: Dict) -> Optional[SearchResult]:
        titles = item.get("title") or []
        title = _strip_jats(titles[0]) if titles else ""
        if not title:
            return None
        doi = str(item.get("DOI") or "").strip()
        url = str(item.get("URL") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if not url or not is_safe_url(url):
            return None

        year = ""
        parts = (item.get("issued") or {}).get("date-parts") or []
        if parts and parts[0]:
            year = str(parts[0][0])
        journal = ""
        containers = item.get("container-title") or []
        if containers:
            journal = _strip_jats(containers[0])

        authors = []
        for author in (item.get("author") or [])[:3]:
            given = author.get("given") or ""
            family = author.get("family") or author.get("name") or ""
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        abstract = _strip_jats(item.get("abstract") or "")
        meta = "；".join(x for x in (year, journal, ", ".join(authors)) if x)
        snippet = (abstract[:MAX_SNIPPET_CHARS] if abstract else f"（该来源未提供摘要）{meta}")
        if abstract and meta:
            snippet = f"{abstract[:MAX_SNIPPET_CHARS]}（{meta}）"
        return SearchResult(title=title, url=url, snippet=snippet,
                            source=f"crossref.org{' · DOI:' + doi if doi else ''}",
                            trusted=True)

    def search(self, query: str, limit: int = 5) -> AcademicOutcome:
        started = time.time()
        # 查询里直接带 DOI 时走精确接口
        doi_match = re.search(r"10\.\d{4,9}/[^\s，,。]+", query)
        if doi_match:
            doi = doi_match.group(0).rstrip(".")
            try:
                status, body = _academic_get(
                    f"{self.ENDPOINT}/{urllib.parse.quote(doi)}?mailto={self.mailto}",
                    timeout=self.timeout)
                item = json.loads(body).get("message", {})
                result = self._item_to_result(item)
                if result:
                    return AcademicOutcome(True, query, self.name, [result],
                                           elapsed=round(time.time() - started, 2))
                return AcademicOutcome(False, query, self.name, error="DOI 存在但缺少标题",
                                       elapsed=round(time.time() - started, 2))
            except Exception as exc:
                return AcademicOutcome(False, query, self.name,
                                       error=f"{type(exc).__name__}: {exc}",
                                       elapsed=round(time.time() - started, 2))

        params = urllib.parse.urlencode({
            "query": query, "rows": max(limit, 3),
            "select": "DOI,title,abstract,URL,issued,container-title,author",
            "mailto": self.mailto,
        })
        try:
            status, body = _academic_get(f"{self.ENDPOINT}?{params}", timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            return AcademicOutcome(False, query, self.name,
                                   error=f"HTTP {exc.code}", elapsed=round(time.time() - started, 2))
        except Exception as exc:
            return AcademicOutcome(False, query, self.name,
                                   error=f"{type(exc).__name__}: {exc}",
                                   elapsed=round(time.time() - started, 2))
        try:
            items = (json.loads(body).get("message") or {}).get("items") or []
        except json.JSONDecodeError:
            return AcademicOutcome(False, query, self.name, error="返回内容不是合法 JSON",
                                   elapsed=round(time.time() - started, 2))

        results = [r for r in (self._item_to_result(i) for i in items) if r][:limit]
        if not results:
            return AcademicOutcome(False, query, self.name, error="没有匹配的文献",
                                   elapsed=round(time.time() - started, 2))
        return AcademicOutcome(True, query, self.name, results,
                               elapsed=round(time.time() - started, 2))


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

class SemanticScholarProvider:
    """Semantic Scholar Graph API。

    **无 API Key 时实测几乎必然 429**（公共池按 IP 共享）。
    因此默认只在配置了 key 时启用；没 key 时调用会明确回报"被限流"，
    由上层退避到 arXiv/CrossRef，而不是静默失败。
    """

    name = "semantic_scholar"
    ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,url,year,authors,venue,externalIds"

    def __init__(self, api_key: str = "", timeout: int = 20, retries: int = 2,
                 backoff_seconds: float = 4.0):
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff_seconds = backoff_seconds

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def search(self, query: str, limit: int = 5) -> AcademicOutcome:
        started = time.time()
        params = urllib.parse.urlencode({"query": query, "limit": max(limit, 3),
                                         "fields": self.FIELDS})
        last_error = ""
        rate_limited = False
        for attempt in range(self.retries + 1):
            try:
                status, body = _academic_get(f"{self.ENDPOINT}?{params}",
                                             timeout=self.timeout, headers=self._headers())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rate_limited = True
                    last_error = ("HTTP 429 被限流（公共池按 IP 共享，"
                                  "配置 SEMANTIC_SCHOLAR_API_KEY 可显著提高额度）")
                    if attempt < self.retries:
                        time.sleep(self.backoff_seconds * (attempt + 1))
                        continue
                    break
                return AcademicOutcome(False, query, self.name, error=f"HTTP {exc.code}",
                                       rate_limited=False,
                                       elapsed=round(time.time() - started, 2))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                break
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return AcademicOutcome(False, query, self.name, error="返回内容不是合法 JSON",
                                       elapsed=round(time.time() - started, 2))
            results: List[SearchResult] = []
            for item in payload.get("data") or []:
                title = (item.get("title") or "").strip()
                url = (item.get("url") or "").strip()
                if not title or not url or not is_safe_url(url):
                    continue
                abstract = (item.get("abstract") or "").strip()
                venue = (item.get("venue") or "").strip()
                year = item.get("year")
                names = [a.get("name") for a in (item.get("authors") or [])[:3] if a.get("name")]
                meta = "；".join(str(x) for x in (year, venue, ", ".join(names)) if x)
                snippet = f"{abstract[:MAX_SNIPPET_CHARS]}（{meta}）" if abstract else meta
                results.append(SearchResult(title=title, url=url, snippet=snippet,
                                            source=f"semanticscholar.org" + (f" · {year}" if year else ""),
                                            trusted=True))
                if len(results) >= limit:
                    break
            if not results:
                return AcademicOutcome(False, query, self.name, error="没有匹配的文献",
                                       elapsed=round(time.time() - started, 2))
            return AcademicOutcome(True, query, self.name, results,
                                   elapsed=round(time.time() - started, 2))
        return AcademicOutcome(False, query, self.name, error=last_error or "请求失败",
                               rate_limited=rate_limited,
                               elapsed=round(time.time() - started, 2))


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

class ArxivProvider:
    """arXiv API：免费、无需 key、摘要完整。学术检索的主要可用来源。"""

    name = "arxiv"
    ENDPOINT = "http://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> AcademicOutcome:
        started = time.time()
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}", "start": 0, "max_results": max(limit, 3),
        })
        request = urllib.request.Request(f"{self.ENDPOINT}?{params}",
                                         headers={"User-Agent": ACADEMIC_UA})
        try:
            with net.open(request, timeout=self.timeout) as response:
                body = _decode_body(response.read(600_000), response.headers)
        except Exception as exc:
            return AcademicOutcome(False, query, self.name,
                                   error=f"{type(exc).__name__}: {exc}",
                                   elapsed=round(time.time() - started, 2))

        results: List[SearchResult] = []
        for entry in re.findall(r"<entry>(.*?)</entry>", body, re.S):
            title_match = re.search(r"<title>(.*?)</title>", entry, re.S)
            title = re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip() if title_match else ""
            link_match = re.search(r"<id>(.*?)</id>", entry)
            url = (link_match.group(1).strip() if link_match else "")
            if not title or not url or not is_safe_url(url):
                continue
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.S)
            abstract = re.sub(r"\s+", " ", html.unescape(summary_match.group(1))).strip() if summary_match else ""
            published = re.search(r"<published>(\d{4})", entry)
            authors = re.findall(r"<name>(.*?)</name>", entry)
            meta = "；".join(x for x in (
                published.group(1) if published else "",
                ", ".join(authors[:3])) if x)
            snippet = f"{abstract[:MAX_SNIPPET_CHARS]}（{meta}）" if abstract else meta
            results.append(SearchResult(
                title=title, url=url, snippet=snippet,
                source=f"arxiv.org" + (f" · {published.group(1)}" if published else ""),
                trusted=True))
            if len(results) >= limit:
                break
        if not results:
            return AcademicOutcome(False, query, self.name, error="没有匹配的预印本",
                                   elapsed=round(time.time() - started, 2))
        return AcademicOutcome(True, query, self.name, results,
                               elapsed=round(time.time() - started, 2))


# ---------------------------------------------------------------------------
# 组合服务
# ---------------------------------------------------------------------------

class AcademicService:
    """把多个学术源串成一条链，并保留"谁失败了"的信息。"""

    def __init__(self, providers: List, cache_ttl: int = 1800):
        self.providers = list(providers)
        self._cache: Dict[str, AcademicOutcome] = {}
        self._cache_ttl = cache_ttl
        self._cache_at: Dict[str, float] = {}

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def provider_names(self) -> str:
        return ", ".join(p.name for p in self.providers)

    @staticmethod
    def _has_doi(query: str) -> bool:
        return bool(re.search(r"10\.\d{4,9}/\S+", query or ""))

    def _ordered(self, query: str) -> List:
        """查询里带 DOI 时，把 CrossRef 提到最前。

        CrossRef 有精确的 /works/{doi} 接口，能直接命中那一篇；
        而 arXiv 会把 DOI 当成普通文本去搜，结果可能不是目标论文。
        """
        if not self._has_doi(query):
            return self.providers
        preferred = [p for p in self.providers if p.name == "crossref"]
        others = [p for p in self.providers if p.name != "crossref"]
        return preferred + others

    def search(self, query: str, limit: int = 5) -> AcademicOutcome:
        cached = self._cache.get(query)
        if cached and time.time() - self._cache_at.get(query, 0) < self._cache_ttl:
            return cached

        attempts: List[str] = []
        rate_limited = False
        for provider in self._ordered(query):
            outcome = provider.search(query, limit=limit)
            if outcome.ok and outcome.results:
                self._cache[query] = outcome
                self._cache_at[query] = time.time()
                return outcome
            attempts.append(f"{provider.name}: {outcome.error}")
            rate_limited = rate_limited or outcome.rate_limited

        combined = AcademicOutcome(False, query, self.provider_names(),
                                   error="；".join(attempts) or "没有可用的学术来源",
                                   rate_limited=rate_limited)
        self._cache[query] = combined
        self._cache_at[query] = time.time()
        return combined


def build_academic_service(config) -> Optional[AcademicService]:
    """按配置组装学术检索链。顺序即优先级。"""
    settings = getattr(config, "academic", None)
    if settings is None or not settings.enabled:
        return None

    providers: List = []
    # 有 S2 key 时它优先：摘要质量最好
    if settings.semantic_scholar_enabled and settings.semantic_scholar_api_key:
        providers.append(SemanticScholarProvider(
            api_key=settings.semantic_scholar_api_key,
            timeout=settings.timeout_seconds, retries=settings.s2_retries))
    elif settings.semantic_scholar_enabled and settings.semantic_scholar_without_key:
        providers.append(SemanticScholarProvider(
            timeout=settings.timeout_seconds, retries=settings.s2_retries))
    if settings.arxiv_enabled:
        providers.append(ArxivProvider(timeout=settings.timeout_seconds))
    if settings.crossref_enabled:
        providers.append(CrossRefProvider(timeout=settings.timeout_seconds))
    if not providers:
        return None
    return AcademicService(providers, cache_ttl=int(settings.cache_ttl_seconds or 1800))


def format_academic(outcome: AcademicOutcome, max_results: int = 4) -> str:
    """把学术结果整理成给模型看的文本块。"""
    if not outcome.ok:
        lines = [f"（学术检索失败：{outcome.error}）"]
        if outcome.rate_limited:
            lines.append("（注意：Semantic Scholar 的公共额度被限流。"
                         "可以如实说明这一点，并建议用户稍后再试或改用标题/DOI 精确查找。）")
        return "\n".join(lines)
    lines = [f"（以下为学术文献结果，来源：{outcome.provider}。"
             "引用时请给出标题与链接，不要编造页码、卷号或引用数）", ""]
    for index, r in enumerate(outcome.results[:max_results], 1):
        lines.append(f"[{index}] {r.title}")
        lines.append(f"    来源：{r.source}")
        lines.append(f"    链接：{r.url}")
        if r.snippet:
            lines.append(f"    摘要：{r.snippet}")
        lines.append("")
    return "\n".join(lines).strip()
