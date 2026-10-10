# -*- coding: utf-8 -*-
"""在 B站搜爱丽丝的配音素材（用于 GPT-SoVITS 零样本克隆）。"""
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, r"D:\DSHWORK\alice-qq-agent")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "Chrome/124.0 Safari/537.36")

# B站是国内的，直连即可；但系统代理掉线会干扰，所以显式直连
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

session_data = ""
env_path = r"D:\DSHWORK\alice-qq-agent\.env"
if os.path.isfile(env_path):
    for line in open(env_path, encoding="utf-8"):
        if line.strip().startswith("BILIBILI_SESSDATA"):
            session_data = line.split("=", 1)[1].strip().strip('"')
print("  SESSDATA:", "已配置(%d 字符)" % len(session_data) if session_data else "未配置")

HEADERS = {
    "User-Agent": UA,
    "Referer": "https://www.bilibili.com/",
    "Cookie": "SESSDATA=" + session_data if session_data else "",
}

QUERIES = [
    "蔚蓝档案 爱丽丝 中文配音",
    "天童爱丽丝 语音",
    "蔚蓝档案 中配 爱丽丝",
    "蔚蓝档案 爱丽丝 语音包",
]


def search(query, page=1):
    params = urllib.parse.urlencode({"search_type": "video", "keyword": query, "page": page})
    url = "https://api.bilibili.com/x/web-interface/search/type?" + params
    req = urllib.request.Request(url, headers=HEADERS)
    with OPENER.open(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    if data.get("code") != 0:
        raise RuntimeError("API code=%s msg=%s" % (data.get("code"), data.get("message")))
    return (data.get("data") or {}).get("result") or []


import re
for query in QUERIES:
    print()
    print("=== %s ===" % query)
    try:
        results = search(query)
    except Exception as exc:
        print("  搜索失败:", type(exc).__name__, str(exc)[:100])
        continue
    print("  命中 %d 条" % len(results))
    for item in results[:8]:
        title = re.sub(r"<[^>]+>", "", item.get("title") or "")
        bvid = item.get("bvid") or ""
        duration = item.get("duration") or ""
        play = item.get("play") or 0
        author = item.get("author") or ""
        print("    %-52s" % title[:52])
        print("      %s | %s | 播放 %s | UP %s" % (bvid, duration, play, author[:14]))
