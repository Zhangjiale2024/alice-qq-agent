"""唱歌：从"我想听爱丽丝唱 X"到一条语音成品。

这个模块**尽量只依赖标准库**，所以能离线测试；真正跑 RVC 的那条链在
`D:\\DSHWORK\\GPT-SoVITS-setup\\sing.py`，通过子进程调用。

为什么要单独一个模块，而不是把逻辑塞进适配层：
  * "要不要唱"的判定、去重、缓存、限流都是**业务逻辑**，属于核心；
  * 适配层只该负责"把成品当语音发出去"这件事。

**最重要的一条设计**：整条流水线要跑 60~90 秒，
所以绝不能阻塞消息处理——
`SingingJob` 只是"一个正在跑的活"，由调用方（适配层）放到线程里跑，
跑完再回来取结果。判定和状态都是纯数据，因此可测。

合规：产物仅本地/熟人自用，不做公开服务（见 docs 项目状态交接 第十五节）。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---- 唱歌意图识别 ---------------------------------------------------------

# 明确"在点一首歌"的说法。用**动词开头**来定位，只取动词之后的歌名。
#
# 为什么不用 `唱([^，。]{1,30})` 这种宽泛写法（第一版就是那样，漏判连连）：
#   "我想听你唱《虫儿飞》" -> 会抽出 "你唱《虫儿飞"
#   "爱丽丝唱首歌"        -> 会抽出 "首歌"
# 正确的是先找到"唱/来一首/点歌"这些词，**再只看它后面**的内容。
_SING_VERB = re.compile(r"(唱|来一首|点歌)")
# 明显不是点歌的场景（在聊唱歌这件事，而不是点具体的歌）
_NOT_SINGING = re.compile(
    r"(唱歌(好听|难听|怎么样|吗)|会唱|能不能唱|喜欢唱|唱歌技巧|"
    r"歌单|歌词|推荐.{0,4}歌|什么歌|哪首歌|歌名|合唱|对唱|唱得)"
)
# 歌名前面常见的填充词，需要剥掉
# （"首歌"也在内："唱首歌来听听" 里的"首歌"不是歌名）
_FILLER = re.compile(
    r"^(一下|一首|首歌|一段|一个|个|首|段|点|的|呗|吧|呀|啊|嘛|"
    r"给我|给我听|我听|听|来|那个|这个|\s)+"
)
# 句子结尾的语气词/无意义成分
_TAIL = re.compile(
    r"(来听听|来听|听听|吧|呗|呀|啊|嘛|哦|喔|啦|了|好不好|行吗|可以吗|给我听|给我)+$"
)
# 「作品名 + 的」前缀：用户常写"唱一首蔚蓝档案的《X》"，
# "蔚蓝档案的"是出处不是歌名，必须剥掉再去搜。
# 实测（2026-10-10 截图）：整串当歌名搜出来的东西完全不相干——
# 要《准备出发》却搜出《钱是万恶之源吗》。
_ORIGIN_PREFIX = re.compile(r"^[^《<「『]{1,12}?(的|中的|里的|里面的)(?=[《<「『])")
# 用户纠正自己时常见说法："唱全速向光不是反乌托邦"——
# 整句带判断词，**不是歌名**，应当拒绝而不是硬搜。
_NOT_A_TITLE = re.compile(
    r"(不是|不对|而是|应该|换成|改成|怎么样|怎么唱|为什么|吗$|呢$)"
)


def _extract_bracketed(line):
    """优先取书名号里的内容——那是用户明确标出的歌名。"""
    for left, right in (("《", "》"), ("<", ">"), ("「", "」"), ("『", "』")):
        start = line.find(left)
        if start < 0:
            continue
        end = line.find(right, start + 1)
        if end < 0:
            continue
        inner = line[start + 1:end].strip()
        if "《" in inner:                    # 嵌套时取最里层
            inner = inner.rsplit("《", 1)[-1].strip()
        if inner:
            return inner
    return None


def detect_song(text: str) -> Optional[str]:
    """从一句话里抽出"要唱哪首歌"。抽不到返回 None。

    只认"明确在点一首歌"的说法，宁可漏判也不要乱唱：
    说"你会唱歌吗"、"唱歌好听吗"、"你唱得真好"这种聊天
    不该触发一次 90 秒的重活。
    """
    if not text:
        return None
    line = text.strip()
    verb = _SING_VERB.search(line)
    if not verb:
        return None
    if _NOT_SINGING.search(line):
        return None

    # 只取动词之后的内容当候选歌名
    song = line[verb.end():].strip()
    # 1) 有书名号就以它为准（用户明确标了歌名）
    bracketed = _extract_bracketed(song)
    if bracketed:
        song = bracketed
    else:
        # 2) 没书名号：先剥"出处 + 的"前缀，再剥填充词
        song = song.strip("《》<>「」\"'“” \t:：,，")
        song = _ORIGIN_PREFIX.sub("", song)
        prev = None
        while prev != song:                  # 反复剥，直到稳定
            prev = song
            song = _FILLER.sub("", song).strip("《》<>「」\"'“” \t:：,，")
        song = _TAIL.sub("", song).strip().strip("《》<>「」\"'“” \t")
    if len(song) < 2 or len(song) > 30:
        return None
    # 只剩"歌/歌曲"这类词不算歌名
    if song in ("歌", "首歌", "歌曲", "这个", "那个", "什么", "什么歌"):
        return None
    # 带判断词的句子不是歌名（纠正、疑问），硬搜只会搜出不相干的歌
    if _NOT_A_TITLE.search(song):
        return None
    return song


# ---- 缓存：同一首歌不重复跑 90 秒 -----------------------------------------

def _safe_name(text: str) -> str:
    """把歌名变成安全的文件名（去掉路径分隔符与非法字符）。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:60] or "song"


class SongCache:
    """已做好的歌：命中就直接发，不再跑流水线。

    key 用规范化后的歌名（小写、去空格），所以
    "反方向的钟" / " 反方向的钟 " 会命中同一条。
    """

    def __init__(self, root: str):
        self.root = root

    def _key(self, song: str) -> str:
        return re.sub(r"\s+", "", (song or "").lower())

    def _path(self, song: str) -> str:
        return os.path.join(self.root, _safe_name(self._key(song)) + ".mp3")

    def lookup(self, song: str) -> Optional[str]:
        path = self._path(song)
        return path if os.path.isfile(path) else None

    def store(self, song: str, source: str) -> Optional[str]:
        """把成品拷进缓存目录。拷贝失败返回 None（不影响本次发送）。"""
        if not source or not os.path.isfile(source):
            return None
        try:
            os.makedirs(self.root, exist_ok=True)
            dest = self._path(song)
            if os.path.abspath(source) != os.path.abspath(dest):
                with open(source, "rb") as src, open(dest, "wb") as out:
                    out.write(src.read())
            return dest
        except OSError:
            return None


# ---- 任务：一次点歌跑完的产物 ---------------------------------------------

@dataclass
class SingResult:
    ok: bool
    song: str
    audio: str = ""          # 成品音频绝对路径（ok=True 时有效）
    title: str = ""          # 实际选中的视频标题
    error: str = ""          # ok=False 时的**真实**失败原因
    elapsed: float = 0.0
    cached: bool = False


# 一次只允许一个任务：这台机器跑一次转换就把虚拟内存吃得很紧，
# 再叠一个必然两个都崩（实测过"页面文件太小"）。
_LOCK = threading.Lock()

# 取消标志与"当前任务代次"。
# 截图实录：用户发了"停"，机器人还是把歌发出来了——
# 因为结果已经在返回路上。所以只杀进程不够，还要让**已跑完的结果失效**。
_CANCEL = threading.Event()
_ACTIVE_TOKEN = {"value": 0}


def request_cancel() -> None:
    """请求取消当前任务（用户说"停"，或又点了另一首）。

    做两件事：
      1. 置取消标志——工作线程会看到并杀掉子进程；
      2. 递增 token——已经跑完但还没发送的结果会被判定为**过期**，直接丢弃。

    第 2 点同样重要，否则"停"形同虚设：结果已经拿到手了，
    只杀进程是拦不住它被发出去的。
    """
    _CANCEL.set()
    _ACTIVE_TOKEN["value"] += 1


def is_stale(token: int) -> bool:
    """这个 token 对应的任务是否已被取消或被新任务顶替。"""
    return token != _ACTIVE_TOKEN["value"]


def current_token() -> int:
    return _ACTIVE_TOKEN["value"]


def reset_cancel() -> None:
    """清掉取消标志。每轮任务结束时调用，否则下一次点歌会被立刻取消。"""
    _CANCEL.clear()


def busy() -> bool:
    """是否已有任务在跑。"""
    return _LOCK.locked()


def make_runner(script_path: str, python: str, timeout: int = 900):
    """返回一个 `run(song) -> SingResult` 函数。

    单独抽出来是为了让测试能注入假的 runner（不真的跑 90 秒流水线）。
    """

    def run(song: str) -> SingResult:
        started = time.time()
        if not os.path.isfile(script_path):
            return SingResult(False, song,
                              error=f"唱歌脚本不存在：{script_path}")
        acquired = _LOCK.acquire(blocking=False)
        if not acquired:
            return SingResult(False, song,
                              error="已经有一首歌在做了，等它完成再点")
        try:
            proc = subprocess.Popen([python, "-u", script_path, song],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8",
                                    errors="ignore",
                                    env=dict(os.environ, PYTHONUTF8="1",
                                             PYTHONIOENCODING="utf-8"))
            # 边等边看取消标志：用户说"停"时要**真的杀掉**子进程，
            # 而不是等它跑完再把结果丢掉（那要白等 90 秒）。
            deadline = started + timeout
            while True:
                if _CANCEL.is_set():
                    _kill(proc)
                    return SingResult(False, song, error="已按你的要求停下",
                                      elapsed=time.time() - started)
                if proc.poll() is not None:
                    break
                if time.time() > deadline:
                    _kill(proc)
                    return SingResult(False, song,
                                      error=f"超时（超过 {timeout} 秒）",
                                      elapsed=time.time() - started)
                time.sleep(0.3)
            out, err = proc.communicate()
            out = out or ""
            title = ""
            for line in out.splitlines():
                if line.startswith("选中："):
                    title = line.split("：", 1)[1].strip()
            # 脚本约定：把成品路径打印成"成品：<路径>"
            audio = ""
            for line in out.splitlines():
                if line.startswith("成品：") and line.endswith(".mp3"):
                    audio = line.split("：", 1)[1].strip()
            if proc.returncode != 0 or not audio or not os.path.isfile(audio):
                # 失败时把脚本输出的最后几行带回去——**便于定位真因**，
                # 也让用户看到的不是一句空泛的"失败了"。
                tail = " | ".join(
                    out.strip().splitlines()[-3:]
                    or (err or "").strip().splitlines()[-3:])
                return SingResult(False, song, title=title,
                                  error=tail[:300] or f"退出码 {proc.returncode}",
                                  elapsed=time.time() - started)
            return SingResult(True, song, audio=audio, title=title,
                              elapsed=time.time() - started)
        except Exception as exc:
            return SingResult(False, song,
                              error=f"{type(exc).__name__}: {str(exc)[:200]}",
                              elapsed=time.time() - started)
        finally:
            _LOCK.release()
            # 本轮结束就把取消标志清掉，否则下一次点歌会立刻被"取消"
            reset_cancel()

    return run


def _kill(proc) -> None:
    """尽力杀掉子进程及其子进程（流水线会再起子进程）。"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
