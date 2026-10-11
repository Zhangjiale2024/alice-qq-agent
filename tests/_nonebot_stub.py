# -*- coding: utf-8 -*-
"""最小 NoneBot 桩：让适配层的 `_handle` 能在零依赖环境里被真跑一次。

**为什么需要它。**

适配层整个包在 `if HAS_NONEBOT:` 里面，CI（零依赖）里那段代码
根本不会被定义，所以 `_handle` 从来没被执行过。之前只能对它做
静态检查（"抽了但忘了接线"），但静态检查证明不了：
事件里的图片是否真的走到了引擎、是否真的触发了视觉识别。

这个桩提供适配层真正用到的那几个名字：

    get_driver / on_message / on_notice / PluginMetadata
    Bot / Message / MessageEvent / PrivateMessageEvent
    GroupIncreaseNoticeEvent / GroupDecreaseNoticeEvent

装好之后 `_handle` 会被注册进 `HANDLERS`，测试可以直接调用它。
这就把"事件 → 引擎"这一段从"静态检查"升级成"真的跑过"。

**它不替代真机验证**：桩不能证明 NoneBot 真的会把事件派发过来，
也不能证明 QQ/NapCat 给的字段和文档一致。它只堵住"适配层自身接线错了"
这一类问题——而这恰恰是最容易出错、又最难在真机上定位的一类。
"""
from __future__ import annotations

import sys
import types
from typing import Any, Callable, Dict, List

# 注册到的 handler：键是事件类型名，值是协程函数
HANDLERS: Dict[str, Callable] = {}


class _Matcher:
    """`on_message()` 的返回值。只需要支持 `@_matcher.handle()`。"""

    def __init__(self, kind: str):
        self.kind = kind

    def handle(self):
        def deco(func):
            HANDLERS[self.kind] = func
            return func
        return deco

    # 适配层可能还会用到这几个；留着免得 AttributeError
    def __getattr__(self, name):
        def noop(*a, **k):
            return self
        return noop


class _Driver:
    def __getattr__(self, name):
        def deco(*a, **k):
            if a and callable(a[0]):
                return a[0]

            def inner(func):
                return func
            return inner
        return deco


class PluginMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Message:
    """够用就行：适配层只用 `Message(...)` 构造要发送的内容。"""

    def __init__(self, *segments):
        self.segments = list(segments)

    def __str__(self):
        return "".join(str(s) for s in self.segments)

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return "Message(%r)" % (self.segments,)


class _Segment:
    """消息段。测试里直接造这个来模拟 QQ 给的结构。"""

    def __init__(self, type_: str, **data):
        self.type = type_
        self.data = dict(data)

    def __repr__(self):
        return "Segment(%r, %r)" % (self.type, self.data)


class MessageEvent:
    """消息事件。字段用 QQ/NapCat 实际会给的那些。"""

    def __init__(self, user_id="10001", message=None, self_id="999",
                 message_id="1", time=0.0, sender=None, **extra):
        self.user_id = str(user_id)
        self.self_id = str(self_id)
        self.message_id = str(message_id)
        self.time = time
        self.sender = sender or _Sender()
        self._segments = list(message or [])
        for k, v in extra.items():
            setattr(self, k, v)

    def get_message(self):
        return list(self._segments)

    def get_plaintext(self) -> str:
        return "".join(str(s.data.get("text", "")) for s in self._segments
                       if s.type == "text")


class PrivateMessageEvent(MessageEvent):
    pass


class GroupMessageEvent(MessageEvent):
    def __init__(self, group_id="88888", **kwargs):
        super().__init__(**kwargs)
        self.group_id = str(group_id)


class _Sender:
    def __init__(self, card="", nickname="测试用户", role="member"):
        self.card = card
        self.nickname = nickname
        self.role = role


class GroupIncreaseNoticeEvent(MessageEvent):
    def __init__(self, group_id="88888", **kwargs):
        super().__init__(**kwargs)
        self.group_id = str(group_id)


class GroupDecreaseNoticeEvent(GroupIncreaseNoticeEvent):
    pass


class Bot:
    """假 Bot。只记录调用，不做真实网络请求。"""

    def __init__(self):
        self.sent: List[Any] = []
        self.calls: List[tuple] = []

    async def send(self, event=None, message=None, **kwargs):
        self.sent.append(message)
        self.calls.append(("send", str(message)))
        return {"message_id": len(self.sent)}

    async def get_group_member_info(self, group_id=None, user_id=None, **kwargs):
        self.calls.append(("get_group_member_info", group_id, user_id))
        return {"role": "member", "user_id": str(user_id)}

    async def get_group_member_list(self, group_id=None, **kwargs):
        self.calls.append(("get_group_member_list", group_id))
        return []

    async def call_api(self, api=None, **kwargs):
        self.calls.append(("call_api", api))
        return {}

    def __getattr__(self, name):
        async def noop(*a, **k):
            self.calls.append((name, a))
            return {}
        return noop


def install() -> None:
    """把桩装进 sys.modules。必须在 import 适配层之前调用。"""
    nonebot = types.ModuleType("nonebot")
    nonebot.get_bots = lambda: {}
    nonebot.get_driver = lambda: _Driver()
    nonebot.on_message = lambda **kw: _Matcher("message")
    nonebot.on_notice = lambda **kw: _Matcher("notice")
    nonebot.__version__ = "0.0-stub"

    adapter = types.ModuleType("nonebot.adapters")
    onebot = types.ModuleType("nonebot.adapters.onebot")
    v11 = types.ModuleType("nonebot.adapters.onebot.v11")
    for name, obj in (("Bot", Bot), ("Message", Message),
                      ("MessageEvent", MessageEvent),
                      ("PrivateMessageEvent", PrivateMessageEvent),
                      ("GroupMessageEvent", GroupMessageEvent),
                      ("GroupIncreaseNoticeEvent", GroupIncreaseNoticeEvent),
                      ("GroupDecreaseNoticeEvent", GroupDecreaseNoticeEvent),
                      ("MessageSegment", _Segment)):
        setattr(v11, name, obj)
    onebot.v11 = v11
    adapter.onebot = onebot

    plugin = types.ModuleType("nonebot.plugin")
    plugin.PluginMetadata = PluginMetadata

    sys.modules["nonebot"] = nonebot
    sys.modules["nonebot.adapters"] = adapter
    sys.modules["nonebot.adapters.onebot"] = onebot
    sys.modules["nonebot.adapters.onebot.v11"] = v11
    sys.modules["nonebot.plugin"] = plugin


def load_adapter():
    """装桩并**全新**导入适配层，返回模块对象。

    **必须先把模块从 sys.modules 里清掉再导入。**

    第一版用的是 `importlib.reload()`，实测不可靠：如果适配层
    在此之前已经被导入过一次（那时桩还没装，`HAS_NONEBOT` 是 False，
    整个 `if HAS_NONEBOT:` 块没执行），reload 之后 `HANDLERS` 仍然是空的，
    于是 `HANDLERS['message']` 直接 KeyError，或者拿到上一次的残留对象
    （表现成「_handle() takes 2 positional arguments but 3 were given」
    这种完全误导人的报错）。
    """
    install()
    import importlib

    module_name = "aliceqq.adapters.nonebot_plugin_alice"
    for key in list(sys.modules):
        if key == module_name or key.startswith(module_name + "."):
            del sys.modules[key]
    # 父包也一并清掉，确保相对导入重新解析
    for key in ("aliceqq.adapters",):
        sys.modules.pop(key, None)
    return importlib.import_module(module_name)
