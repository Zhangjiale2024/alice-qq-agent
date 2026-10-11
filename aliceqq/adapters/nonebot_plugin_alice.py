"""NoneBot2 适配层（OneBot v11 / NapCat）。

这一层只做四件事：
1. 把 OneBot 事件翻译成 IncomingMessage；
2. 调用 Engine.handle()；
3. 把回复发回去；
4. 跑一个低频的"主动关心"后台任务。

所有业务逻辑都在 aliceqq 里，所以核心能力可以脱离 NoneBot 独立测试。

使用方法见 integrations/nonebot/run.py。
"""
from __future__ import annotations

import asyncio
import os
import time

try:  # pragma: no cover - 依赖可选
    from nonebot import get_bots, get_driver, on_message, on_notice
    from nonebot.adapters.onebot.v11 import (Bot, GroupDecreaseNoticeEvent,
                                             GroupIncreaseNoticeEvent, Message,
                                             MessageEvent, PrivateMessageEvent)
    from nonebot.plugin import PluginMetadata
    HAS_NONEBOT = True
except ImportError:  # pragma: no cover
    HAS_NONEBOT = False

from .. import asr as asr_mod
from .. import knowledge as knowledge_mod
from .. import media_input as media_input_mod
from .. import media as media_mod
from .. import policy as policy_mod
from .. import proactive
from .. import singing as singing_mod
from .. import vision as vision_mod
from ..config import AppConfig
from ..engine import Engine, IncomingMessage

TEXT_FILE_SUFFIXES = (".txt", ".md", ".markdown", ".log", ".csv", ".json")


def _media_cq(outcome) -> str:
    """把 Outcome.media 转成 CQ 码。

    **逻辑在 `aliceqq.media.cq_code_for`，这里只是转发**——
    适配层整块包在 `if HAS_NONEBOT` 里，没装 NoneBot 时根本不存在，
    放在这里就没法离线测试。而绝对路径与逗号转义这两处
    恰恰只在真机上暴露过问题（媒体静默发不出去），必须有测试看着。
    """
    return media_mod.cq_code_for(outcome)


if HAS_NONEBOT:  # pragma: no cover
    __plugin_meta__ = PluginMetadata(
        name="alice-qq-agent",
        description="《蔚蓝档案》爱丽丝角色陪伴 Agent",
        usage="/爱丽丝 切换角色模式；/助手 切换助手模式；/记忆 查看记忆",
    )

    _config = AppConfig.load()
    _engine = Engine(_config)
    # 排障开关：ALICE_DEBUG=1 时打印每条消息的原文与引擎决策。
    # 指令不生效、群聊不回复这类问题，看这个日志比猜快得多。
    # 注意必须从"已合并 .env 的配置"里读，不能只看 os.environ——
    # 否则写在 .env 里的 ALICE_DEBUG 根本读不到。
    _debug_flag = _config.env("ALICE_DEBUG") or os.environ.get("ALICE_DEBUG", "")
    _debug = str(_debug_flag).strip().lower() in ("1", "true", "yes", "on")

    def _log(message: str) -> None:
        print(f"[alice] {message}")

    def _debug_log(message: str) -> None:
        """只在 ALICE_DEBUG=1 时打印。

        专给那些**吞掉异常但业务上确实要继续**的地方用
        （例如发提示消息失败不该中断流程）。这样：
          * 正常运行时日志保持干净；
          * 排障时打开 ALICE_DEBUG，就能看到"某一步其实失败了"。

        为什么需要它：这类静默失败最难查。实测踩过——
        `_was_mentioned()` 解析 at 段失败后直接 return False，
        表现是"群里 @ 了却不回复"，而日志里一个字都没有。
        """
        if _debug:
            print(f"[alice][debug] {message}")

    _driver = get_driver()
    _matcher = on_message(priority=50, block=True)

    # ------------------------------------------------------------------
    # 连接状态日志
    #
    # 为什么加：用户反馈"一段时间没人理就断线"，但**项目里没有任何连接记录**——
    # NoneBot 的日志只在控制台，NapCat 的 logs 目录是空的。
    # 于是断线时既不知道断开时刻、也不知道是 WS 断还是 QQ 登录被踢。
    #
    # 记下"维持了多久"很有诊断价值：
    #   * 每次都很短（几分钟）-> 更像是链路被反复掐断，查心跳/网络
    #   * 每次都很长（几小时）-> 更像是 NapCat 重连太慢，或 QQ 静默离线把连接带走了
    #     （空闲越久越容易中，这正是"没人理就断线"的形态）
    #
    # 两种情况的区别（对策相反，别搞混）：
    #   "连接已断开"          -> WS 断了。NoneBot 每 3 秒重试，但**要等 NapCat**
    #                            按自己的"重连间隔"（当前 30 秒）发起新连接。
    #   "已连接但不再收消息"   -> QQ 登录被**静默**踢下线（NapCat 已知问题，
    #                            见 PR NapNeko/NapCatQQ#2080），WS 看着还是好的，
    #                            但消息永远不来，只能重新登录。
    # ------------------------------------------------------------------
    _connected_at = {"value": 0.0}
    _reconnects = {"value": 0}

    @_driver.on_bot_connect
    async def _on_bot_connect(bot) -> None:
        _connected_at["value"] = time.time()
        n = _reconnects["value"]
        suffix = f"（第 {n} 次重连）" if n else ""
        _log(f"连接已建立：{bot.self_id}{suffix} @ {time.strftime('%H:%M:%S')}")

    @_driver.on_bot_disconnect
    async def _on_bot_disconnect(bot) -> None:
        started = _connected_at["value"]
        lasted = (time.time() - started) if started else 0.0
        _reconnects["value"] += 1
        _log(f"连接已断开：{bot.self_id} @ {time.strftime('%H:%M:%S')}"
             f"（本次维持 {lasted / 60:.1f} 分钟）")
        _log("  NoneBot 会每 3 秒重试，但需要 NapCat 按自己的『重连间隔』"
             "发起新连接（NapCat 当前配置 30 秒）")
        if 0 < lasted < 300:
            _log("  提示：本次只维持了几分钟就断。若反复如此，"
                 "更像是链路/网络问题而不是空闲掉线。")

    # ------------------------------------------------------------------
    # 唱歌（可选功能，默认关闭）
    #
    # 整条流水线要跑 60~90 秒，**绝不能阻塞消息处理**，
    # 否则这 90 秒里机器人对所有消息都不响应。
    # 所以：立刻回一句"我去练一下"，把重活丢到线程里，跑完再发语音。
    # ------------------------------------------------------------------
    _sing_script = _config.env("SINGING_SCRIPT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "GPT-SoVITS-setup", "sing.py")
    _sing_script = os.path.abspath(_sing_script)
    # 跑 sing.py 用哪个解释器：它要 import bili_api / requests，
    # 这些在 GPT-SoVITS 那个环境里备好了，所以优先用那个。
    _sing_python = (_config.env("SINGING_PYTHON")
                    or r"D:\DSHWORK\envs\gptsovits\Scripts\python.exe")
    if not os.path.isfile(_sing_python):
        _sing_python = sys.executable
    _sing_runner = singing_mod.make_runner(_sing_script, _sing_python)
    _sing_cache = singing_mod.SongCache(
        _config.env("SINGING_CACHE_DIR")
        or os.path.join(os.path.dirname(_config.db_file), "songs"))

    async def _handle_singing(bot: Bot, event: MessageEvent) -> bool:
        """处理"唱一首 X"。返回 True 表示这条消息已经被唱歌流程接管。

        为什么在适配层做：需要 bot 对象来发消息，而核心引擎要保持
        可离线测试。判定逻辑本身在 `aliceqq.singing` 里（纯函数，有测试）。
        """
        if not _config.features.singing:
            return False
        text = event.get_plaintext().strip()

        # 先处理"停"：用户说停就真的要停。
        # 截图里出现过"发了停、歌还是发出来了"——因为结果已经在返回路上。
        # request_cancel() 除了杀掉正在跑的子进程，还会递增 token，
        # 让**已跑完但还没发送**的结果被判定为过期而丢弃。
        if text.strip("！!。.~～ ") in ("停", "停止", "别唱了", "不用了", "算了"):
            if singing_mod.busy():
                singing_mod.request_cancel()
                await bot.send(event=event, message="好，爱丽丝不唱了～")
                _log("唱歌已被用户取消")
            else:
                await bot.send(event=event, message="现在没有在唱哦～")
            return True

        # 指令式入口优先：/唱歌 歌名
        command = ""
        for prefix in ("/唱歌", "/唱"):
            if text.startswith(prefix):
                command = text[len(prefix):].strip()
                break
        if command:
            song = command
        else:
            song = singing_mod.detect_song(text) or ""
        if not song:
            return False

        # 权限：只允许 owner 和自己授权的群用。
        # 理由：一次 90 秒吃满 CPU，而且涉及翻唱版权，不该对外开放。
        group_id = str(getattr(event, "group_id", "") or "")
        is_private = not group_id
        if not is_private:
            if not _config.features.group_management:
                return False
            verdict = policy_mod.check_group_access(_engine.storage, _config, group_id)
            if verdict.decision is not policy_mod.Decision.ALLOW:
                return False

        # 同一首歌做过就直接发，不再等 90 秒
        cached = _sing_cache.lookup(song)
        if cached:
            _log(f"唱歌命中缓存：{song} -> {cached}")
            await _send_voice(bot, event, cached)
            return True

        # 新请求**顶替**旧任务：用户改主意了（截图中"唱全速向光不是反乌托邦"
        # 之后又把旧的发出来就是这个问题）。先作废旧结果，再开新的。
        if singing_mod.busy():
            singing_mod.request_cancel()
            await bot.send(event=event,
                           message=f"好，换成《{song}》，爱丽丝重新练～")
        else:
            await bot.send(event=event, message=f"爱丽丝去练一下《{song}》，稍等一会儿～")
        _log(f"开始唱歌：{song}")

        token = singing_mod.current_token()

        async def _worker() -> None:
            try:
                result = await asyncio.to_thread(_sing_runner, song)
            except Exception as exc:      # 兜底：后台任务绝不能把进程搞挂
                _log(f"唱歌任务异常：{type(exc).__name__}: {exc}")
                return
            # 结果过期（用户说了"停"或又点了别的歌）就丢掉，**不要发出去**
            if singing_mod.is_stale(token):
                _log(f"唱歌结果已过期，丢弃：{song}")
                return
            if not result.ok:
                # **如实报告失败**，不假装唱了
                _log(f"唱歌失败：{song} -> {result.error}")
                try:
                    await bot.send(event=event,
                                   message=f"这首爱丽丝没唱好：{result.error}")
                except Exception:
                    pass
                return
            _sing_cache.store(song, result.audio)
            _log(f"唱歌完成：{song}（{result.elapsed:.1f} 秒）-> {result.audio}")
            title = result.title or song
            try:
                await bot.send(event=event, message=f"练好了，这首《{title}》送给你～")
                await _send_voice(bot, event, result.audio)
            except Exception as exc:
                _log(f"唱歌成品发送失败：{exc}")

        asyncio.create_task(_worker())
        return True

    async def _send_voice(bot: Bot, event: MessageEvent, path: str) -> None:
        """用语音段发送音频。构造方式与 _media_cq 保持一致。"""
        if not os.path.isfile(path):
            _log(f"语音文件不存在，已跳过：{path}")
            return
        uri = "file:///" + os.path.abspath(path).replace("\\", "/").lstrip("/")
        uri = uri.replace(",", "%2C").replace("[", "%5B").replace("]", "%5D")
        try:
            await bot.send(event=event, message=Message(f"[CQ:record,file={uri}]"))
        except Exception as exc:
            _log(f"语音发送失败（{path}）：{exc}")

    def _is_command(event: MessageEvent) -> bool:
        text = event.get_plaintext().strip()
        return text.startswith("/") or text.startswith(_config.bot.command_prefix)

    def _was_mentioned(event: MessageEvent) -> bool:
        if isinstance(event, PrivateMessageEvent):
            return True
        try:
            for segment in event.get_message():
                if segment.type == "at" and str(segment.data.get("qq")) == str(event.self_id):
                    return True
        except Exception as exc:
            # **这里吞掉异常是有风险的**：返回 False 意味着"没被 @ 到"，
            # 于是这条群消息会被判为不需要回复。一旦 at 段结构变化
            # （或 event.get_message() 抛错），表现就是"群里 @ 了却不回复"，
            # 而日志里一个字都没有，极难定位。所以至少要留下线索。
            _debug_log(f"_was_mentioned 解析失败，按『未提及』处理："
                       f"{type(exc).__name__}: {exc}")
        return False

    def _replied_to_bot(event: MessageEvent) -> bool:
        reply = getattr(event, "reply", None)
        sender = getattr(reply, "sender", None) if reply is not None else None
        return bool(sender and getattr(sender, "user_id", None) == getattr(event, "self_id", None))

    def _extract_images(event: MessageEvent) -> list:
        """转发到 media_input.segments_to_images。

        解析逻辑**不放在这一层**：适配层整个包在 `if HAS_NONEBOT:` 里，
        没有 NoneBot 的环境（包括 CI）执行不到，于是那层逻辑一行都测不到。
        详见 media_input.py 的模块说明。
        """
        try:
            return media_input_mod.segments_to_images(event.get_message(), _debug_log)
        except Exception as exc:
            _debug_log(f"_extract_images 失败，本轮按『没有图片』处理：{exc}")
            return []

    def _extract_audios(event: MessageEvent) -> list:
        """转发到 media_input.segments_to_audios。理由同 _extract_images。"""
        try:
            return media_input_mod.segments_to_audios(event.get_message(), _debug_log)
        except Exception as exc:
            _debug_log(f"_extract_audios 失败，本轮按『没有语音』处理：{exc}")
            return []

    def _extract_videos(event: MessageEvent) -> list:
        """转发到 media_input.segments_to_videos。理由同 _extract_images。

        视频段以前也是完全丢弃的——项目根本不认 `video`。
        """
        try:
            return media_input_mod.segments_to_videos(event.get_message(), _debug_log)
        except Exception as exc:
            _debug_log(f"_extract_videos 失败，本轮按『没有视频』处理：{exc}")
            return []

    async def _record_admin(bot: Bot, event: MessageEvent, group_id: str) -> None:
        """把群主/管理员记进群配置，供权限判断使用。

        只记录 role 为 owner/admin 的成员；查询失败静默忽略，
        因为这属于"锦上添花"的能力，不能影响正常回复。
        """
        try:
            info = await bot.get_group_member_info(group_id=int(group_id),
                                                   user_id=int(event.user_id), no_cache=True)
        except Exception:
            return
        role = str(info.get("role", ""))
        if role not in ("owner", "admin"):
            return
        record = _engine.storage.get_group(group_id)
        extra = dict(record.extra) if record else {}
        admins = set(str(a) for a in extra.get("admins", []))
        if str(event.user_id) in admins:
            return
        admins.add(str(event.user_id))
        extra["admins"] = sorted(admins)
        _engine.storage.upsert_group(group_id, extra=extra)
        _log(f"已记录群 {group_id} 的管理员：{event.user_id}（{role}）")

    async def _maybe_capture_file(bot: Bot, event: MessageEvent) -> None:
        """把所有者发来的文本文件暂存起来，供 /资料上传 使用。

        OneBot 的文件段通常是 file:/// 本地路径或 http 链接，
        这里只处理能直接读到的本地文本文件；读不到就静默跳过，
        用户改用「/资料上传 标题｜正文」即可。
        """
        try:
            segments = list(event.get_message())
        except Exception:
            return
        for segment in segments:
            if getattr(segment, "type", "") != "file":
                continue
            data = getattr(segment, "data", {}) or {}
            name = str(data.get("file") or data.get("name") or "上传文件")
            if not name.lower().endswith(TEXT_FILE_SUFFIXES):
                continue
            target = str(data.get("url") or data.get("file") or "")
            if target.startswith("file:///"):
                target = target[8:]
            if not target or not os.path.isfile(target):
                continue
            try:
                with open(target, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            if not content.strip():
                continue
            knowledge_mod.record_pending(_config.knowledge_path, str(event.user_id), name, content)
            _log(f"已暂存文件资料：{name}（{len(content)} 字），发送 /资料上传 即可入库")


    async def _maybe_capture_media(bot: Bot, event: MessageEvent) -> None:
        """所有者执行 /素材上传 后，抓取其下一条消息里的图片/表情/语音进素材库（V4）。

        判定逻辑（哪些段能入库、怎么读字节）都放在 aliceqq.media 里，
        因为适配层在没装 NoneBot 的环境测不到——这些逻辑正是在这里悄悄坏过：

        1. 只认本地路径，而 NapCat 给照片的是 http URL -> 照片全被跳过；
        2. 只认 image/record，而 QQ 表情是 face 段   -> 表情完全不进分支；
        3. 失败时不给任何反馈                       -> 用户发完图什么也看不到。
        """
        try:
            pending = _engine.storage.kv_get(f"media_pending:{event.user_id}")
        except Exception:
            return
        if not pending:
            return
        tags = str((pending or {}).get("tags") or "")
        desc = str((pending or {}).get("desc") or "")
        try:
            segments = list(event.get_message())
        except Exception:
            return

        async def tell(text: str) -> None:
            try:
                await bot.send(event=event, message=text)
            except Exception:
                pass

        for segment in segments:
            seg_type = str(getattr(segment, "type", "") or "")
            data = dict(getattr(segment, "data", {}) or {})
            kind, source, reason = media_mod.classify_segment(seg_type, data)
            if reason:
                if reason == media_mod.FACE_REASON:
                    # 内置表情拿不到图片数据，给可执行的两个替代方案
                    _log(f"收到 face 段（{data.get('faceText') or data.get('faceIndex')}）")
                    await tell(
                        "这个是 QQ 内置表情，爱丽丝拿不到它的图片文件，存不进素材库。\n\n"
                        "两个办法：\n"
                        "  1) 把表情**转发**给爱丽丝（转发过来通常会变成图片）；\n"
                        "  2) 截图后当图片发过来。\n\n"
                        "发完爱丽丝就收进素材库。")
                    return
                continue

            blob = await asyncio.to_thread(media_mod.read_media_bytes, source)
            if not blob:
                _log(f"媒体读取失败：{source[:90]}")
                await tell(f"这张素材爱丽丝没能取到（{source[:60]}…）。"
                           "可以重新发一次，或者换成截图。")
                continue

            try:
                library = _engine.media_library
                if library is None:
                    library = media_mod.MediaLibrary(_config.media.path)
                    library.load()
                item = media_mod.store_media(
                    library, source, kind, blob, tags, desc,
                    hint_name=media_mod.segment_hint_name(data))
                _engine.storage.kv_set(f"media_pending:{event.user_id}", None)
                _log(f"素材已入库：{item.id}（{desc or kind}｜标签：{tags}）")
                await tell(
                    f"已入库：{item.id}\n"
                    f"  类型：{'语音' if kind == 'voice' else '图片/表情'}\n"
                    f"  标签：{tags or '（无）'}\n"
                    f"  说明：{desc or '（无）'}\n"
                    f"  大小：{len(blob) // 1024} KB\n\n"
                    "之后话题命中这些标签时，爱丽丝就会用它。")
            except Exception as exc:
                _log(f"素材入库失败：{type(exc).__name__}: {exc}")
                await tell(f"入库失败：{exc}")
            return

    async def _proactive_loop() -> None:
        """低频主动关心。失败绝不抛出，避免拖垮整个机器人进程。"""
        interval = max(60, int(_config.proactive.check_interval_seconds))
        while True:
            try:
                await asyncio.sleep(interval)
                tasks = proactive.plan(_config, _engine.storage)
                if not tasks:
                    continue
                bots = list(get_bots().values())
                if not bots:
                    continue
                bot = bots[0]
                for task in tasks:
                    try:
                        await bot.send_private_msg(user_id=int(task.user_id), message=task.text)
                        proactive.mark_sent(_engine.storage, task.user_id)
                        _engine.storage.append_message(task.user_id, "assistant", task.text, "")
                        print(f"[alice] 主动关心 -> {task.user_id}（{task.reason}）")
                    except Exception as exc:
                        # 失败要记下来并退避，否则每一轮都会重撞同一个错误。
                        # 实测：10001/10002 这类示例号会让 QQ 一直回
                        # retcode=1200 无法获取用户信息，日志被刷满。
                        proactive.mark_failed(_engine.storage, task.user_id)
                        print(f"[alice] 主动关心发送失败 {task.user_id}（已退避 24 小时）: {exc}")

                # V4：群聊低频主动话题。门槛很高（群主显式开启 + 群里足够安静 + 每日上限），
                # 且发送失败不影响私聊那部分。
                group_tasks = proactive.plan_group_topics(_config, _engine.storage)
                for task in group_tasks:
                    try:
                        await bot.send_group_msg(group_id=int(task.group_id), message=task.text)
                        proactive.mark_group_sent(_engine.storage, task.group_id)
                        _engine.storage.kv_set(f"bot_last_reply:{task.group_id}", time.time())
                        print(f"[alice] 群主动话题 -> {task.group_id}（{task.reason}）")
                    except Exception as exc:
                        print(f"[alice] 群主动话题发送失败 {task.group_id}: {exc}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 兜底：调度器绝不能把进程搞挂
                print(f"[alice] 主动关心任务异常（已忽略）: {exc}")

    @_driver.on_startup
    async def _startup() -> None:
        mode = "真实模型" if _config.llm.is_real else "离线 mock"
        print(f"[alice] 已启动")
        print(f"[alice]   模型     = {_config.llm.model}（{mode}）@ {_config.llm.base_url}")
        print(f"[alice]   数据库   = {_config.db_file}")
        print(f"[alice]   提示词   = {_config.prompts_path}")
        print(f"[alice]   角色资料 = {_config.knowledge_path}")
        if _config.search.enabled:
            print(f"[alice]   联网检索 = { _config.search.provider }"
                  f"（自动检索 {'开' if _config.search.auto_search else '关'}）")
            if "bilibili" in (_config.search.chain or []):
                try:
                    from ..search import BilibiliProvider
                    status = BilibiliProvider(
                        sessdata=_config.search.bilibili_sessdata or "").verify_login()
                    if status["configured"]:
                        mark = "有效" if status["valid"] else "无效"
                    else:
                        mark = "未配置"
                    who = f"（{status['user']}）" if status.get("user") else ""
                    print(f"[alice]   B站登录态 = {mark}{who}｜{status['note']}")
                except Exception as exc:
                    print(f"[alice]   B站登录态校验跳过：{type(exc).__name__}")
        else:
            print("[alice]   联网检索 = 关闭")
        print(f"[alice]   群访问   = {_config.group.access}；触发 = {', '.join(_config.group.triggers)}")
        print(f"[alice]   群关键词 = {', '.join(_config.group.keywords)}")
        print("   手动切换 = /爱丽丝（或 /alice）与 /助手")
        if "relevant_topic" in _config.group.triggers:
            print(f"[alice]   话题延续 = "
                  f"{'仅同一人连续说' if _config.group.relevant_topic_same_user_only else '窗口内任何人'}，"
                  f"窗口 {_config.group.continuity_seconds}s")
        _log(f"   OWNER_IDS= {_config.bot.owner_ids or '（未配置）'}"
             + ("" if _config.bot.owner_ids else "  <- 未配置时，第一个执行 /授权 的人会被认领为 owner"))
        _log(f"   调试日志 = {'开启' if _debug else '关闭（设 ALICE_DEBUG=1 可开启）'}")
        if _config.proactive.enabled and _config.features.proactive_private:
            print(f"[alice]   主动关心 = 开启（{_config.proactive.frequency}，"
                  f"闲置 {_config.proactive.min_idle_minutes} 分钟起）")
            asyncio.create_task(_proactive_loop())
        else:
            print("[alice]   主动关心 = 关闭")
        # V4：多媒体真实状态（能力不可用时必须让运维一眼看到）
        media = _config.media
        if media.enabled:
            sticker_count = 0
            try:
                from ..media import MediaLibrary
                sticker_count = len(MediaLibrary(media.path).load())
            except Exception:
                pass
            print(f"[alice]   素材库   = {media.path}（{sticker_count} 项）"
                  f"｜表情包 {'开' if media.sticker_enabled else '关'}")
            print(f"[alice]   图片生成 = "
                  f"{'已配置 ' + media.image_gen_model if media.image_gen_enabled else '未配置（无可用素材时无法生成）'}")
            try:
                from ..generation import describe_voice_pipeline
                print(f"[alice]   {describe_voice_pipeline(media)}")
            except Exception as exc:
                print(f"[alice]   语音合成 = 状态读取失败：{type(exc).__name__}")
            # 网络出口：系统代理反复掉线时，这一层会自动改走直连。
            # 启动时把探测结果打出来，出问题好定位是不是网络层的锅。
            try:
                import socket
                import urllib.request
                from .. import net as net_mod
                proxies = urllib.request.getproxies()
                probe = socket.socket()
                probe.settimeout(1)
                alive = True
                proxy_url = proxies.get("https") or proxies.get("http") or ""
                if proxy_url:
                    host_port = proxy_url.split("//")[-1].rstrip("/")
                    host, _, port = host_port.partition(":")
                    try:
                        probe.connect((host or "127.0.0.1", int(port or 80)))
                    except Exception:
                        alive = False
                probe.close()
                if not proxy_url:
                    state = "未配置代理，直连"
                elif alive:
                    state = f"系统代理 {proxy_url} 可用"
                else:
                    state = f"系统代理 {proxy_url} 连不上，自动改走直连"
                print(f"[alice]   网络     = {state}｜出口顺序 {' → '.join(net_mod.route_names())}")
            except Exception as exc:
                print(f"[alice]   网络     = 状态读取失败：{type(exc).__name__}")
        else:
            print("[alice]   多媒体   = 关闭")
        for name in ("alice_system.md", "assistant_system.md"):
            if not (_config.prompts_path / name).is_file():
                print(f"[alice] 警告：缺少提示词模板 {_config.prompts_path / name}")

    @_matcher.handle()
    async def _handle(bot: Bot, event: MessageEvent) -> None:
        # 忽略机器人自己发的消息，避免自问自答
        if str(event.user_id) == str(event.self_id):
            return

        text = event.get_plaintext().strip()
        is_private = isinstance(event, PrivateMessageEvent)
        group_id = "" if is_private else str(getattr(event, "group_id", ""))

        # 用户这次发的图片要带进消息里，否则模型看不到图。
        #
        # **这里以前是完全丢弃的**：只取 `get_plaintext()`，
        # 图片段既不进 text 也不进别的字段，所以"理解图片"这件事
        # 在真实 QQ 上根本无从谈起。文本为空但有图的情况也很常见
        # （用户直接甩一张图不说话），那种消息以前会被当成空消息跳过。
        images = _extract_images(event)
        audios = _extract_audios(event)
        videos = _extract_videos(event)

        incoming = IncomingMessage(
            user_id=str(event.user_id),
            text=text,
            is_private=is_private,
            group_id=group_id,
            user_name=(event.sender.card or event.sender.nickname or "") if event.sender else "",
            mentioned=_was_mentioned(event),
            replied_to_bot=_replied_to_bot(event),
            is_command=_is_command(event),
            message_id=str(getattr(event, "message_id", "")),
            timestamp=float(getattr(event, "time", time.time())),
            images=images,
            audios=audios,
            videos=videos,
        )

        if _debug:
            scope = "私聊" if is_private else f"群{group_id}"
            _log(f"收到[{scope}] user={incoming.user_id} name={incoming.user_name!r} "
                 f"mentioned={incoming.mentioned} command={incoming.is_command} text={text!r}")

        # 唱歌意图优先接管：它要跑 60~90 秒，交给后台线程，
        # 不能等引擎那条"正常回复"路径（否则这轮消息会卡住）。
        # 未命中就返回 False，流程照常往下走。
        if await _handle_singing(bot, event):
            return

        # 记录群管理角色：让群主/管理员也能执行 /授权、/群规则
        if not is_private and _config.features.group_management:
            await _record_admin(bot, event, group_id)

        # 所有者发来的文本文件 -> 暂存，等待 /资料上传
        if _config.features.knowledge_upload:
            await _maybe_capture_file(bot, event)
        # V4：所有者 /素材上传 之后的那条图片消息，抓进素材库
        await _maybe_capture_media(bot, event)

        try:
            outcome = _engine.handle(incoming)
        except Exception as exc:
            print(f"[alice] 处理消息异常: {type(exc).__name__}: {exc}")
            return

        if _debug:
            _log(f"决策[{outcome.decision}] mode={outcome.mode} reason={outcome.reason!r}"
                 + (f" 护栏={outcome.debug}" if outcome.debug else ""))

        if not outcome.reply:
            # 群聊里"没被叫到"是正常沉默，不必刷日志；其它原因值得记一笔
            if outcome.decision not in ("silent",):
                _log(f"未回复（{outcome.decision}）：{outcome.reason}")
            return

        # 分条发送：小鲸鱼把一句话拆成几条，逐条发出并留出间隔，
        # 这样才像真人打字，而不是一条长消息。
        parts = outcome.outgoing
        if not parts:
            return
        media_cq = _media_cq(outcome)      # 媒体只挂在最后一条上
        if _debug:
            _log(f"回复（{len(parts)} 条）：{parts!r}")

        for index, part in enumerate(parts):
            body = Message(part)
            if index == 0 and not is_private:
                body = Message(f"[CQ:at,qq={incoming.user_id}] ") + body
            if media_cq and index == len(parts) - 1:
                body = body + Message(" " + media_cq)
            try:
                await bot.send(event=event, message=body)
            except Exception as exc:
                print(f"[alice] 发送失败（第 {index + 1}/{len(parts)} 条）：{exc}")
                return
            if index < len(parts) - 1:
                # 间隔随消息长短浮动，太规律反而不像人
                await asyncio.sleep(min(2.2, 0.5 + len(part) * 0.06))
        for item in outcome.media:
            _record_media_sent(item)
        # V5 群管理：如果这一轮决定要执行管理动作，交给适配层真正调用 QQ 接口
        await _run_moderation_action(bot, event, outcome)

    # ------------------------------------------------------------------
    # V5：群管理动作执行（撤回 / 禁言 / 移出）
    # ------------------------------------------------------------------
    # 为什么放在适配层：核心引擎必须能脱离 NoneBot 离线测试，所以它只产出
    # "动作请求"（outcome.debug["action"]），由这里翻译成 OneBot 调用。
    # **执行结果必须回填**（成功/失败都回），因为 V5 明确要求
    # "不得伪造已经撤回、禁言或踢出成功"——失败时要把真实原因说给管理员听。
    async def _run_moderation_action(bot: Bot, event, outcome) -> None:
        request = (outcome.debug or {}).get("action")
        if not request or not _config.features.group_management:
            return
        action = str(request.get("action") or "")
        group_id = int(request.get("group_id") or getattr(event, "group_id", 0) or 0)
        target_id = int(request.get("target_id") or 0)
        payload = request.get("payload") or {}
        operator_id = str(request.get("operator_id") or "")
        reason = str(request.get("reason") or "")
        ok, detail = False, "未知原因"
        try:
            if action == "recall":
                message_id = payload.get("message_id") or getattr(event, "message_id", "")
                if not message_id:
                    ok, detail = False, "缺少消息 ID，无法撤回"
                else:
                    await bot.delete_msg(message_id=int(message_id))
                    ok, detail = True, ""
            elif action == "mute":
                seconds = int(payload.get("seconds") or 600)
                await bot.set_group_ban(group_id=group_id, user_id=target_id,
                                        duration=seconds)
                ok, detail = True, ""
            elif action == "kick":
                await bot.set_group_kick(group_id=group_id, user_id=target_id,
                                         reject_add_request=bool(payload.get("blacklist")))
                ok, detail = True, ""
            else:
                ok, detail = False, f"适配层不认识的动作 {action}"
        except Exception as exc:
            # 失败要如实上报，不能假装成功。常见原因：机器人不是管理员、
            # 目标比自己权限高、QQ 侧限流。
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        try:
            reply = _engine.record_action_result(
                str(group_id), action, ok, target_id=str(target_id),
                operator_id=operator_id, reason=reason, detail=detail)
            await bot.send(event=event, message=Message(reply))
        except Exception as exc:
            print(f"[alice] 管理动作结果回执发送失败：{exc}")

    # ------------------------------------------------------------------
    # V5：群成员变动 -> 入群欢迎 / 退群告别
    # ------------------------------------------------------------------
    # 为什么用 on_notice 而不是轮询成员列表：QQ 的成员变动只有通知事件，
    # 轮询既费流量又必然漏掉短时间内的进出。
    # 业务逻辑全在 Engine.member_event 里（可离线测试），这一层只做事件翻译。
    if HAS_NONEBOT:
        _notice_matcher = on_notice(priority=50, block=False)

        # **必须用 Rule 在匹配器层拦掉**，而不是在 handler 里 isinstance 判断。
        # 实测（2026-10-10 日志）：`notice.notify.input_status`（"对方正在输入…"）
        # 每次都被这个 handler 接住、跑一遍依赖注入、再静默 return——
        # 群里有人打字就刷一次日志，纯属噪音。
        # 放在 Rule 里的话，类型不对的事件连 handler 都不会进。
        def _is_member_change(event) -> bool:
            return isinstance(event, (GroupIncreaseNoticeEvent, GroupDecreaseNoticeEvent))

        @_notice_matcher.handle()
        async def _handle_member_change(bot: Bot, event) -> None:
            if not _is_member_change(event):
                return  # 双保险：Rule 已拦，这里再确认一次
            if not _config.features.group_management:
                return
            if isinstance(event, GroupIncreaseNoticeEvent):
                action = "approve"
            else:
                # GroupDecreaseNoticeEvent：
                # sub_type: leave（主动退群）| kick（被管理员移出）| kick_me（机器人自己被踢）
                action = str(getattr(event, "sub_type", "") or "leave")
            group_id = str(getattr(event, "group_id", "") or "")
            user_id = str(getattr(event, "user_id", "") or "")
            if not group_id or not user_id:
                return
            # 机器人自己被踢：不发任何消息（也发不出去），但要留痕
            if str(user_id) == str(getattr(event, "self_id", "")):
                _log(f"机器人已被移出群 {group_id}")
                return
            name = ""
            try:
                info = await bot.get_group_member_info(group_id=int(group_id),
                                                       user_id=int(user_id), no_cache=True)
                name = str(info.get("card") or info.get("nickname") or "")
            except Exception:
                # 取不到名字不是失败：欢迎语里有 {name} 时用兜底称呼
                pass
            try:
                outcome = _engine.member_event(group_id=group_id, user_id=user_id,
                                               user_name=name, action=action)
            except Exception as exc:
                print(f"[alice] 成员变动处理异常（已忽略）: {type(exc).__name__}: {exc}")
                return
            if not outcome.reply:
                if _debug:
                    _log(f"成员变动[{action}] 不发言：{outcome.reason}")
                return
            if _debug:
                _log(f"成员变动[{action}] 发送：{outcome.reply!r}")
            try:
                await bot.send(event=event, message=Message(outcome.reply))
            except Exception as exc:
                print(f"[alice] 成员变动消息发送失败：{exc}")
