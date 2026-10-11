"""决策引擎：把配置、存储、记忆、提示词、模型串成一条完整的回复流水线。

这是 V1 的核心，且完全不依赖 NoneBot —— adapters/nonebot_plugin_alice.py 只是把
OneBot 事件翻译成 IncomingMessage 再调用这里，CLI 模拟器同理。
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import honesty as honesty_mod
from . import knowledge as knowledge_mod
from . import memory as memory_mod
from . import moderation as moderation_mod
from . import prompt as prompt_mod
from . import asr as asr_mod
from . import tools as tools_mod
from . import ffmpeg as ffmpeg_mod
from . import video as video_mod
from . import vision as vision_mod
from .config import AppConfig, GroupSettings
from . import db as db_mod
from .db import Storage
from .llm import BaseLLM, LLMError, Message, Reply, build_llm
from .academic import (AcademicOutcome, AcademicService, build_academic_service,
                      format_academic, is_academic_query)
from .asr import build_client as build_asr_client
from . import characters as characters_mod
from . import style as style_mod
from . import emotion as emotion_mod
from .search import (SearchOutcome, SearchService, build_service,
                     has_topic as search_mod_has_topic, needs_search)
from .search import mine_topic as search_mod_mine_topic
from .policy import (Decision, Verdict, check_asr_quota, check_feature_quota, check_group_access, check_quota,
                     check_rate_limit, consume_asr_quota, consume_feature_quota, consume_quota,
                     consume_rate_limit,
                     feature_quota_text, permission_context_text, quota_status_text)
from .triggers import TriggerContext, evaluate as evaluate_trigger, strip_mention

MODE_ALICE = characters_mod.MODE_ALICE
MODE_ASSISTANT = characters_mod.MODE_ASSISTANT
MODE_WHALE = characters_mod.MODE_WHALE

# 工具循环的默认最大轮数。
#
# 模型可能一直调工具不出结论（"再查一次""再记一条"），所以必须有上限。
# 撞上限时要**如实说明**，不能假装回答完了——这是和项目其它护栏
# 一致的取舍：宁可说"没查清楚"，也不编一个结论。
MAX_TOOL_ROUNDS = 3

# 指令别名：只做精确的整串替换，避免"包含即替换"伤到别的指令。
# 打字习惯差异（全角、大小写、英文名）都收敛到同一条指令上。
COMMAND_ALIASES = {
    "/alice": "/爱丽丝",
    "/アリス": "/爱丽丝",
    "/アシスタント": "/助手",
}


def canonical_command(command: str) -> str:
    if command in COMMAND_ALIASES:
        return COMMAND_ALIASES[command]
    # 已经是标准形式就原样返回，避免把 /alice2 之类的旧指令也改写掉
    return command


def _command_lookup():
    """构建"小写形式 -> 规范形式"的映射，实现大小写不敏感。"""
    lookup = {"/爱丽丝": "/爱丽丝", "/助手": "/助手"}
    for alias, canonical in COMMAND_ALIASES.items():
        lookup[alias.lower()] = canonical
    return lookup


_COMMAND_LOOKUP = _command_lookup()

ALICE_SYSTEM_TEMPLATE = "alice_system.md"
ASSISTANT_SYSTEM_TEMPLATE = "assistant_system.md"


@dataclass
class IncomingMessage:
    user_id: str
    text: str
    is_private: bool = True
    group_id: str = ""
    user_name: str = ""
    mentioned: bool = False
    replied_to_bot: bool = False
    is_command: Optional[bool] = None      # None 表示自动判定
    force_reply: bool = False
    message_id: str = ""
    timestamp: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)
    # 用户这次发来的图片。
    #
    # **这是"能发图"到"看得懂图"的关键一步。** 在此之前，适配层把图片
    # 归为 sticker 存进素材库就结束了，消息里根本没有"这张图"这个信息，
    # 模型只能靠文件名和用户的话猜图里有什么。
    #
    # 类型写 Any 是为了让 engine 不依赖 vision 模块——核心包要保持
    # 零第三方依赖，也要保持能独立测试。实际放的是 vision.ImageRef。
    images: List[Any] = field(default_factory=list)
    # 用户这次发来的语音。
    #
    # 和图片是同一个道理，但**这里以前更彻底地被丢掉了**：
    # ASR 能力早就有（`asr.py`），却只用在"转写 B 站视频音轨"那条链路上，
    # 用户直接发来的语音消息一次都没有被识别过。
    # 同样写 Any 以免 engine 依赖 asr 模块的具体类型（实际是 asr.AudioRef）。
    audios: List[Any] = field(default_factory=list)
    # 用户这次发来的视频。
    #
    # 视频走的是**已有的两条链路**：抽帧交给视觉模型，抽音轨交给 ASR。
    # 所以它不需要新的额度类型——vision 和 asr_voice 的门禁自动生效。
    videos: List[Any] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def has_audios(self) -> bool:
        return bool(self.audios)

    @property
    def has_videos(self) -> bool:
        return bool(self.videos)

    @property
    def has_media_input(self) -> bool:
        return bool(self.images or self.audios or self.videos)

    @property
    def scene_id(self) -> str:
        return "private" if self.is_private else f"group:{self.group_id}"


@dataclass
class MediaAttachment:
    """要随回复发送的一个媒体。

    origin 必须如实记录：V4 要求"不要把 AI 生成图片说成官方素材"。
    """
    kind: str                    # sticker | image | voice
    path: str                    # 本地文件绝对路径
    item_id: str = ""
    origin: str = "builtin"      # builtin | generated
    desc: str = ""

    @property
    def is_generated(self) -> bool:
        return self.origin == "generated"


@dataclass
class Outcome:
    """一次处理的结果。reply=None 表示不回复。"""
    reply: Optional[str] = None
    decision: str = "silent"
    reason: str = ""
    mode: str = MODE_ALICE
    record_only: bool = False
    memory_saved: List[str] = field(default_factory=list)
    memory_explicit: List[str] = field(default_factory=list)
    memory_saved_this_turn: List[str] = field(default_factory=list)
    memory_rejected: List[str] = field(default_factory=list)
    memory_pending_review: bool = False
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    search: Optional[SearchOutcome] = None
    search_query: str = ""
    academic: Optional[AcademicOutcome] = None
    # V4：随回复一起发送的媒体（表情包/图片/语音）。每次最多一个。
    media: List["MediaAttachment"] = field(default_factory=list)
    # 分条发送：像真人那样把一句话拆成几条发出（小鲸鱼的核心习惯）。
    # 为空时按单条 reply 处理，保持向后兼容。
    replies: List[str] = field(default_factory=list)

    @property
    def outgoing(self) -> List[str]:
        """真正要发出的消息列表。"""
        if self.replies:
            return list(self.replies)
        return [self.reply] if self.reply else []
    prompt_system: str = ""
    messages: List[Message] = field(default_factory=list)
    llm_usage: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_send(self) -> bool:
        return bool(self.reply)


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


class Engine:
    def __init__(self, config: Optional[AppConfig] = None, storage: Optional[Storage] = None,
                 llm: Optional[BaseLLM] = None, *, rng: Optional[random.Random] = None):
        self.config = config or AppConfig.load()
        self.storage = storage or Storage(self.config.db_file)
        self.llm = llm or build_llm(self.config.llm)
        self.rng = rng or random.Random()
        self._templates: Dict[str, str] = {}
        self._knowledge_cache: Dict[str, str] = {}
        # V4：素材库与多媒体生成（都可在未配置时优雅降级）
        try:
            from .media import build_library as build_media_library
            self.media_library = build_media_library(self.config)
        except Exception:
            self.media_library = None
        try:
            from .generation import build_image_client, build_tts
            self.image_client = build_image_client(self.config)
            # 用 build_tts 而不是 build_tts_client：
            # 前者会自动套上音色转换层（配置了 TTS_VC_URL 时）。
            self.tts_client = build_tts(self.config)
        except Exception:
            self.image_client = None
            self.tts_client = None

        # 语音识别（可选）：只有配置了 key 才构建。
        # 它是"B 站没给字幕时"的兜底，用来把视频音频转成文字。
        try:
            self.asr_client = build_asr_client(self.config)
        except Exception:
            self.asr_client = None

        # 图片理解（可选）。没配 to 就是 None，此时引擎会**明确告诉模型
        # "这次看不到图"**，而不是让它根据用户的描述瞎猜图里有什么。
        try:
            self.vision_client = vision_mod.build_client(self.config.vision)
        except Exception:
            self.vision_client = None
        # 本轮图片理解的结果，供 build_context 注入提示词用。
        # 用"每轮清一次"的实例属性，避免上一轮的图片描述串到这一轮
        # （和 _pending_search 是同一个套路，那里也踩过串轮的坑）。
        self._pending_vision: Optional[vision_mod.VisionResult] = None
        # 本轮语音转写的结果。同样每轮清一次：
        # 把 A 的语音转写带进 B 的对话，等于泄漏他说过的话。
        self._pending_transcript: Optional[asr_mod.Transcription] = None
        # 本轮视频解析结果。视频会同时用到视觉和语音两条链路，
        # 所以这里要分别存画面结果和音轨转写。
        self._pending_video = None
        self._pending_video_frames = None
        self._pending_video_transcript = ""
        self._pending_video_transcript_note = ""

        # V2：联网检索。构建失败也没关系——工具会如实显示为不可用，
        # 引擎不会假装搜过（诚实护栏负责兜底）。
        try:
            self.search_service: Optional[SearchService] = build_service(
                self.config, asr_client=self.asr_client, asr_guard=self._asr_guard)
        except Exception:
            self.search_service = None
        self._pending_search: Optional[SearchOutcome] = None
        self._pending_search_owner: Optional[str] = None
        # 当前这一轮的群级设置（V3）；私聊为 None
        self._current_settings: Optional[GroupSettings] = None
        # 学术检索（CrossRef / Semantic Scholar / arXiv）
        try:
            self.academic_service: Optional[AcademicService] = build_academic_service(self.config)
        except Exception:
            self.academic_service = None
        self._pending_academic: Optional[AcademicOutcome] = None

    # ------------------------------------------------------------------
    # 资源
    # ------------------------------------------------------------------
    def template(self, name: str) -> str:
        if name not in self._templates:
            path = self.config.prompts_path / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少提示词模板：{path}")
            self._templates[name] = path.read_text(encoding="utf-8")
        return self._templates[name]

    def knowledge_base(self) -> "knowledge_mod.KnowledgeBase":
        """按优先级加载知识库（带缓存）。"""
        if self._knowledge_cache.get("__base__") is None:
            self._knowledge_cache["__base__"] = knowledge_mod.load(
                self.config.knowledge_path,
                max_chars=self.config.bot.knowledge_max_chars,
                max_file_chars=self.config.bot.knowledge_max_file_chars)
        return self._knowledge_cache["__base__"]

    def knowledge_block(self) -> str:
        """全局角色资料 / 世界观 / 所有者上传，渲染成提示词区块。"""
        return knowledge_mod.render(self.knowledge_base())

    def reload_knowledge(self) -> "knowledge_mod.KnowledgeBase":
        self._knowledge_cache.clear()
        self._templates.clear()
        return self.knowledge_base()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def handle(self, message: IncomingMessage) -> Outcome:
        now = message.timestamp or time.time()
        text = (message.text or "").strip()
        if len(text) > self.config.bot.max_input_chars:
            text = text[: self.config.bot.max_input_chars]
        mode = self.current_mode(message)
        outcome = Outcome(mode=mode, decision="silent", reason="")
        self._pending_search = None  # 清掉上一轮的检索结果，避免串轮
        self._pending_academic = None
        self._pending_search_owner = None
        # 图片描述同样要清。**这个坑在检索那边踩过一次**：
        # "群 A 的检索结果渲染到群 B"，因为 _pending_search_owner 是实例属性，
        # 忘了清就会把上一轮（甚至上一个人）的结果带进来。图片更敏感——
        # 把 A 发的图描述给 B 看，等于泄漏内容。
        self._pending_vision = None
        self._current_settings = None

        # 0) 太短/空消息
        #
        # **带媒体的消息不能因为文字为空就丢掉。** 用户直接甩一张图、
        # 或者发一条语音不说话，都是很常见的用法；以前的判断只看 text，
        # 于是这类消息在真实 QQ 上被当成"空消息"直接 return，
        # 后面的图片理解和语音识别根本没有机会执行
        # （实测：语音消息 reason='空消息'，_analyze_audio 一次都没被调用）。
        if not text and not message.has_media_input:
            outcome.reason = "空消息"
            return outcome
        if not text:
            outcome.debug["empty_text_with_media"] = True

        # 1) 指令优先。
        # 指令是零成本的本地操作，必须排在"群聊触发"和"限流"之前。
        # 早期实现把它们排在后头，导致：
        #   - 群里被回满限流后 /爱丽丝 完全没反应（用户看到的现象就是"切不回爱丽丝"）；
        #   - 群消息没命中触发时，指令被一并静默丢弃。
        if self.is_command(text):
            self.record(message, now)
            command_outcome = self.try_command(message, mode, now)
            if command_outcome is not None:
                if command_outcome.reply:
                    self.storage.append_message(message.user_id, "assistant",
                                                command_outcome.reply, message.group_id, now)
                return command_outcome
            # 有指令前缀但不是已知指令：直接提示，别丢给模型当闲聊
            return Outcome(reply="爱丽丝不认识这条指令。发送 /帮助 可以看到目前的指令列表。",
                           decision="command", reason="未知指令", mode=mode)

        # 2) 群聊访问控制
        group_record = None
        if not message.is_private:
            access = check_group_access(self.storage, self.config, message.group_id)
            if not access.allowed:
                outcome.decision = "denied"
                outcome.reason = access.reason
                return outcome
            group_record = self.storage.get_group(message.group_id)
            # 群级设置：本轮的触发/额度/字数等覆盖都从这里取
            self._current_settings = GroupSettings.from_extra(
                group_record.extra if group_record else {})
            # V5 群管理：审核判定要排在触发判断**之前**——
            # 广告消息往往不会命中触发词，如果放在后面就永远不会被审到。
            moderation_outcome = self._run_moderation(message, now, mode)
            if moderation_outcome is not None:
                return moderation_outcome

        # 2) 群聊是否该回复
        if not message.is_private:
            is_command = self.is_command(text) if message.is_command is None else message.is_command
            last_reply_to = self.storage.kv_get(f"bot_last_reply_to:{message.group_id}")
            same_user_as_last = (last_reply_to is not None
                                 and str(last_reply_to) == str(message.user_id))
            if not self.config.group.relevant_topic_same_user_only:
                same_user_as_last = True
            ctx = TriggerContext(
                text=text,
                mentioned=message.mentioned,
                is_command=is_command,
                replied_to_bot=message.replied_to_bot,
                # 话题延续只对同一个人计时；别人说话时这个窗口视为不存在
                seconds_since_bot_reply=(self._seconds_since_bot_reply(message.group_id, now)
                                         if same_user_as_last else None),
                seconds_since_own_message=self._seconds_since_own_message(message.user_id, message.group_id, now),
                recent_topic_with_me=same_user_as_last,
                force=message.force_reply,
            )
            trigger = evaluate_trigger(self.config, ctx, group_record,
                                       extra_keywords=(self._current_settings.keyword_list()
                                                       if self._current_settings else None))
            if not trigger.should_reply:
                self.record(message, now)
                outcome.record_only = True
                outcome.reason = trigger.reason
                return outcome
            # 群级回复概率覆盖（/群设置 回复概率）
            probability = self.config.group.reply_probability
            if self._current_settings is not None and self._current_settings.reply_probability is not None:
                probability = float(self._current_settings.reply_probability)
            if trigger.should_reply and probability < 1.0:
                if self.rng.random() > probability and not message.mentioned:
                    self.record(message, now)
                    outcome.record_only = True
                    outcome.reason = "概率判定：本次不回复"
                    return outcome
            outcome.debug["trigger"] = trigger.matched

        # 4) 频率与额度
        verdict = self._check_limits(message, now)
        if not verdict.allowed:
            self.record(message, now)
            outcome.decision = verdict.decision.value
            outcome.reason = verdict.reason
            if verdict.decision is Decision.QUOTA_EXCEEDED:
                outcome.reply = "今天的对话额度已经用完了……剩下的部分，等重置之后再继续吧。"
            return outcome

        # 5) 正常对话
        self.record(message, now)
        # 图片理解放在检索之前：它是"看见用户发了什么"，
        # 而检索是"去外面查"。把图先看明白，后续即使要检索，
        # 检索词也能更准（而且模型的整段上下文是完整的）。
        self._analyze_inputs(message, outcome)
        if self._maybe_search(message, outcome):
            # 检索话题不明确：已经给出澄清询问，不要再让模型生成回复覆盖它
            self._remember_from_message(message, outcome, now)
            self.storage.append_message(message.user_id, "assistant", outcome.reply or "",
                                        message.group_id, now)
            consume_rate_limit(self.storage, "rl_user", f"{message.user_id}:{message.scene_id}", now)
            if not message.is_private:
                consume_rate_limit(self.storage, "rl_group", message.group_id, now)
            consume_quota(self.storage, self.config, message.user_id, message.is_private, now)
            return outcome
        try:
            reply_text, meta = self.generate_reply(message, mode, outcome)
        except LLMError as exc:
            self._remember_from_message(message, outcome, now)
            outcome.decision = "error"
            outcome.reason = str(exc)
            outcome.reply = "……爱丽丝这边的信号不太稳定，刚才那句话没能处理好。可以再说一次吗？"
            return outcome

        # **记忆写入放在模型调用之后**，这样才看得到本轮是否调过 remember 工具。
        # 放在之前会导致两条路径重复写入（详见 _remember_from_message 的说明）。
        self._remember_from_message(message, outcome, now)

        reply_text = self.postprocess(reply_text, message)
        if not reply_text:
            outcome.reason = "模型没有产生可发送内容"
            return outcome

        # 护栏 A：删掉模型泄漏进回复的内部思考（提示词禁止了，但不能只靠提示词）
        sanitized = honesty_mod.sanitize_reply(reply_text)
        if sanitized != reply_text:
            outcome.debug["sanitized"] = True
            if not sanitized:
                # 整条回复都是思考过程 -> 不要发出去，也不要说谎
                outcome.decision = "error"
                outcome.reason = "模型只输出了内部思考，没有可发送内容"
                outcome.reply = "……爱丽丝刚才走神了。可以再说一次吗？"
                return outcome
            reply_text = sanitized

        # 护栏 D：学术检索失败，但模型仍报出了具体文献信息（DOI/期刊/引用数）。
        # 这类编造用户很难发现，必须确定性拦掉。
        if outcome.academic is not None and not outcome.academic.ok:
            if honesty_mod.claims_paper_findings(reply_text):
                outcome.debug["academic_failure_replaced"] = outcome.academic.error
                reply_text = honesty_mod.academic_failure_text(
                    outcome.academic.query, outcome.academic.error,
                    outcome.academic.rate_limited)

        # 护栏 B/C：区分"没接入检索"与"接入了但这次失败"两种情形
        search_failed = bool(outcome.search is not None and not outcome.search.ok)
        outcome.debug["search_failed"] = search_failed

        # 检索失败是一种强信号：既然已经判定"这条需要最新信息"，就不能让模型
        # 用旧知识以"最新消息"的口吻作答——即使用户没说"搜索"二字。
        if search_failed and honesty_mod.claims_search_success(reply_text):
            outcome.debug["search_failure_replaced"] = (
                outcome.search.error if outcome.search else "")
            reply_text = honesty_mod.search_failure_text(
                outcome.search_query or message.text,
                outcome.search.error if outcome.search else "")

        verdict = honesty_mod.check(
            message.text, reply_text,
            honesty_mod.available_capabilities(self.config, bool(self.search_service)),
            can_search=bool(self.search_service), search_failed=search_failed)
        if verdict.replaced:
            outcome.debug["honesty_replaced"] = verdict.reason
            reply_text = verdict.text or reply_text

        reply_text = self._apply_memory_feedback(reply_text, outcome)

        consume_rate_limit(self.storage, "rl_user", f"{message.user_id}:{message.scene_id}", now)
        if not message.is_private:
            consume_rate_limit(self.storage, "rl_group", message.group_id, now)
        consume_quota(self.storage, self.config, message.user_id, message.is_private, now)
        self.storage.append_message(message.user_id, "assistant", reply_text, message.group_id, now)
        if not message.is_private:
            self.storage.kv_set(f"bot_last_reply:{message.group_id}", now)
            # 记住"上次在群里回的是谁"，用于判断话题延续
            self.storage.kv_set(f"bot_last_reply_to:{message.group_id}", str(message.user_id))

        # 角色风格护栏：小鲸鱼的"AI 味黑名单"与括号内心戏必须确定性清掉，
        # 因为这类漏网对用户是直接可见的（一条"（我不想接话）"发到群里就露馅）。
        character = characters_mod.get(mode)
        if character.style == "whale":
            report = style_mod.sanitize(reply_text)
            if report.changed:
                outcome.debug["style"] = report.summary()
            reply_text = report.cleaned
            # 分条：模型常把几句话塞进一条，这里按句子拆开，
            # 由适配层间隔发送，就像真人在补刀
            if character.split_messages:
                parts = style_mod.split_messages(reply_text, character.max_split_parts)
                if len(parts) > 1:
                    outcome.replies = parts
                    outcome.debug["split"] = len(parts)

        outcome.reply = reply_text
        # V4：按素材库与用户请求决定是否附带一个媒体（表情包/图片/语音）
        try:
            self._attach_media(message, outcome, reply_text, now)
        except Exception as exc:
            outcome.debug["media_error"] = f"{type(exc).__name__}: {exc}"
        # 媒体挂载完成之后，再拦一道"声称发了媒体但实际没发"。
        # 必须放在 _attach_media **之后**：那时才知道到底有没有挂上媒体。
        cleaned = self._strip_media_overclaim(reply_text, bool(outcome.media))
        if cleaned != reply_text:
            outcome.debug["media_overclaim"] = "已清理虚假的发送媒体声明"
            reply_text = cleaned
            outcome.reply = cleaned
        outcome.decision = "replied"
        outcome.reason = "已回复"
        outcome.llm_usage = meta.get("usage", {})
        outcome.debug.update(meta.get("debug", {}))
        self.storage.trim_conversation(message.user_id, self.config.memory.conversation_retention)
        return outcome

    # 「声称正在/已经发送媒体」的说法。**故意收得很紧**：
    # 只认动作词（发送/发出/发过去/已发/试试发…）+ 媒体词（语音/录音/图片/表情/照片），
    # 这样"爱丽丝的语音不是官方配音""这段歌词爱丽丝背不下来"这类正常陈述不会误伤。
    #
    # 为什么需要它：实测用户说"把语音发给我"时，模型在正文里编了
    # "现在爱丽丝把这段发成语音——启动！""（语音尝试发送中…）"，
    # 而那次语音**根本没发出去**。这比"直接说做不到"更糟——
    # 用户会一直等一个永远不来的语音。V5 明确要求不得编造媒体发送结果。
    _MEDIA_OVERCLAIM_PATTERNS = (
        r"[发送](出|过去|给你|给你了|送了|来了)?[^，。！？!?]{0,4}(语音|录音|声音|表情|图片|照片)",
        r"(语音|录音|表情|图片|照片)[^，。！？!?]{0,6}(已发|发了|发出|发送|来啦|来了)",
        # 语序反过来 + 中间插"已经"：「图片已经发过去了」。
        # 上面那条要求媒体词紧跟"已发/发了"，会被"已经"卡住，所以补这条。
        r"(语音|录音|表情|图片|照片)[^，。！？!?]{0,4}(已经)?[^，。！？!?]{0,2}"
        r"(发|送)(出|过去|了|给你|来)",
        r"(正在|马上|立刻|这就)[^，。！？!?]{0,6}(发送|发出|发)",
        r"尝试[^，。！？!?]{0,4}(发送|发出)",
        r"启动[^，。！？!?]{0,4}语音",
    )
    _MEDIA_OVERCLAIM_RE = re.compile("|".join(_MEDIA_OVERCLAIM_PATTERNS))

    @classmethod
    def _strip_media_overclaim(cls, text: str, has_media: bool) -> str:
        """如果正文声称发送了媒体、但本轮**没有**挂上任何媒体，就把那些说法去掉。

        has_media=True 时原样返回（确实发了，说了也没关系）。
        只删掉**含该说法的整句**，并在末尾补一句如实说明，避免留下断句。
        """
        raw = text or ""
        if has_media or not raw:
            return raw
        # 拆句（保留标点），逐句判断
        pieces = re.split(r"(?<=[。！？!?…\n])", raw)
        kept = [p for p in pieces if not cls._MEDIA_OVERCLAIM_RE.search(p)]
        if len(kept) == len(pieces):
            return raw
        body = "".join(kept).strip()
        hint = "（刚才那句「已经发了」是错的——这次并没有成功发出语音，爱丽丝不该那样说。）"
        return (body + "\n" + hint).strip() if body else hint

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_command(text: str) -> str:
        """把指令文本归一化。

        实测中发现三类会让指令"看起来收到了、其实没匹配上"的情况：
        1. 中文输入法打出的全角斜杠 ／（U+FF0F）；
        2. 前后夹带空格或全角空格；
        3. QQ 客户端/协议层留下的 CQ 码或零宽字符。
        """
        cleaned = (text or "")
        cleaned = re.sub(r"\[CQ:[^\]]*\]", "", cleaned)
        cleaned = cleaned.replace("\u3000", " ").replace("\uff0f", "/").replace("\uff3c", "/")
        cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
        return cleaned.strip()

    def is_command(self, text: str) -> bool:
        prefix = self.config.bot.command_prefix or "/"
        normalized = self.normalize_command(text)
        return normalized.startswith(prefix) or normalized.startswith("/")

    def try_command(self, message: IncomingMessage, mode: str, now: float) -> Optional[Outcome]:
        text = self.normalize_command(message.text)
        if not self.is_command(text):
            return None
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        command = _COMMAND_LOOKUP.get(command.strip().lower(), canonical_command(command.strip()))

        # 角色切换统一查注册表：新增角色不用再往这里加分支
        target = characters_mod.by_command(command)
        if target is not None:
            if target.id == MODE_ASSISTANT and not self.config.bot.assistant_mode_enabled:
                return Outcome(reply="普通助手模式当前未启用。", decision="command",
                               reason="助手模式被禁用", mode=mode)
            # V5：群级开关也要拦一道——群配置不允许时，成员不能在本群切成该模式
            if not message.is_private:
                allowed, why = self._group_allows_mode(message.group_id, target.id)
                if not allowed:
                    return Outcome(reply=why, decision="command",
                                   reason="本群不允许该模式", mode=mode)
            self.storage.set_mode(message.user_id, target.id, scope=self._mode_scope(message))
            scope_hint = "本群" if not message.is_private else "私聊"
            return Outcome(reply=f"{target.switch_reply}\n（只对{scope_hint}生效，"
                                 f"其他群和私聊不受影响）",
                           decision="command",
                           reason=f"切换到{target.name}", mode=target.id)
        if command in ("/记忆", "/查看记忆"):
            facts = self.storage.list_memory(message.user_id)
            if not facts:
                body = "现在还没有关于你的长期记录。想让我记住的话，直接说「记住……」。"
            else:
                lines = [f"{i + 1}. {f.content}" for i, f in enumerate(facts)]
                body = "爱丽丝保存的、只属于你的记录：\n" + "\n".join(lines)
            return Outcome(reply=body, decision="command", reason="查看记忆", mode=mode)
        if command in ("/忘记", "/清空记忆"):
            if not self.config.memory.enabled:
                return Outcome(reply="记忆功能当前是关闭状态。", decision="command", reason="记忆未启用", mode=mode)
            keyword = argument

            # V2 规范：「匹配内容不明确时先询问用户」。
            # 关键词命中多条时不能默默全删——那等于用户说"忘掉猫"却把别的也删了。
            if keyword:
                matches = [f for f in self.storage.list_memory(message.user_id) if keyword in f.content]
                if len(matches) > 1:
                    listing = "\n".join(f"{i}. {f.content}" for i, f in enumerate(matches, 1))
                    return Outcome(
                        reply=(f"「{keyword}」匹配到 {len(matches)} 条记录，爱丽丝不确定要删哪一条：\n"
                               f"{listing}\n"
                               "请把要删除的那条说得更具体一些，或者直接发送 /清空记忆 清空全部。"),
                        decision="command", reason="匹配不明确，等待确认", mode=mode)

            removed = self.storage.delete_memory(message.user_id, keyword)
            if keyword:
                body = (f"删除了 {removed} 条包含「{keyword}」的记录。" if removed
                        else f"没有找到包含「{keyword}」的记录。")
            else:
                body = (f"已经清空你在爱丽丝这里的全部记录（{removed} 条）。" if removed
                        else "你这里本来就没有记录，已经确认是空的。")
            return Outcome(reply=body, decision="command", reason="删除记忆", mode=mode)
        if command in ("/帮助", "/help", "/菜单"):
            return Outcome(reply=self.help_text(), decision="command", reason="帮助", mode=mode)
        if command in ("/状态", "/status"):
            stats = self.storage.stats()
            body = (f"角色：{characters_mod.get(mode).name}\n"
                    f"{quota_status_text(self.storage, self.config, message.user_id, message.is_private)}\n"
                    f"{feature_quota_text(self.storage, self.config, message.user_id, message.group_id, message.is_private)}\n"
                    f"你的记忆条数：{self.storage.count_memory(message.user_id)}\n"
                    f"服务统计：用户 {stats['users']} / 群 {stats['groups']} / 记忆 {stats['memory']} / 消息 {stats['messages']}")
            return Outcome(reply=body, decision="command", reason="状态", mode=mode)
        if command in ("/群配置", "/群设置", "/群状态"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command",
                               reason="场景不符", mode=mode)
            settings = self.storage.get_group_settings(message.group_id)
            header = f"本群配置（群号 {message.group_id}）：\n"
            keys = "、".join(sorted(set(GroupSettings.EDITABLE.keys())))
            usage = (f"用法：/群设置 <项目> <值>\n可设置项：{keys}\n"
                     "值填「默认」表示跟随全局配置。")

            # 只看不改：/群状态、/群配置（无参数）
            if command == "/群状态" or (command == "/群配置" and not argument):
                return Outcome(reply=header + settings.describe(self.config),
                               decision="command", reason="查看群配置", mode=mode)
            if not argument:
                return Outcome(reply=header + settings.describe(self.config) + "\n\n" + usage,
                               decision="command", reason="群设置用法", mode=mode)

            # 修改：需要群主/管理员/所有者
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            key, _, value = argument.partition(" ")
            if not value.strip():
                return Outcome(reply=f"缺少值。\n{usage}",
                               decision="command", reason="缺少值", mode=mode)
            ok, detail = settings.apply(key.strip(), value.strip())
            if not ok:
                return Outcome(reply=f"{detail}\n{usage}",
                               decision="command", reason="设置失败", mode=mode)
            self.storage.save_group_settings(message.group_id, settings)
            self._current_settings = settings
            return Outcome(reply=f"已更新：{detail}\n\n" + settings.describe(self.config),
                           decision="command", reason="更新群设置", mode=mode)

        if command in ("/素材", "/素材列表", "/素材上传", "/素材删除"):
            if not self._is_owner(message.user_id):
                return Outcome(reply="素材库只有机器人所有者可以管理。", decision="command",
                               reason="权限不足", mode=mode)
            if self.media_library is None:
                return Outcome(reply="素材库未启用（MEDIA_ENABLED=0）。", decision="command",
                               reason="未启用", mode=mode)
            return Outcome(reply=self._media_command(command, argument, message),
                           decision="command", reason="素材管理", mode=mode)

        if command in ("/运营", "/stats", "/运营数据"):
            if not self._is_owner(message.user_id):
                return Outcome(reply="运营数据只有机器人所有者可以查看。", decision="command",
                               reason="权限不足", mode=mode)
            data = self.operation_report()
            return Outcome(reply=data, decision="command", reason="运营数据", mode=mode)

        # ---- V5 群管理：查看审核状态（管理员）----
        if command in ("/审核", "/审查"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command",
                               reason="场景不符", mode=mode)
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            settings = self._moderation_settings(message.group_id)
            rules = moderation_mod.build_rules(settings)
            warnings = self.storage.list_warnings(message.group_id, limit=10)
            logs = self.storage.recent_moderation(message.group_id, limit=8)
            lines = [
                "【本群审核状态】",
                f"- 审核：{'开' if rules.enabled else '关'}",
                f"- 模式：{'试运行（只记录不动作）' if rules.dry_run else '会真的执行警告'}",
                f"- 自动警告：{'开' if rules.auto_warn else '关'}",
                f"- 升级禁言阈值：{rules.warn_threshold or '不升级'}",
                f"- 自定义敏感词：{'、'.join(rules.ban_words) or '（无）'}",
                f"- 白名单词：{'、'.join(rules.allow_words) or '（无）'}",
            ]
            if warnings:
                lines.append("- 本群警告累计：")
                for row in warnings:
                    lines.append(f"    {row['user_id']}：{row['count']} 次"
                                 f"（{row.get('last_reason') or '未记录'}）")
            else:
                lines.append("- 本群暂无警告记录")
            if logs:
                lines.append("- 最近动作：")
                for row in logs:
                    mark = {"ok": "完成", "failed": "失败", "skipped": "未动作",
                            "pending": "待执行"}.get(row["result"], row["result"])
                    lines.append(f"    [{mark}] {row['action']} 目标 {row['target_id'] or '-'}"
                                 f"：{(row.get('reason') or '')[:24]}")
            return Outcome(reply="\n".join(lines), decision="command",
                           reason="查看审核状态", mode=mode)

        # ---- V5 群管理：手动警告 ----
        if command in ("/警告", "/warn"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command",
                               reason="场景不符", mode=mode)
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            target, _, why = argument.partition(" ")
            target = target.strip()
            if not target:
                return Outcome(reply="用法：/警告 成员ID 原因（原因可省略）",
                               decision="command", reason="警告用法", mode=mode)
            # 不能警告自己、也不能警告管理员（避免内讧式误伤）
            if target == str(message.user_id):
                return Outcome(reply="不用警告自己。", decision="command",
                               reason="目标是本人", mode=mode)
            if self._is_group_admin(target, message.group_id):
                return Outcome(reply="对方是群主或管理员，机器人不能对其警告。",
                               decision="command", reason="目标是管理员", mode=mode)
            count = self.storage.add_warning(message.group_id, target, why or "管理员手动警告")
            self.storage.log_moderation(message.group_id, "warn", "ok", target_id=target,
                                        operator_id=message.user_id,
                                        reason=why or "管理员手动警告",
                                        detail="手动警告")
            settings = self._moderation_settings(message.group_id)
            threshold = int(getattr(settings, "warn_threshold", 0) or 0)
            if threshold and count >= threshold:
                return Outcome(
                    reply=(f"已警告 {target}（第 {count} 次，原因：{why or '未说明'}）。\n"
                           f"已达阈值 {threshold}，建议禁言。"),
                    decision="moderation_warn", reason="手动警告达阈值", mode=mode)
            return Outcome(reply=f"已警告 {target}（第 {count} 次，原因：{why or '未说明'}）。",
                           decision="moderation_warn", reason="手动警告", mode=mode)

        # ---- V5 群管理：高风险动作需要管理员确认 ----
        if command in ("/确认", "/confirm", "/取消", "/cancel"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command",
                               reason="场景不符", mode=mode)
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            action_id = 0
            if argument.strip().isdigit():
                action_id = int(argument.strip())
            pending = self.storage.get_pending_action(message.group_id, action_id)
            if not pending:
                return Outcome(reply="本群没有等待确认的管理操作。", decision="command",
                               reason="无待确认操作", mode=mode)
            label = moderation_mod.ACTION_LABELS.get(pending["action"], pending["action"])
            if command in ("/取消", "/cancel"):
                self.storage.resolve_pending_action(pending["id"], "cancelled")
                self.storage.log_moderation(message.group_id, pending["action"], "skipped",
                                            target_id=pending["target_id"],
                                            operator_id=message.user_id,
                                            reason=pending.get("reason") or "",
                                            detail="管理员取消了该操作")
                return Outcome(reply=f"已取消：{label}（#{pending['id']}）。",
                               decision="command", reason="取消待确认操作", mode=mode)
            payload = {}
            try:
                payload = json.loads(pending.get("payload") or "{}")
            except json.JSONDecodeError:
                payload = {}
            self.storage.resolve_pending_action(pending["id"], "done")
            outcome = self.execute_action(
                message.group_id, pending["action"], target_id=pending["target_id"],
                payload=payload, reason=pending.get("reason") or "",
                operator_id=message.user_id)
            outcome.reply = f"已确认 #{pending['id']}：{outcome.reply}"
            return outcome

        if command in ("/公告", "/notice"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command",
                               reason="场景不符", mode=mode)
            settings = self.storage.get_group_settings(message.group_id)
            notice = (settings.notice if settings else "") or ""
            if not argument:
                # 查看：所有人可用
                if not notice:
                    return Outcome(reply="本群还没有设置公告。", decision="command",
                                   reason="查看公告", mode=mode)
                return Outcome(reply=f"【本群公告】\n{notice}", decision="command",
                               reason="查看公告", mode=mode)
            # 设置：群主/管理员/所有者
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            if settings is None:
                settings = GroupSettings()
                self.storage.save_group_settings(message.group_id, settings)
            ok, why = settings.apply("公告", argument)
            if not ok:
                return Outcome(reply=f"公告没设置成功：{why}", decision="command",
                               reason="公告参数不合法", mode=mode)
            self.storage.save_group_settings(message.group_id, settings)
            if not settings.notice:
                return Outcome(reply="本群公告已清空。", decision="command",
                               reason="清空公告", mode=mode)
            return Outcome(reply=f"公告已更新，成员可以用 /公告 查看：\n{settings.notice}",
                           decision="command", reason="更新公告", mode=mode)

        if command in ("/群规则", "/group"):
            if message.is_private:
                return Outcome(reply="这个指令要在群里使用。", decision="command", reason="场景不符", mode=mode)
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            if not argument:
                record = self.storage.get_group(message.group_id)
                rules = (record.rules if record and record.rules else "（本群还没有单独设置规则）")
                return Outcome(reply=f"本群规则：\n{rules}", decision="command", reason="查看群规则", mode=mode)
            if argument in ("清空", "clear", "清除"):
                self.storage.upsert_group(message.group_id, rules="")
                return Outcome(reply="本群规则已清空。", decision="command", reason="清空群规则", mode=mode)
            self.storage.upsert_group(message.group_id, rules=argument)
            return Outcome(reply="本群规则已更新，爱丽丝会按新规则配合。", decision="command",
                           reason="更新群规则", mode=mode)
        if command in ("/开启", "/关闭") and not message.is_private:
            if not self._is_owner(message.user_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            enabled = command == "/开启"
            self.storage.upsert_group(message.group_id, enabled=enabled)
            return Outcome(reply="本群服务已开启。" if enabled else "本群服务已关闭，之后爱丽丝不会再主动回应。",
                           decision="command", reason="群开关", mode=mode)
        if command in ("/授权", "/approve") and not message.is_private:
            if not self._is_group_admin(message.user_id, message.group_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            record = self.storage.get_group(message.group_id) or None
            extra = dict(record.extra) if record else {}
            extra["approved"] = True
            self.storage.upsert_group(message.group_id, enabled=True, extra=extra)
            return Outcome(reply="本群已加入授权列表。", decision="command", reason="授权群", mode=mode)
        if command in ("/资料", "/reload", "/资料列表"):
            if not self._is_owner(message.user_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            base = self.reload_knowledge()
            body = ("已重新加载角色资料与提示词。\n"
                    + knowledge_mod.catalog_text(base))
            return Outcome(reply=body, decision="command", reason="重载资料", mode=mode)

        if command in ("/资料上传", "/upload"):
            if not self._is_owner(message.user_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            if not self.config.features.knowledge_upload:
                return Outcome(reply="资料上传功能当前未启用。", decision="command",
                               reason="上传未启用", mode=mode)
            # 两种用法：/资料上传 标题｜内容  或  先发文件再执行本指令
            pending = knowledge_mod.take_pending(self.config.knowledge_path, message.user_id)
            if pending and not argument:
                title, content = pending
            elif argument:
                title, _, content = argument.partition("｜")
                if not content:
                    title, _, content = argument.partition("|")
                if not content:
                    # 没写分隔符：把整段当正文，标题用前 12 字
                    title, content = argument[:12].strip(), argument
            else:
                return Outcome(
                    reply="用法：把资料内容按「标题｜正文」发给我，例如：\n"
                          "/资料上传 活动规则｜本期活动到 5 月 20 日结束……\n"
                          "也可以先发一个 txt 文件，再发送 /资料上传。",
                    decision="command", reason="缺少内容", mode=mode)
            if len(content.strip()) < 5:
                return Outcome(reply="内容太短了，没有保存。", decision="command",
                               reason="内容过短", mode=mode)
            if memory_mod.looks_sensitive(content):
                return Outcome(reply="这段内容看起来包含敏感信息，为安全起见没有保存。",
                               decision="command", reason="疑似敏感", mode=mode)
            path = knowledge_mod.save_upload(
                self.config.knowledge_path, title or "上传资料", content,
                prefix=self.config.bot.knowledge_upload_prefix)
            base = self.reload_knowledge()
            return Outcome(reply=f"已保存资料：{path.name}\n" + knowledge_mod.catalog_text(base),
                           decision="command", reason="上传资料", mode=mode)

        if command in ("/资料删除", "/资料移除"):
            if not self._is_owner(message.user_id):
                return Outcome(reply=self._owner_hint(message.user_id), decision="command",
                               reason="权限不足", mode=mode)
            if not argument:
                return Outcome(reply="用法：/资料删除 文件名.md", decision="command",
                               reason="缺少文件名", mode=mode)
            removed = knowledge_mod.delete_doc(self.config.knowledge_path, argument)
            if not removed:
                return Outcome(reply=f"没有找到可删除的资料：{argument}", decision="command",
                               reason="未找到", mode=mode)
            base = self.reload_knowledge()
            return Outcome(reply=f"已删除 {argument}。\n" + knowledge_mod.catalog_text(base),
                           decision="command", reason="删除资料", mode=mode)
        return Outcome(reply="爱丽丝不认识这条指令。发送 /帮助 可以看到目前的指令列表。",
                       decision="command", reason="未知指令", mode=mode)

    _MEDIA_HELP = (
        "素材库指令（仅所有者）：\n"
        "/素材列表 —— 列出素材（含 id、类型、标签）\n"
        "/素材删除 <id> —— 删除素材\n"
        "/素材上传 <标签>｜<说明> —— 标记后，发一张图片/表情即可入库\n"
        "  例：/素材上传 困,累,睡觉｜打哈欠\n"
        "素材按标签匹配话题；只有匹配上才会发送，不会随便挑一张。"
    )

    def _media_command(self, command: str, argument: str, message: IncomingMessage) -> str:
        library = self.media_library
        if command == "/素材删除":
            if not argument:
                return "用法：/素材删除 <id>\n\n" + self._media_help_text()
            item = library.get(argument.strip())
            if item is None:
                return f"没有找到素材 {argument.strip()}"
            ok = library.remove(item.id)
            return f"已删除 {item.id}" if ok else f"删除失败：{item.id}"
        if command == "/素材上传":
            if not argument:
                return "用法：/素材上传 <标签>｜<说明>\n\n" + self._media_help_text()
            tags, _, desc = argument.partition("｜")
            if not desc:
                tags, _, desc = argument.partition("|")
            tags, desc = tags.strip(), desc.strip()
            if not tags:
                return "标签不能为空。\n\n" + self._media_help_text()
            self.storage.kv_set(f"media_pending:{message.user_id}",
                                {"tags": tags, "desc": desc})
            # 说明是可选的：只给标签时不要输出"说明「」"这种别扭的话
            summary = f"标签「{tags}」" + (f"、说明「{desc}」" if desc else "（没写说明，没关系）")
            return (f"好的，{summary}。\n\n"
                    "现在把图片或表情发过来，我就收进素材库。\n"
                    "如果是 QQ 内置小黄脸，请**转发**过来或截图发送——"
                    "内置表情拿不到图片文件。")
        # /素材 与 /素材列表
        items = library.all()
        if not items:
            return ("素材库还是空的。\n\n" + self._media_help_text() +
                    "\n\n提示：没有素材时，爱丽丝不会发送任何图片或表情，"
                    "也不会声称已经发送。")
        lines = [f"素材库共 {len(items)} 项："]
        for kind_label, kind in (("表情包", "sticker"), ("图片", "image"),
                                 ("生成图", "generated"), ("语音", "voice")):
            group_items = [i for i in items if i.kind == kind]
            if not group_items:
                continue
            lines.append(f"\n【{kind_label}】{len(group_items)} 项")
            for item in group_items[:30]:
                tags = "、".join(item.tags) or "无标签"
                origin = "（AI 生成）" if item.is_generated else ""
                lines.append(f"  {item.id}｜{item.desc or '未命名'}{origin}｜标签：{tags}")
        lines.append("\n" + self._media_help_text())
        return "\n".join(lines)

    @staticmethod
    def _media_help_text() -> str:
        return Engine._MEDIA_HELP

    def operation_report(self) -> str:
        """所有者可见的运营数据。

        V3 明确：所有者"可以查看运营数据，但不能随意查看用户私人聊天内容"。
        所以这里**只输出聚合计数与配置状态**，不含任何消息正文、记忆内容或用户身份明细。
        """
        stats = self.storage.stats()
        period = time.strftime("%Y%m%d")
        lines = [
            "【运营数据】（仅聚合统计，不含任何聊天内容或用户明细）",
            f"- 用户数：{stats['users']}",
            f"- 群数：{stats['groups']}",
            f"- 记忆条数：{stats['memory']}",
            f"- 消息条数：{stats['messages']}",
            f"- 今日全站对话额度：{self.storage.get_counter('quota_global', 'all', period)}"
            f"/{self.config.quota.daily_global}",
            f"- 今日全站联网检索：{self.storage.get_counter('quota_search_global', 'all', period)}"
            f"/{self.config.quota.daily_search_global}",
        ]
        groups = self.storage.list_groups()
        if groups:
            lines.append(f"- 已授权群：{len(groups)} 个")
            for record in groups[:10]:
                settings = GroupSettings.from_extra(record.extra)
                flags = []
                for name, value in (("主动", settings.proactive), ("搜索", settings.search)):
                    if value is not None:
                        flags.append(f"{name}{'开' if value else '关'}")
                lines.append(f"    · 群 {record.group_id}：{'启用' if record.enabled else '停用'}"
                             + (f"（{'、'.join(flags)}）" if flags else ""))
        lines.append("- 说明：出于隐私保护，此报表不展示任何用户消息或记忆内容。")
        return "\n".join(lines)

    @staticmethod
    def _character_switch_lines() -> str:
        """角色切换说明跟着注册表走，新增角色不用改这里。"""
        lines = []
        for character in characters_mod.CHARACTERS.values():
            if not character.switch_commands:
                continue
            primary = character.switch_commands[0]
            aliases = character.switch_commands[1:]
            suffix = ("（别名 " + "、".join(aliases) + "）") if aliases else ""
            lines.append(f"{primary}{suffix} —— 切换到{character.name}")
        return "\n".join(lines)

    def help_text(self) -> str:
        return (
            "== 所有人可用 ==\n"
            + self._character_switch_lines() + "\n"
            "/记忆 —— 查看只属于你的长期记忆\n"
            "/忘记 [关键词] —— 删除记录；命中多条会先问你；不带关键词即清空\n"
            "/状态 —— 查看当前模式、额度与记忆条数\n"
            "/帮助 —— 显示这份列表\n"
            "\n== 群内可用 ==\n"
            "/群配置 —— 查看本群设置（所有人可看）\n"
            "/群规则 —— 查看本群规则\n"
            "/公告 —— 查看本群公告\n"
            "\n== 群主 / 群管理员 / 所有者 ==\n"
            "/群设置 <项目> <值> —— 例如 /群设置 搜索 关；填「默认」恢复跟随全局\n"
            "/群规则 <文本> —— 设置本群规则\n"
            "/公告 <文本> —— 设置本群公告；发「关」清空\n"
            "/审核 —— 查看本群审核状态、警告记录与最近动作\n"
            "/警告 <成员ID> [原因] —— 手动警告（不会自动禁言）\n"
            "/确认 —— 执行待确认的高影响操作（禁言/踢出/撤回）\n"
            "/取消 —— 放弃待确认的操作\n"
            "\n== 仅机器人所有者 ==\n"
            "/开启 /关闭 —— 开关本群服务\n"
            "/授权 —— 把本群加入授权列表\n"
            "/资料 [/资料列表] —— 重载并列出角色资料\n"
            "/资料上传 标题｜正文 —— 添加资料（也可先发 txt 再执行）\n"
            "/资料删除 文件名.md —— 删除资料\n"
            "/运营 —— 查看运营数据（仅聚合统计，不含任何聊天内容）\n"
            "/素材列表 —— 查看表情包/图片素材\n"
            "/素材上传 <标签>｜<说明> —— 标记后发一张图即入库\n"
            "/素材删除 <id> —— 删除素材"
        )

    # ------------------------------------------------------------------
    # V5：群管理（审核判定 + 动作执行 + 管理员确认）
    # ------------------------------------------------------------------
    def _moderation_settings(self, group_id: str):
        """取本群设置（优先用本轮已算好的，避免重复读库）。"""
        if self._current_settings is not None:
            return self._current_settings
        return self.storage.get_group_settings(group_id)

    def _run_moderation(self, message: IncomingMessage, now: float, mode: str):
        """对一条群消息做审核。返回 Outcome 表示"本轮到此为止"，None 表示继续正常流程。

        安全设计（对应 V5 第六节"默认安全策略"）：

        - 群里**默认 dry_run**：命中只记审计日志 + 提示管理员，不发警告、不处罚。
          一上来就自动处罚，误伤的代价远大于漏掉一条广告。
        - 高风险动作（禁言/踢出/撤回）**一律进待确认队列**，必须管理员确认。
        - 只有 medium 以上、且开了自动警告，才会真的发警告。
        - 命中的处罚理由一律先过 `moderation.is_never_punish`——
          批评机器人、求助、观点分歧永远不构成处罚理由。
        """
        if message.is_private or not message.group_id:
            return None
        settings = self._current_settings
        rules = moderation_mod.build_rules(settings)
        if not rules.enabled:
            return None
        # 管理员豁免：群主/管理员/所有者自己发的消息不审
        if self._is_group_admin(message.user_id, message.group_id):
            return None
        history = self.storage.recent_user_group_messages(
            message.group_id, message.user_id,
            within_seconds=rules.repeat_window, limit=12)
        verdict = moderation_mod.judge_ad(message.text, history=history, rules=rules)
        if not verdict.hit:
            return None

        detail = verdict.explain()
        # 只记录不动作（试运行），或这轮不允许自动动作
        if rules.dry_run or not rules.auto_warn:
            self.storage.log_moderation(
                message.group_id, "review", "skipped", target_id=message.user_id,
                reason=verdict.reason, detail=detail)
            if verdict.confidence in (moderation_mod.CONF_MEDIUM, moderation_mod.CONF_HIGH):
                # 提示管理员（不公开点名，避免误伤时把人也一起挂了）
                mode_note = "试运行" if rules.dry_run else "仅观察（自动警告已关）"
                return Outcome(
                    reply=(f"【审核观察】这条消息可能有问题：{verdict.label}。"
                           f"当前是{mode_note}，没有对它做任何处理。\n"
                           f"依据：{verdict.reason}\n"
                           "要真的处理，管理员可以 /群设置 审核试运行 关，"
                           "或用 /警告 目标 原因 手动警告。"),
                    decision="moderation_review", reason=detail, mode=mode)
            return None

        # 低置信度：即使允许动作也不处罚，只记录
        if verdict.confidence == moderation_mod.CONF_LOW:
            self.storage.log_moderation(
                message.group_id, "review", "skipped", target_id=message.user_id,
                reason=verdict.reason, detail=detail + "（置信度不足，未动作）")
            return None

        # 到这里说明允许自动动作：先警告
        count = self.storage.add_warning(message.group_id, message.user_id, verdict.reason)
        self.storage.log_moderation(
            message.group_id, "warn", "ok", target_id=message.user_id,
            operator_id="auto", reason=verdict.reason, detail=detail)
        threshold = rules.warn_threshold or int(
            getattr(settings, "warn_threshold", 0) or 0)
        if threshold and count >= threshold:
            # 达到阈值 -> 升级为禁言，但仍要管理员确认
            action_id = self.storage.create_pending_action(
                message.group_id, moderation_mod.ACTION_MUTE, target_id=message.user_id,
                payload={"seconds": 600}, reason=verdict.reason,
                requested_by="auto")
            return Outcome(
                reply=(f"已给出第 {count} 次警告（{verdict.label}）。\n"
                       f"累计达到 {threshold} 次，建议禁言 10 分钟。"
                       f"管理员发送 /确认 执行，或 /取消 放弃。（待确认编号 {action_id}）"),
                decision="moderation_pending", reason=detail,
                mode=mode)
        return Outcome(
            reply=f"（这是第 {count} 次警告：{verdict.label}。{verdict.reason}）",
            decision="moderation_warn", reason=detail,
            mode=mode)

    def _pending_action_text(self, group_id: str, action_id: int = 0) -> str:
        pending = self.storage.get_pending_action(group_id, action_id)
        if not pending:
            return "本群没有等待确认的管理操作。"
        label = moderation_mod.ACTION_LABELS.get(pending["action"], pending["action"])
        who = pending.get("target_id") or "?"
        return (f"待确认 #{pending['id']}：{label} 目标 {who}\n"
                f"原因：{pending.get('reason') or '（未说明）'}\n"
                f"发起人：{pending.get('requested_by') or '?'}\n"
                "管理员发送 /确认 执行，或 /取消 放弃。")

    # ------------------------------------------------------------------
    # 动作执行：把引擎的决定翻译成"适配层要调用的 QQ 接口"。
    #
    # 为什么不在引擎里直接调 QQ API：核心必须能脱离 NoneBot 离线测试，
    # 而且"执行失败"必须如实上报（V5 明确不得伪造成功），
    # 所以这里只产出**动作请求**，由适配层执行后回填结果。
    # ------------------------------------------------------------------
    def queue_action(self, group_id: str, action: str, target_id: str = "",
                     payload: Optional[Dict[str, Any]] = None, reason: str = "",
                     operator_id: str = "") -> Outcome:
        """把一个管理动作排进队列（高风险动作需要确认）。"""
        label = moderation_mod.ACTION_LABELS.get(action, action)
        settings = self._current_settings
        if settings is None:
            settings = self.storage.get_group_settings(group_id)
        rules = moderation_mod.build_rules(settings)
        if moderation_mod.needs_confirmation(action, rules):
            action_id = self.storage.create_pending_action(
                group_id, action, target_id=target_id, payload=payload or {},
                reason=reason, requested_by=operator_id)
            return Outcome(
                reply=(f"将要执行：{label}（目标 {target_id or '?'}）。\n"
                       f"这是高影响操作，需要确认。管理员发送 /确认 执行，"
                       f"或 /取消 放弃。（待确认编号 {action_id}）"),
                decision="moderation_pending", reason=f"待确认 {label}")
        return self.execute_action(group_id, action, target_id=target_id,
                                   payload=payload or {}, reason=reason,
                                   operator_id=operator_id)

    def execute_action(self, group_id: str, action: str, target_id: str = "",
                       payload: Optional[Dict[str, Any]] = None,
                       reason: str = "", operator_id: str = "") -> Outcome:
        """真正执行一个管理动作 —— 产出动作请求并记审计。

        返回的 Outcome 里 `debug["action"]` 是适配层需要执行的动作描述；
        文本回复是"已发起/失败"的如实说明。
        **注意：这里不代表 QQ 那边已经成功**，适配层执行后必须调用
        `record_action_result()` 回填真实结果。
        """
        label = moderation_mod.ACTION_LABELS.get(action, action)
        outcome = Outcome(reply=f"已发起{label}。", decision="moderation_action",
                          reason=f"执行{label}")
        outcome.debug["action"] = {
            "action": action, "group_id": str(group_id), "target_id": str(target_id),
            "payload": dict(payload or {}), "reason": str(reason),
            "operator_id": str(operator_id),
        }
        self.storage.log_moderation(group_id, action, "pending", target_id=target_id,
                                    operator_id=operator_id, reason=reason,
                                    detail=f"{label} 已交给适配层执行")
        return outcome

    def record_action_result(self, group_id: str, action: str, ok: bool, target_id: str = "",
                             operator_id: str = "", reason: str = "",
                             detail: str = "") -> str:
        """适配层执行完后回填结果，返回给用户看的说明。

        V5：**不得伪造已撤回/禁言/踢出成功**。所以失败时必须说清原因，
        成功时也不夸大（例如"已撤回"而不是"已教育"）。
        """
        label = moderation_mod.ACTION_LABELS.get(action, action)
        self.storage.log_moderation(group_id, action, "ok" if ok else "failed",
                                    target_id=target_id, operator_id=operator_id,
                                    reason=reason, detail=detail)
        if ok:
            return f"{label}已完成。"
        why = detail or "未知原因"
        return f"{label}**失败**了：{why}"


    def current_mode(self, message: IncomingMessage) -> str:
        """取当前**会话**的模式（V5：模式按会话隔离）。

        优先级：本会话显式设置 -> 用户全局默认 -> 配置默认。
        为什么要隔离：同一个用户在 A 群切成助手后，B 群和私聊不应该跟着变——
        V5 明确要求"不把一个群的私人信息带到另一个群"，模式同属会话状态。
        """
        scope = self._mode_scope(message)
        session_mode = self.storage.get_session_mode(message.user_id, scope)
        if session_mode:
            mode = session_mode
        else:
            profile = self.storage.get_profile(message.user_id)
            if profile is not None:
                mode = profile.mode or self.config.bot.default_mode
            else:
                mode = self.config.bot.default_mode
        if not self.config.bot.assistant_mode_enabled and mode == MODE_ASSISTANT:
            mode = MODE_ALICE
        # 群级模式开关：群配置不允许时，即使本会话设过也要退回允许的模式。
        # 这样"群里禁掉助手模式"真的生效，而不是只拦住切换动作。
        if not message.is_private and message.group_id:
            settings = self.storage.get_group_settings(message.group_id)
            if settings is not None:
                if mode == MODE_ASSISTANT and settings.allow_assistant is False:
                    mode = MODE_ALICE
                elif mode == MODE_ALICE and settings.allow_alice is False:
                    mode = MODE_ASSISTANT
        # 未知/脏值回落到默认角色，不抛异常
        return characters_mod.get(mode).id

    def _group_allows_mode(self, group_id: str, mode: str) -> tuple:
        """群是否允许某模式，返回 (允许?, 拒绝说明)。"""
        settings = self.storage.get_group_settings(group_id)
        if settings is None:
            return True, ""
        if mode == MODE_ASSISTANT and settings.allow_assistant is False:
            return False, "本群已关闭普通助手模式，请联系群主或管理员在群里用 /群设置 助手 开 打开。"
        if mode == MODE_ALICE and settings.allow_alice is False:
            return False, "本群已关闭爱丽丝模式，请联系群主或管理员在群里用 /群设置 爱丽丝 开 打开。"
        return True, ""

    @staticmethod
    def _mode_scope(message: IncomingMessage) -> str:
        """模式的作用域：私聊用 'private'，群聊用群号。"""
        if message.is_private or not message.group_id:
            return db_mod.SCOPE_PRIVATE
        return str(message.group_id)

    # ------------------------------------------------------------------
    # V5：群成员变动（入群欢迎 / 退群告别）
    # ------------------------------------------------------------------
    def member_event(self, group_id: str, user_id: str, user_name: str,
                     action: str, operator_id: str = "") -> Outcome:
        """处理一次群成员变动事件，返回要发送的话（没有就返回空 Outcome）。

        action：'approve'（入群）| 'leave'（主动退群）| 'kick'（被移出）| 'kick_me'

        设计要点（对应 V5 第六节）：
        - 只在**群配置里确实设了**欢迎语/告别语时才发言；没配置就完全不打扰。
        - 欢迎语里的 {name}/{group} 用简单替换实现，不用 str.format——
          否则用户写的花括号会被当成格式符，配置里写个表情就抛异常。
        - 不泄露新成员隐私：只用显示名，不带任何记忆或历史信息。
        - 群服务被关掉时，连欢迎语也不发。
        """
        if not group_id:
            return Outcome(reply=None, decision="silent", reason="没有群号")
        record = self.storage.get_group(group_id)
        if record is not None and not record.enabled:
            return Outcome(reply=None, decision="silent", reason="本群服务已关闭")
        settings = self.storage.get_group_settings(group_id)
        if settings is None:
            return Outcome(reply=None, decision="silent", reason="本群没有单独配置")
        if action in ("approve", "invite"):
            template = settings.welcome
            label = "入群欢迎"
        elif action in ("leave",):
            template = settings.farewell
            label = "退群告别"
        else:
            # 被踢出（含机器人自己被踢）不发告别：那是冲突场景，
            # 发"再见"容易被理解成嘲讽。V5 也只要求欢迎和告别。
            return Outcome(reply=None, decision="silent", reason=f"事件 {action} 不发言")
        if not template:
            return Outcome(reply=None, decision="silent", reason=f"未配置{label}语")
        text = (template.replace("{name}", user_name or "新成员")
                        .replace("{group}", str(group_id)))
        # 兜底限长：欢迎语是发到群里的，不能变成刷屏
        if len(text) > 200:
            text = text[:200]
        return Outcome(reply=text, decision="member_event", reason=label,
                       mode=self.config.bot.default_mode)

    def build_context(self, message: IncomingMessage, mode: str) -> Dict[str, Any]:
        profile = self.storage.ensure_profile(message.user_id, message.user_name, self.config.bot.default_mode)
        facts = self.storage.list_memory(message.user_id) if self.config.memory.enabled else []
        history = self.storage.recent_messages(message.user_id, self.config.llm.context_messages,
                                               group_id=message.group_id if not message.is_private else "")
        summary = self._summarize(history, facts, message, profile)
        group_context = "（当前为私聊场景）" if message.is_private else self._group_context(message)
        group_record = None if message.is_private else self.storage.get_group(message.group_id)

        variables = {
            "mode": mode,
            "user_id": message.user_id,
            "user_name": profile.preferred_name,
            "group_id": message.group_id or "（私聊）",
            "is_private_chat": "true" if message.is_private else "false",
            "current_time": self._now_text(message.timestamp),
            "user_memory": memory_mod.render_memory_block(facts),
            "conversation_summary": summary,
            "group_context": group_context,
            "knowledge_base": self.knowledge_block() or "（未加载角色资料）",
            "group_config": self._group_config_text(group_record, message),
            "available_tools": tools_mod.catalog_text(
                self.config, bool(self.search_service), bool(self.academic_service)),
            "quota_status": (quota_status_text(self.storage, self.config, message.user_id, message.is_private)
                             + "；" + feature_quota_text(self.storage, self.config, message.user_id,
                                                          message.group_id, message.is_private)),
            "permission_context": permission_context_text(self.storage, self.config, message.user_id,
                                                         message.is_private, message.group_id),
            "media_catalog": self._media_catalog(message),
            "media_intent": self._media_intent(message),
            # 用户这次发的图片的识别结果。
            # 没发图时是空串；发了图但看不到时是一段"你看不到图"的说明
            # ——**这时候也必须注入**，否则模型会以为没图这回事，转而瞎猜。
            "vision_result": self._vision_block(message),
            # 用户这次发的语音转写。和图片同理：听不到也要说。
            "transcript": self._transcript_block(message),
            # 用户这次发的视频（画面抽帧 + 音轨转写）。
            "video_result": self._video_block(message),
            "search_results": self._search_block(message, mode),
            "event_type": "private_message" if message.is_private else "group_message",
            "agent_name": self.config.bot.agent_name,
            "character": self.config.bot.character,
            "relationship": self.config.bot.relationship,
            "romance_allowed": "true" if self.config.bot.romance_allowed else "false",
            "group_rules": (group_record.rules if group_record and group_record.rules else "（本群没有额外规则）"),
        }
        return variables

    def build_messages(self, message: IncomingMessage, mode: str) -> tuple:
        variables = self.build_context(message, mode)
        template_name = characters_mod.get(mode).template
        system = prompt_mod.render(self.template(template_name), variables)
        history = self.storage.recent_messages(
            message.user_id, self.config.llm.context_messages,
            group_id=message.group_id if not message.is_private else "",
            exclude_current=True)
        messages: List[Message] = [{"role": "system", "content": system}]
        for item in history:
            if item["role"] in ("user", "assistant"):
                messages.append({"role": item["role"], "content": item["content"]})
        current_text = message.text.strip() if message.is_private else strip_mention(
            message.text, [self.config.bot.character, self.config.bot.agent_name])
        messages.append({"role": "user",
                         "content": current_text or self._media_placeholder(message)})
        return system, messages, variables

    def _media_placeholder(self, message: IncomingMessage) -> str:
        """文字为空但带媒体时，给 user 消息一个占位文本。

        **为什么需要这个。** 用户直接甩一张图 / 发一条语音时
        `message.text` 是空的，原来就原样塞进 `content`，于是发给模型的
        user 消息是**空字符串**。图片描述虽然在 system 里，但连续几条
        「空 user 消息」是个退化输入模式——实测（真机 + 本地复现）
        模型会回"你只发了一个空消息""爱丽丝没收到内容哦"，
        哪怕描述就在它的 system 里。

        改成给一句固定的占位文本，模型就有具体的东西可以回应，
        不必靠"注意 system 里那段"。

        注意这里**不描述媒体内容**，只说"发了什么类型"——
        真正的描述由 `_vision_block` / `_transcript_block` / `_video_block`
        注入 system，那是唯一的信息来源，不能让两处都描述内容
        （否则模型可能把占位文本当成描述来编）。
        """
        kinds = []
        if message.has_images:
            kinds.append("%d 张图片" % len(message.images))
        if message.has_audios:
            kinds.append("%d 条语音" % len(message.audios))
        if message.has_videos:
            kinds.append("%d 段视频" % len(message.videos))
        if not kinds:
            return ""
        return "（用户发来了%s，但没有打字。请根据上面的识别结果回应。）" % "、".join(kinds)

    def _tool_rounds_limit(self) -> int:
        """工具循环最多跑几轮。可配置，便于测试时收紧。"""
        try:
            n = int(getattr(self.config.llm, "max_tool_rounds", MAX_TOOL_ROUNDS) or MAX_TOOL_ROUNDS)
        except (TypeError, ValueError):
            n = MAX_TOOL_ROUNDS
        return max(1, min(n, 6))          # 上限 6：再多对 QQ 对话没有意义

    def _run_tool_loop(self, messages: List[Message], schemas: List[Dict[str, Any]],
                       reply: Reply, message: IncomingMessage, outcome: Outcome,
                       debug: Dict[str, Any]) -> Reply:
        """多轮工具编排：让模型能连续调多个工具，而不是只回灌一次。

        以前的实现只做一次"执行 -> 回灌 -> 再问一次"，而且那次再问
        外面套着 `except LLMError: pass`——**失败了什么也不说**。
        现在改成真正的循环：

            模型要工具 -> 执行 -> 以 role=tool 回灌 -> 再问
                       -> 它可能还要下一个工具 -> ...

        三条约束，都是有意加的：

        **1) 有轮数上限。** 模型可能陷入"一直调工具不出结论"。
        超出上限时**如实说明**，不能假装回答完了。

        **2) 同一个工具不重复调。** 带不同参数不算重复（比如先查
        A 的论文再查 B 的）；但完全相同的调用再来一次只会浪费
        额度和时间，而且往往意味着模型在绕圈。

        **3) 失败不再静默。** 某轮调用模型失败时，把"中途出错"
        记进 debug，并用已有的结果组织回复，而不是无声无息。
        """
        limit = self._tool_rounds_limit()
        all_results: List[Dict[str, Any]] = []
        seen_calls = set()
        rounds = 0
        loop_error = ""
        hit_limit = False

        while reply.tool_calls and rounds < limit:
            rounds += 1
            followup: List[Message] = list(messages)
            followup.append({"role": "assistant", "content": reply.text or "",
                             "tool_calls": reply.tool_calls})

            for call in reply.tool_calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "").strip()
                raw_args = fn.get("arguments") or "{}"
                try:
                    arguments = (json.loads(raw_args) if isinstance(raw_args, str)
                                 else dict(raw_args))
                except (json.JSONDecodeError, TypeError, ValueError):
                    arguments = {}

                # 完全相同的调用不再重复执行，但仍然回灌一条说明，
                # 让模型知道"这个你已经调过了"，避免它以为是失败而重试。
                fingerprint = (name, json.dumps(arguments, sort_keys=True,
                                                ensure_ascii=False))
                if fingerprint in seen_calls:
                    result = tools_mod.ToolResult(
                        name, False, "这个调用刚刚已经执行过了，结果见上一条。")
                else:
                    seen_calls.add(fingerprint)
                    try:
                        result = self._execute_tool(name, arguments, message, outcome)
                    except Exception as exc:
                        # 工具执行本身抛异常不能带崩整轮对话
                        result = tools_mod.ToolResult(
                            name, False,
                            "执行出错：%s: %s" % (type(exc).__name__, str(exc)[:100]))
                all_results.append({"name": name, "ok": result.ok,
                                    "message": result.message})
                followup.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "name": name,
                    "content": json.dumps({"ok": result.ok, "message": result.message},
                                          ensure_ascii=False),
                })

            try:
                nxt = self.llm.chat(followup, config=self.config.llm,
                                    tools=schemas or None)
            except LLMError as exc:
                # **以前这里是 `except LLMError: pass`**，失败完全无声。
                loop_error = str(exc)[:160]
                break
            if nxt.text:
                reply = nxt
            elif nxt.tool_calls:
                reply = nxt              # 只有工具调用没有文字，继续下一轮
            else:
                break                    # 空回复且不再要工具，没得继续

        if reply.tool_calls and rounds >= limit:
            hit_limit = True

        if all_results:
            outcome.tool_results = all_results

        # 没有任何文字时兜底。**要区分"做成了"和"没做成"**，
        # 也要区分"中途出错"——不能一律说"已经处理了"。
        if not reply.text:
            if loop_error:
                reply.text = ("……爱丽丝在处理这件事的中途遇到了一点问题"
                              "（%s），所以没能给出完整结果。" % loop_error)
            elif hit_limit:
                reply.text = "这件事爱丽丝查了几轮还没理清楚，先给你目前的结果吧。"
            elif all_results:
                ok_count = sum(1 for t in all_results if t["ok"])
                reply.text = ("……嗯，这件事爱丽丝已经处理了。" if ok_count
                              else "这件事爱丽丝现在做不到，需要换一种方式。")

        debug["tool_rounds"] = rounds
        debug["tool_calls_total"] = len(all_results)
        debug["tools_used"] = [t["name"] for t in all_results]
        if hit_limit:
            debug["tool_round_limit"] = limit
        if loop_error:
            debug["tool_loop_error"] = loop_error
        # 循环结束后模型可能还想调工具（撞上限），这些没执行的不该
        # 留在 outcome 里让上层以为执行过了。
        reply.tool_calls = [] if hit_limit else reply.tool_calls
        return reply

    def generate_reply(self, message: IncomingMessage, mode: str, outcome: Outcome) -> tuple:
        system, messages, variables = self.build_messages(message, mode)

        # 把工具 schema 真正传给模型。
        # **这是修一个接线漏洞**：`OpenAICompatibleLLM.chat()` 一直支持 `tools=`、
        # 也会解析 `tool_calls`，但调用方从来没传过——于是
        # `reply.tool_calls` 恒为空，下面那段"执行工具再回灌"的循环**永远不执行**。
        # 工具以前只是以文字形式写在提示词里（available_tools 变量），
        # 模型无法真正发起调用，只能"声称"自己用了工具。
        schemas = tools_mod.openai_tool_schemas(
            self.config, bool(self.search_service), bool(self.academic_service))
        debug_tools = [s["function"]["name"] for s in schemas]

        reply = self.llm.chat(messages, config=self.config.llm,
                              tools=schemas or None)
        debug = {"provider": reply.provider, "model": reply.model, "raw": reply.raw,
                 "tools_offered": debug_tools}

        if reply.tool_calls:
            reply = self._run_tool_loop(messages, schemas, reply, message, outcome, debug)

        outcome.prompt_system = system
        outcome.messages = messages
        return reply.text, {"usage": reply.usage, "debug": debug}

    def _execute_tool(self, name: str, arguments: Dict[str, Any], message: IncomingMessage,
                      outcome: Outcome) -> tools_mod.ToolResult:
        if name == "remember":
            content = str(arguments.get("content", "")).strip()
            if not self.config.memory.enabled:
                return tools_mod.ToolResult(name, False, "记忆功能当前未启用。")
            if memory_mod.looks_sensitive(content):
                return tools_mod.ToolResult(name, False, "这条内容涉及敏感信息，按规则不会保存。")
            if content:
                self.storage.add_memory(message.user_id, content, kind="explicit", source="tool", confidence=0.9)
                self.storage.enforce_memory_limit(message.user_id, self.config.memory.per_user_limit)
                outcome.memory_saved.append(content)
                return tools_mod.ToolResult(name, True, "已记录。", {"content": content})
            return tools_mod.ToolResult(name, False, "没有提供要记录的内容。")
        if name == "academic_search":
            if self.academic_service is None:
                return tools_mod.ToolResult(name, False, "学术检索当前未启用。")
            query = str(arguments.get("query") or arguments.get("q") or "").strip()
            if not query:
                return tools_mod.ToolResult(name, False, "没有给出要检索的文献主题。")
            try:
                academic = self.academic_service.search(
                    query, limit=self.config.search.max_results)
            except Exception as exc:
                return tools_mod.ToolResult(name, False, f"学术检索异常：{type(exc).__name__}: {exc}")
            outcome.academic = academic
            self._pending_academic = academic
            self._pending_search_owner = (str(message.user_id), message.scene_id)
            outcome.debug["academic"] = {
                "provider": academic.provider, "ok": academic.ok,
                "count": len(academic.results), "rate_limited": academic.rate_limited,
                "error": academic.error,
            }
            if not academic.ok:
                hint = "（可能是被限流，如实说明并建议稍后再试）" if academic.rate_limited else ""
                return tools_mod.ToolResult(name, False, f"学术检索失败：{academic.error}{hint}")
            return tools_mod.ToolResult(
                name, True,
                f"检索到 {len(academic.results)} 篇文献，请给出标题与链接后作答。",
                {"results": [r.to_dict() for r in academic.results],
                 "formatted": format_academic(academic)})

        if name == "web_search":
            if self.search_service is None:
                return tools_mod.ToolResult(name, False,
                                            "联网检索当前未启用，爱丽丝查不到最新信息。")
            query = str(arguments.get("query") or arguments.get("q") or "").strip()
            if not query:
                return tools_mod.ToolResult(name, False, "没有给出要检索的内容。")
            try:
                result = self.search_service.search(query, limit=self.config.search.max_results)
            except Exception as exc:
                return tools_mod.ToolResult(name, False, f"检索失败：{type(exc).__name__}: {exc}")
            outcome.search = result
            outcome.search_query = result.query
            self._pending_search = result
            self._pending_search_owner = (str(message.user_id), message.scene_id)
            outcome.debug["search"] = {"provider": result.provider, "ok": result.ok,
                                       "count": len(result.results), "elapsed": result.elapsed,
                                       "error": result.error}
            if not result.ok:
                return tools_mod.ToolResult(name, False, f"检索失败：{result.error}")
            if not result.results:
                return tools_mod.ToolResult(name, False, "没有检索到相关结果。")
            return tools_mod.ToolResult(
                name, True,
                f"检索到 {len(result.results)} 条结果，请据此回答并标注来源。",
                {"results": [r.to_dict() for r in result.results],
                 "formatted": self.search_service.format_for_model(result)})
        return tools_mod.call(self.config, name, arguments)

    def postprocess(self, text: str, message: IncomingMessage) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        # 所有者上传的资料里可能夹着手机号，输出侧统一脱敏，避免群里公开传播
        cleaned = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]", cleaned)
        # 去掉模型可能吐出的内部标记
        cleaned = re.sub(r"^(爱丽丝|Alice|助手)\s*[：:]\s*", "", cleaned)
        cleaned = re.sub(r"\{\{[^}]+\}\}", "", cleaned)
        if message.is_private:
            limit = self.config.private.max_reply_chars
        else:
            # 群级字数覆盖（/群设置 字数上限）
            settings = self._current_settings
            limit = (int(settings.max_reply_chars) if settings is not None
                     and settings.max_reply_chars is not None
                     else self.config.group.max_reply_chars)
        if limit > 0 and len(cleaned) > limit:
            cleaned = cleaned[:limit].rstrip() + "……"
        return cleaned.strip()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def record(self, message: IncomingMessage, now: float) -> None:
        if not message.text:
            return
        if message.is_private:
            self.storage.touch_user(message.user_id, message.user_name, now)
        else:
            self.storage.ensure_profile(message.user_id, message.user_name, self.config.bot.default_mode)
            self.storage.set_display_name(message.user_id, message.user_name)
        self.storage.append_message(message.user_id, "user", message.text, message.group_id, now)

    def _remember_tool_already_handled(self, outcome: Outcome) -> bool:
        """本轮是否已经通过 remember 工具成功写入过记忆。"""
        for item in (outcome.tool_results or []):
            if item.get("name") == "remember" and item.get("ok"):
                return True
        return False

    def _remember_from_message(self, message: IncomingMessage, outcome: Outcome, now: float) -> None:
        if not self.config.memory.enabled or not self.config.memory.explicit_save:
            return
        # **模型已经用 remember 工具写过了，启发式就让路。**
        #
        # 这个函数是在调用模型**之前**跑的（顺序不能反：模型需要看到
        # 已保存的记忆），所以它没法提前知道模型会不会调工具。
        # 两条路径同时写就会出现重复——实测：
        #
        #     source='explicit'  "用户明确要求记住：我喜欢猫"   <- 这里写的
        #     source='tool'      "猫"                        <- 工具写的
        #
        # 一个"记住"请求存两条，既浪费额度又让用户看到重复的记忆。
        # 工具写的是模型**明确表达**要保存的内容，比启发式从原话里
        # 切出来的更准，所以让工具优先。
        if self._remember_tool_already_handled(outcome):
            return
        has_request = memory_mod.has_explicit_request(message.text)
        facts = memory_mod.extract(message.text, allow_sensitive=self.config.features.memory_sensitive_data)
        for fact in facts:
            if fact.kind == "explicit_rejected":
                outcome.memory_rejected.append("内容过短或包含敏感信息")
                continue
            self.storage.add_memory(message.user_id, fact.content, kind=fact.kind,
                                    source=fact.source, confidence=fact.confidence, now=now)
            outcome.memory_saved.append(fact.content)
            outcome.memory_saved_this_turn.append(fact.content)
            if fact.source == "explicit":
                outcome.memory_explicit.append(fact.content)
            if fact.kind == "nickname":
                match = re.search(r"「(.+?)」", fact.content)
                if match:
                    self.storage.set_nickname(message.user_id, match.group(1))
        if outcome.memory_saved:
            self.storage.enforce_memory_limit(message.user_id, self.config.memory.per_user_limit)
        # 用户明确要求记住、但提取器什么都没保存下来：标记为"需要如实说明"
        if has_request and not outcome.memory_saved and not outcome.memory_rejected:
            outcome.memory_pending_review = True

    def _apply_memory_feedback(self, reply_text: str, outcome: Outcome) -> str:
        """用户明确要求"记住"时，保证回复里有一句如实的确认（不假装保存）。

        规则：只有显式请求才追加反馈；追加内容放在回复末尾，不打断原本的语气。
        """
        # 只有"本轮用户明确要求记住"才追加确认。
        # 否则 /记忆 这类指令里回显的旧记忆（source 也是 explicit）会被误加"已经记下了"。
        if outcome.memory_saved_this_turn:
            count = len(outcome.memory_saved_this_turn)
            note = "（已经记下了）" if count == 1 else f"（已经记下 {count} 条）"
            if note not in reply_text:
                return f"{reply_text} {note}"
            return reply_text
        if outcome.memory_rejected and not outcome.memory_saved:
            note = "（这条包含敏感信息，或者内容太短，按规则没有保存）"
            if note not in reply_text:
                return f"{reply_text} {note}"
            return reply_text
        # 用户明确要求"记住"，但确实什么都没存下：模型很可能顺着接了句"记下了"。
        # 这里追加修正，避免出现"假装已保存"。
        if outcome.memory_pending_review and not outcome.memory_saved:
            note = "（补充说明：这条确实没有存进记忆，爱丽丝刚才那句是误以为已经保存了）"
            if note not in reply_text:
                return f"{reply_text} {note}"
        return reply_text

    def _effective_group_quota(self) -> int:
        """群级额度覆盖：/群设置 群额度 会写进群配置。"""
        if self._current_settings is not None and self._current_settings.quota_daily is not None:
            return int(self._current_settings.quota_daily)
        return self.config.quota.daily_group

    def _check_limits(self, message: IncomingMessage, now: float) -> Verdict:
        policy = self.config.private if message.is_private else self.config.group
        if message.is_private and not self.config.private.enabled:
            return Verdict(Decision.DISABLED, "私聊功能未启用")
        if not message.is_private and not self.config.group.enabled:
            return Verdict(Decision.DISABLED, "群聊功能未启用")
        # 群级频率覆盖（/群设置 每分钟上限）
        limit = policy.rate_limit_per_minute
        if (not message.is_private and self._current_settings is not None
                and self._current_settings.rate_limit_per_minute is not None):
            limit = int(self._current_settings.rate_limit_per_minute)
        verdict = check_rate_limit(self.storage, "rl_user", f"{message.user_id}:{message.scene_id}", limit, now)
        if not verdict.allowed:
            return verdict
        if not message.is_private and self.config.group.rate_limit_per_user_per_minute > 0:
            per_user = check_rate_limit(self.storage, "rl_group_user",
                                        f"{message.group_id}:{message.user_id}",
                                        self.config.group.rate_limit_per_user_per_minute, now)
            if not per_user.allowed:
                return per_user
        if not message.is_private and self.config.group.rate_limit_per_minute > 0:
            per_group = check_rate_limit(self.storage, "rl_group", message.group_id,
                                         self.config.group.rate_limit_per_minute, now)
            if not per_group.allowed:
                return per_group
        return check_quota(self.storage, self.config, message.user_id, message.is_private, now,
                           group_cap=self._effective_group_quota() if not message.is_private else None)

    def _seconds_since_bot_reply(self, group_id: str, now: float) -> Optional[float]:
        last = self.storage.kv_get(f"bot_last_reply:{group_id}")
        if not last:
            return None
        return max(0.0, now - float(last))

    def _seconds_since_own_message(self, user_id: str, group_id: str, now: float) -> Optional[float]:
        """这个人上一次在本群发言距今多久。

        注意：当前这条消息已经写进日志了，所以 rows[0] 就是"上一条"。
        即使只发过一条（日志里只有这一条），也能算出"刚刚说过"，不能返回 None，
        否则第一条之后的冷却判定会失效。
        """
        if not group_id:
            return None
        with self.storage._lock:
            rows = self.storage._conn.execute(
                "SELECT created_at FROM conversation_log WHERE group_id = ? AND user_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 2", (str(group_id), str(user_id))).fetchall()
        # 按 id DESC 取，rows[0] 才是"这个人上一句话"。
        # 早期写成 rows[-1]（较旧那条）且要求至少两条，导致冷却判定几乎从不生效。
        if not rows:
            return None
        return max(0.0, now - float(rows[0]["created_at"]))

    def _now_text(self, timestamp: Optional[float] = None) -> str:
        stamp = time.time() if timestamp is None else timestamp
        base = time.strftime(self.config.bot.time_format, time.localtime(stamp))
        hour = time.localtime(stamp).tm_hour
        if hour < 6:
            period = "深夜"
        elif hour < 12:
            period = "上午"
        elif hour < 18:
            period = "下午"
        else:
            period = "晚上"
        return f"{base}（{period}）"

    def _relative(self, seconds: float) -> str:
        if seconds < 60:
            return "刚刚"
        if seconds < 3600:
            return f"{int(seconds // 60)} 分钟前"
        if seconds < 86400:
            return f"{int(seconds // 3600)} 小时前"
        return f"{int(seconds // 86400)} 天前"

    def _summarize(self, history: Sequence[Dict[str, Any]], facts, message: IncomingMessage, profile) -> str:
        parts = []
        if history:
            lines = []
            for item in list(history)[-6:]:
                who = "用户" if item["role"] == "user" else "爱丽丝"
                snippet = str(item["content"]).replace("\n", " ")[:60]
                lines.append(f"- {who}：{snippet}")
            parts.append("最近对话片段：\n" + "\n".join(lines))
        if facts:
            parts.append(f"该用户的长期记忆共 {len(facts)} 条，已在上方列出。")
        # 注意：profile.message_count 在生成回复之前就已经把"这一条"算进去了
        # （record() -> touch_user 会自增），所以当前这次就是第 message_count 次。
        if message.is_private and profile.message_count > 0:
            parts.append(f"这是你们的第 {profile.message_count} 次互动。")
        return "\n".join(parts) if parts else "（这是你们第一次对话）"

    def _group_context(self, message: IncomingMessage) -> str:
        record = self.storage.get_group(message.group_id)
        recent = self.storage.recent_group_messages(message.group_id, 10)
        lines = []
        now = message.timestamp or time.time()
        for item in recent:
            if item["role"] != "user":
                continue
            snippet = str(item["content"]).replace("\n", " ")[:50]
            lines.append(f"- [{self._relative(now - float(item['created_at']))}] {item['user_id']}：{snippet}")
        body = "最近群内公开消息（仅当前群可见，禁止外传）：\n" + "\n".join(lines) if lines else "（本群暂时还没有可参考的历史消息）"
        if record and record.rules:
            body += f"\n本群专用规则：{record.rules}"
        return body

    def _group_config_text(self, record, message: IncomingMessage) -> str:
        if message.is_private:
            return "（私聊场景，无群配置）"
        lines = [
            f"群号：{message.group_id}",
            f"访问模式：{self.config.group.access}",
            f"本群状态：{'启用' if (record is None or record.enabled) else '停用'}",
            f"触发方式：{', '.join(self.config.group.triggers)}",
            f"回复长度上限：{self.config.group.max_reply_chars} 字",
        ]
        if record and record.keywords:
            lines.append("本群额外关键词：" + "、".join(record.keywords))
        # V3：把群级开关也写进上下文。
        # 这里必须**按 message.group_id 现查**，不能复用实例上的 _current_settings——
        # 后者只在 handle() 里赋值，直接调用 build_messages（测试/诊断）时会停留在上一个群，
        # 造成"群 A 的配置渲染到群 B"（实测被 test_settings_do_not_leak_across_groups 抓到）。
        settings = self.storage.get_group_settings(message.group_id)
        if settings is not None:
            flags = []
            for label, value, fallback in (
                ("主动参与", settings.proactive, self.config.features.proactive_group),
                ("联网检索", settings.search, self.config.features.web_search),
                ("图片", settings.image, self.config.features.image),
                ("语音", settings.voice, self.config.features.voice),
            ):
                flags.append(f"{label}={'开' if (fallback if value is None else value) else '关'}")
            if settings.quota_daily is not None:
                flags.append(f"本群每日额度={settings.quota_daily}")
            lines.append("本群功能开关：" + "、".join(flags))
        return "\n".join(lines)

    def _analyze_images(self, message: IncomingMessage, outcome: Outcome) -> None:
        """把用户发来的图片交给视觉模型，结果留给提示词用。

        **为什么它是"感知"步骤而不是一个工具。**
        工具是模型自己决定要不要调的（比如要不要联网查）。但"用户发了图"
        是**事实**——模型没法"决定不看"，它得先知道图里有什么，才有得选。
        所以这里直接在当前轮算好，注入上下文。

        三种情况都要有明确行为，不能静默：

          1. 用户没发图 -> 什么都不做
          2. 发了图但没有视觉模型 -> 注入一段"你看不到图"的说明，
             让模型如实告诉用户，而不是根据用户的话猜图里是什么
          3. 额度不足 -> 同样注入说明，并说明是额度问题

        这一步**不阻塞回复**：视觉调用失败时，对话照常进行，
        只是模型会知道"这次没看到图"。
        """
        self._pending_vision = None
        if not message.has_images:
            return

        images = list(message.images)
        outcome.debug["images_received"] = len(images)

        if self.vision_client is None:
            # 没配视觉模型：不调用，但要让模型知道"有图且我看不到"。
            reason = "没有配置视觉模型"
            if self.config.vision.enabled:
                reason = "视觉模型没有配置完整（检查 VISION_BASE_URL / API_KEY / MODEL）"
            self._pending_vision = vision_mod.VisionResult(False, error=reason)
            outcome.debug["vision"] = {"ok": False, "reason": reason,
                                       "images": len(images)}
            return

        if not self._feature_allowed(message, "vision"):
            reason = self._last_block_reason or "本群或额度不允许"
            self._pending_vision = vision_mod.VisionResult(
                False, error="当前不允许使用图片理解（%s）" % reason)
            outcome.debug["vision"] = {"ok": False, "reason": reason,
                                       "blocked": True, "images": len(images)}
            return

        started = time.time()
        try:
            result = self.vision_client.describe(images)
        except Exception as exc:
            # 视觉是可选增强，任何异常都不能影响主流程
            result = vision_mod.VisionResult(
                False, error="%s: %s" % (type(exc).__name__, str(exc)[:120]))
        self._pending_vision = result
        outcome.debug["vision"] = {
            "ok": result.ok,
            "images": result.images_seen,
            "skipped": result.skipped,
            "elapsed": round(result.elapsed or (time.time() - started), 2),
            "model": result.model,
            "error": result.error,
        }
        if result.ok:
            consume_feature_quota(self.storage, self.config, "vision",
                                  message.user_id, message.group_id,
                                  message.is_private)

    def _vision_block(self, message: IncomingMessage) -> str:
        """给提示词用的图片理解区块。没有图就返回空串。"""
        if not message.has_images:
            return ""
        result = getattr(self, "_pending_vision", None)
        if result is None:
            # 理论上走不到：handle 里必然先调 _analyze_inputs。
            # 但真发生了也要如实说，不能让模型以为没图。
            return "（用户发了图片，但这次没有拿到识别结果。）"
        if not result.ok and not self.config.vision.tell_model_when_unavailable:
            return ""
        character = characters_mod.get(self.current_mode(message))
        return vision_mod.render_for_model(result, mode_is_roleplay=character.is_roleplay)

    def _analyze_audio(self, message: IncomingMessage, outcome: Outcome) -> None:
        """把用户发来的语音转成文字，留给提示词用。

        **这条链路以前是完全断的。** `asr.py` 早就有识别能力，
        但它只被用在"转写 B 站视频音轨"上；用户直接发的语音消息
        在适配层就没被抽出来，所以模型收到的是一条空文本，
        表现就是"发语音它不回"或者"回一句莫名其妙的话"。

        和图片一样，这是**感知**步骤而不是工具：
        "用户说了话"是事实，模型没法决定不听。

        转写结果要连同**质量警告**一起注入。ASR 在噪声上会输出
        "通顺但完全错误"的文本，光看文字看不出来——`quality_warning()`
        就是为这种情况写的，这里必须用上，否则等于把错误信息
        当成用户原话交给模型。
        """
        self._pending_transcript = None
        self._pending_video = None
        self._pending_video_frames = None
        self._pending_video_transcript = ""
        self._pending_video_transcript_note = ""
        if not message.has_audios:
            return

        audios = list(message.audios)
        outcome.debug["audios_received"] = len(audios)

        if self.asr_client is None or not getattr(self.asr_client, "available", False):
            reason = "没有可用的语音识别"
            self._pending_transcript = asr_mod.Transcription(False, error=reason)
            outcome.debug["asr"] = {"ok": False, "reason": reason, "audios": len(audios)}
            return

        ref = audios[0]                       # 一条消息里的语音只认第一条
        blob = ref.read_bytes() if hasattr(ref, "read_bytes") else b""
        if not blob:
            reason = "语音读取失败"
            self._pending_transcript = asr_mod.Transcription(False, error=reason)
            outcome.debug["asr"] = {"ok": False, "reason": reason, "audios": len(audios)}
            return

        if not self._feature_allowed(message, "asr_voice"):
            reason = self._last_block_reason or "本群或额度不允许"
            self._pending_transcript = asr_mod.Transcription(
                False, error="当前不允许识别语音（%s）" % reason)
            outcome.debug["asr"] = {"ok": False, "reason": reason, "blocked": True,
                                    "audios": len(audios)}
            return

        started = time.time()
        try:
            result = self.asr_client.transcribe(blob, getattr(ref, "filename", "voice.m4a"))
        except Exception as exc:
            result = asr_mod.Transcription(
                False, error="%s: %s" % (type(exc).__name__, str(exc)[:120]))
        self._pending_transcript = result
        text = (result.text or "").strip()
        outcome.debug["asr"] = {
            "ok": result.ok,
            "chars": len(text),
            "seconds": round(result.seconds or 0.0, 1),
            "elapsed": round(time.time() - started, 2),
            "audios": len(audios),
            "error": result.error,
        }
        if result.ok and text:
            consume_feature_quota(self.storage, self.config, "asr_voice",
                                  message.user_id, message.group_id,
                                  message.is_private)

    def _transcript_block(self, message: IncomingMessage) -> str:
        """给提示词用的语音转写区块。没有语音就返回空串。"""
        if not message.has_audios:
            return ""
        result = getattr(self, "_pending_transcript", None)
        if result is None:
            return "（用户发了语音，但这次没有拿到识别结果。）"

        if not result.ok or not (result.text or "").strip():
            why = result.error or "识别没有返回内容"
            return ("（用户这次发的是**语音**，但你没有听到内容——%s。"
                    "请如实告诉用户你没能听清，可以请他再说一遍或者打字，"
                    "**不要猜测他说了什么**。）" % why)

        # 质量不可靠时必须说明。ASR 在噪声/背景音乐上会输出
        # 通顺但完全错误的文本，用户看不出来，模型更看不出来。
        warning = asr_mod.quality_warning(
            result.text, seconds=result.seconds,
            avg_logprob=result.avg_logprob, no_speech_prob=result.no_speech_prob)
        head = "（以下是用户刚发的语音转写成的文字，可能有错别字或同音字错误）"
        body = result.text.strip()
        if warning:
            return head + "\n" + body + "\n注意：" + warning
        return head + "\n" + body

    def _analyze_video(self, message: IncomingMessage, outcome: Outcome) -> None:
        """把用户发来的视频拆成画面与声音，分别交给视觉和 ASR。

        **视频不需要新的模型额度**：抽出来的帧走 `vision`，
        音轨转写走 `asr_voice`，两条链路各自的门禁与额度自动生效，
        护栏（"看不到就说看不到"）也自动继承。

        三个失败点要分开报，因为它们对用户的意义不同：
          * 抽不出帧 -> 这段视频的画面没法看
          * 没有音轨 -> 正常情况（纯画面视频），不是错误
          * 转写失败 -> 有声音但没听清
        """
        self._pending_video = None
        self._pending_video_frames = None
        self._pending_video_transcript = ""
        self._pending_video_transcript_note = ""
        if not message.has_videos:
            return

        videos = list(message.videos)
        outcome.debug["videos_received"] = len(videos)
        ref = videos[0]                       # 一条消息里的视频只认第一条
        data = ref.read_bytes() if hasattr(ref, "read_bytes") else b""
        if not data:
            self._pending_video = video_mod.VideoAnalysis(
                False, error="视频读取失败或超过体积上限")
            outcome.debug["video"] = {"ok": False, "reason": "读取失败"}
            return

        exe = ""
        try:
            exe = ffmpeg_mod.find_ffmpeg(self.config)
        except Exception:
            exe = ""
        if not exe:
            self._pending_video = video_mod.VideoAnalysis(
                False, error="没有可用的 ffmpeg，无法解析视频")
            outcome.debug["video"] = {"ok": False, "reason": "没有 ffmpeg"}
            return

        started = time.time()
        try:
            analysis = video_mod.analyze(
                data, exe,
                frame_count=int(getattr(self.config.vision, "video_frames", 3) or 3),
                duration_hint=float(getattr(ref, "duration", 0.0) or 0.0))
        except Exception as exc:
            analysis = video_mod.VideoAnalysis(
                False, error="%s: %s" % (type(exc).__name__, str(exc)[:120]))
        self._pending_video = analysis

        # 画面：复用图片理解那条链路。它有自己的额度与"看不到就说"的护栏，
        # 所以这里不重复判断，直接调即可（build 出来的 ImageRef 走同一条路）。
        frame_result = None
        if analysis.frames and self.vision_client is not None:
            if self._feature_allowed(message, "vision"):
                try:
                    frame_result = self.vision_client.describe(
                        [vision_mod.ImageRef(blob=f, kind="image")
                         for f in analysis.frames])
                except Exception as exc:
                    frame_result = vision_mod.VisionResult(
                        False, error="%s: %s" % (type(exc).__name__, str(exc)[:120]))
                self._pending_video_frames = frame_result
                if frame_result.ok:
                    consume_feature_quota(self.storage, self.config, "vision",
                                          message.user_id, message.group_id,
                                          message.is_private)
            else:
                self._pending_video_frames = vision_mod.VisionResult(
                    False, error=self._last_block_reason or "不允许使用图片理解")
        elif analysis.frames:
            self._pending_video_frames = vision_mod.VisionResult(
                False, error="没有配置视觉模型")

        # 声音：复用语音那条链路
        if analysis.audio:
            if self.asr_client is not None and getattr(self.asr_client, "available", False):
                if self._feature_allowed(message, "asr_voice"):
                    try:
                        tr = self.asr_client.transcribe(analysis.audio, "video_audio.wav")
                    except Exception as exc:
                        tr = asr_mod.Transcription(
                            False, error="%s: %s" % (type(exc).__name__, str(exc)[:120]))
                    if tr.ok and (tr.text or "").strip():
                        self._pending_video_transcript = tr.text.strip()
                        self._pending_video_transcript_note = asr_mod.quality_warning(
                            tr.text, seconds=tr.seconds, avg_logprob=tr.avg_logprob,
                            no_speech_prob=tr.no_speech_prob)
                        consume_feature_quota(self.storage, self.config, "asr_voice",
                                              message.user_id, message.group_id,
                                              message.is_private)
                else:
                    self._pending_video_transcript_note = (
                        self._last_block_reason or "不允许识别语音")

        outcome.debug["video"] = {
            "ok": analysis.ok,
            "frames": len(analysis.frames),
            "frames_ok": bool(getattr(self._pending_video_frames, "ok", False)),
            "has_audio": analysis.has_audio,
            "transcript_chars": len(self._pending_video_transcript),
            "elapsed": round(time.time() - started, 2),
            "error": analysis.error,
            "note": analysis.note,
        }

    def _video_block(self, message: IncomingMessage) -> str:
        """给提示词用的视频区块。没有视频就返回空串。"""
        if not message.has_videos:
            return ""
        analysis = getattr(self, "_pending_video", None)
        if analysis is None:
            return ("（用户发了视频，但这次没有拿到解析结果。）")
        return video_mod.render_for_model(
            analysis,
            frame_result=getattr(self, "_pending_video_frames", None),
            transcript=getattr(self, "_pending_video_transcript", ""),
            transcript_note=getattr(self, "_pending_video_transcript_note", ""))

    def _analyze_inputs(self, message: IncomingMessage, outcome: Outcome) -> None:
        """本轮的**多模态输入**统一入口：图片、语音、视频。

        收敛成一个入口是为了加新模态时不用在多处改调用点——
        这轮加视频就只动了这里和 `handle` 里的一行。
        """
        self._analyze_images(message, outcome)
        self._analyze_audio(message, outcome)
        self._analyze_video(message, outcome)

    def _maybe_search(self, message: IncomingMessage, outcome: Outcome) -> bool:
        """时效性问题先检索，把结果作为上下文注入。

        返回 True 表示"已经给出了澄清回复，调用方应当直接结束本轮"，不要再生成回复。

        这条路径不依赖模型主动调用工具——很多模型在角色扮演模式下
        并不会去调工具，而用户问"今天有什么活动"时我们必须真的查过。
        """
        if not self.config.search.auto_search:
            return False

        # V3：群级开关与"按能力独立额度"的门禁——未授权就不调用高成本工具、也不消耗额度
        if not self._feature_allowed(message, "search"):
            outcome.debug["search_skipped"] = self._last_block_reason
            return False

        # 学术意图走独立链路（文献库），不要丢给网页搜索
        if self.academic_service is not None and is_academic_query(message.text):
            try:
                academic = self.academic_service.search(
                    message.text, limit=self.config.search.max_results)
            except Exception as exc:
                academic = AcademicOutcome(False, message.text, error=f"{type(exc).__name__}: {exc}")
            outcome.academic = academic
            self._pending_academic = academic
            self._pending_search_owner = (str(message.user_id), message.scene_id)
            outcome.debug["academic"] = {
                "provider": academic.provider, "ok": academic.ok,
                "count": len(academic.results), "elapsed": academic.elapsed,
                "rate_limited": academic.rate_limited, "error": academic.error,
            }
            self._consume_feature(message, "search", message.timestamp)
            return False

        if self.search_service is None:
            return False
        if not needs_search(message.text):
            return False

        # 用户追问时常省略主语（"现在有什么活动？"），直接拿去搜会跑到别的游戏上。
        # 先尝试用上文补全话题；补不出来就如实询问，而不是搜索后拿到一堆错信息。
        query = self._enrich_query(message)
        if not query:
            outcome.debug["search_skipped"] = "查询缺少话题，且上文没有可用语境"
            outcome.reply = ("想帮阿岚查最新的消息，但不太确定问的是哪件事。\n"
                             "可以带上对象再说一次吗？比如「蔚蓝档案现在有什么活动」。")
            outcome.decision = "replied"
            outcome.reason = "缺少检索话题"
            return True
        try:
            result = self.search_service.search(query, limit=self.config.search.max_results)
        except Exception as exc:
            result = SearchOutcome(False, query, error=f"{type(exc).__name__}: {exc}")
        outcome.search = result
        outcome.search_query = result.query
        # build_messages 在生成回复时才执行，用实例字段把本轮结果带过去
        self._pending_search = result
        self._pending_search_owner = (str(message.user_id), message.scene_id)
        outcome.debug["search"] = {
            "provider": result.provider, "ok": result.ok, "count": len(result.results),
            "elapsed": result.elapsed, "error": result.error,
        }
        # 真的调用了高成本能力才扣额度（失败也算调用过一次，因为成本已经产生）
        self._consume_feature(message, "search", message.timestamp)
        return False

    # ------------------------------------------------------------------
    # V4：多媒体
    # ------------------------------------------------------------------
    # 用户明确要求语音/图片的说法
    # 语音意图。同样用正则：实测"用温柔的声音安慰我"这种自然说法
    # 不含"用语音"/"说句话"等任何固定子串，靠精确匹配会整条漏掉。
    #
    # 实测又漏了两种很常见的说法（用户截图那次就是栽在这）：
    #   "把语音发给我"  —— 顺序是【语音→发】，而原模式只认【发→语音】
    #   "你来唱给我听"  —— 唱歌请求，原来一条都不认
    # 所以补上"语音…发/来"的反向顺序，以及"唱"这一整类请求。
    _VOICE_PATTERNS = (
        r"用[^，。！？!?]{0,8}(声音|语气|嗓音)",
        r"语[音气][说回]",
        # 正向：【发】…【语音】
        r"发[^，。！？!?]{0,3}语音",
        # 反向：【语音】…【发/来/给】 —— "把语音发给我"属于这一类
        r"语音[^，。！？!?]{0,4}(发|来|给)",
        r"来[一]?[段条]语音",
        # 唱歌请求："唱给我听""唱一首""唱一段""唱一个"
        r"唱[^，。！？!?]{0,4}(给我听|一首|一段|一个|个|首|段|吧)",
        r"唱给我",
        r"说句话",
        r"读出来",
        r"念一下",
        r"voice",
    )

    @classmethod
    def _wants_voice(cls, text: str) -> bool:
        return any(re.search(pattern, text or "", re.IGNORECASE)
                   for pattern in cls._VOICE_PATTERNS)
    # 图片意图用正则：精确子串匹配抓不到"给我来一张蔚蓝档案的图"这种自然说法
    # （实测该说法因为没有"来一张图"这三个连续字而被漏判）
    _IMAGE_PATTERNS = (
        r"来s*[一1]?s*张[^，。！？!?]{0,14}图",
        r"画s*[一1]?s*(张|个|幅|只)",
        r"生成s*[一1]?s*张?[^，。！？!?]{0,14}图",
        r"配s*[一1]?s*张图",
        r"要s*[一1]?s*张[^，。！？!?]{0,14}图",
        r"发s*[一1]?s*张[^，。！？!?]{0,14}图",
        r"给我s*[一1]?s*张",
        r"draw",
    )

    @classmethod
    def _wants_image(cls, text: str) -> bool:
        return any(re.search(pattern, text or "", re.IGNORECASE)
                   for pattern in cls._IMAGE_PATTERNS)

    def _media_intent(self, message: IncomingMessage) -> str:
        """告诉模型"本次回复会附带什么媒体"。

        为什么需要：媒体是在文本生成**之后**才附上的，模型不知道，
        于是会出现"语音还没解锁"这种回复——而语音其实已经发出去了。
        实测就是这么撞上的。
        """
        if self.media_library is None:
            return "本次不附带媒体。"
        if self._wants_voice(message.text):
            if self._media_allowed(message, "voice") and self.tts_client is not None:
                hint = ("本次回复会**以语音形式发送**（用文本转语音合成）。"
                        "请只给一小段口语化、适合朗读的内容（建议不超过两句话），"
                        "**不要**讨论语音功能是否可用，也不要提合成音之类的实现细节。")
                # 用户明确点了语气，文字也得跟上，否则语音和文字会不一致
                tone = emotion_mod.detect_explicit(message.text)
                if tone:
                    hint += (f"\n用户要求用「{emotion_mod.LABELS.get(tone, tone)}」的语气，"
                             "请让文字本身也带上这种情绪，但**不要**写出任何情感标签或括号注释。")
                return hint
            return "用户要求了语音，但语音当前不可用；请如实说明并给出文字替代。"
        if self._wants_image(message.text):
            if self._media_allowed(message, "image"):
                return "用户要求了图片；若素材库有合适图片会附上，否则可能生成或如实说明做不到。"
        if self._media_allowed(message, "sticker"):
            return "可能附带一个与话题匹配的 QQ 表情包；不要声称一定发送。"
        return "本次不附带媒体。"

    def _media_allowed(self, message: IncomingMessage, feature: str) -> bool:
        """多媒体是否可用：全局开关 + 群级开关 + 该项额度。"""
        label = {"sticker": "表情包", "image": "图片", "voice": "语音"}.get(feature, feature)
        settings = self._current_settings if not message.is_private else None
        if feature in ("image", "voice"):
            if settings is not None and getattr(settings, feature, None) is False:
                self._last_block_reason = f"本群已关闭{label}"
                return False
            if not getattr(self.config.features, feature, False):
                self._last_block_reason = f"全局已关闭{label}"
                return False
            # 图片/语音都是高成本能力，走独立额度
            verdict = check_feature_quota(self.storage, self.config, feature, message.user_id,
                                          message.group_id, message.is_private)
            if not verdict.allowed:
                self._last_block_reason = verdict.reason
                return False
        if feature == "sticker":
            if not self.config.features.emoji or not self.config.media.sticker_enabled:
                self._last_block_reason = "表情包未启用"
                return False
            if settings is not None and getattr(settings, "image", None) is False:
                # 群级"图片"开关同时约束表情包（都属于媒体）
                self._last_block_reason = "本群已关闭媒体"
                return False
        self._last_block_reason = ""
        return True

    def _media_rate_ok(self, message: IncomingMessage, now: float) -> bool:
        """表情包的发送间隔限制——V4 要求"不要每条消息都发送图片"。"""
        interval = int(getattr(self.config.media, "sticker_min_interval", 0) or 0)
        if interval <= 0:
            return True
        scope = "media_sticker"
        key = f"{scope}:{'dm' if message.is_private else 'group'}:{message.scene_id}"
        last = self.storage.kv_get(key, 0) or 0
        try:
            last = float(last)
        except (TypeError, ValueError):
            last = 0.0
        if now - last < interval:
            return False
        self.storage.kv_set(key, now)
        return True

    def _attach_media(self, message: IncomingMessage, outcome: Outcome,
                      reply_text: str, now: float) -> None:
        """决定随回复附加哪个媒体。最多一个。

        优先级遵循 V4：优先素材库里已有且与话题匹配的，
        没有合适素材且用户明确要求时才考虑 AI 生成。
        """
        if self.media_library is None:
            return
        combined = f"{message.text} {reply_text}"

        # ① 用户明确要求语音 -> 合成语音（最优先，因为是显式请求）
        if self._wants_voice(message.text):
            if self._voice_attachment(message, outcome, reply_text, now):
                return

        # ② 话题匹配的现成素材（表情包优先，其次图片）
        if self._media_allowed(message, "sticker") and self._media_rate_ok(message, now):
            probability = float(getattr(self.config.media, "sticker_probability", 0) or 0)
            if self.rng.random() <= probability:
                item = self.media_library.pick(message.text, kind="sticker", rng=self.rng)
                if item is not None:
                    outcome.media.append(MediaAttachment(
                        kind="sticker", path=str(self.media_library.path_of(item)),
                        item_id=item.id, origin=item.origin, desc=item.desc))
                    outcome.debug["media"] = f"sticker:{item.id}"
                    return

        # ③ 用户明确要求图片
        wants_image = self._wants_image(message.text)
        if not wants_image:
            return
        if not self._media_allowed(message, "image"):
            reason = self._last_block_reason
            outcome.debug["media_blocked"] = reason
            # 用户明确要了图却被拦下（额度/开关）——如实说明，不假装已发
            self._note_media_failure(outcome, "图片", reason or "当前不可用")
            return
        item = self.media_library.pick(combined, kind="image", rng=self.rng)
        if item is not None:
            outcome.media.append(MediaAttachment(
                kind="image", path=str(self.media_library.path_of(item)),
                item_id=item.id, origin=item.origin, desc=item.desc))
            outcome.debug["media"] = f"image:{item.id}"
            return
        # 没有合适素材 -> 才考虑 AI 生成
        self._generate_image(message, outcome, reply_text, now)

    def _voice_attachment(self, message: IncomingMessage, outcome: Outcome,
                          reply_text: str, now: float) -> bool:
        if not self._media_allowed(message, "voice"):
            reason = self._last_block_reason
            outcome.debug["media_blocked"] = reason
            self._note_media_failure(outcome, "语音", reason or "当前不可用")
            return False
        if self.tts_client is None or not getattr(self.tts_client, "available", False):
            outcome.debug["media_blocked"] = "未配置语音合成"
            self._note_media_failure(outcome, "语音", "未配置语音合成")
            return False
        limit = int(getattr(self.config.media, "tts_max_chars", 120) or 120)
        spoken = (reply_text or "").strip()[:limit]
        if not spoken:
            return False

        # 情感：从用户要求与回复正文推断，再交给 TTS 表达。
        # 推断用确定性规则（见 emotion.py），不让模型自报情感——
        # 模型自报的标签很容易漏进最终回复里。
        emotion = ""
        if getattr(self.config.media, "tts_emotion_enabled", True):
            guess = emotion_mod.infer(message.text, reply_text)
            emotion = guess.emotion
            outcome.debug["emotion"] = emotion_mod.describe(guess)

        try:
            result = self.tts_client.synthesize(spoken, emotion)
        except TypeError:
            # 兼容不带 emotion 参数的旧客户端
            result = self.tts_client.synthesize(spoken)
        if not result.ok:
            # V4：工具失败必须说明，且不能编造结果
            outcome.debug["media_blocked"] = f"语音合成失败：{result.error}"
            self._note_media_failure(outcome, "语音", result.error)
            return False
        # 情感是否真的表达出来了，要如实记录（免费端点做不到）
        if emotion:
            applied = getattr(result, "emotion_applied", False)
            outcome.debug["emotion_applied"] = applied
            note = getattr(result, "emotion_note", "")
            if not applied and note:
                outcome.debug["emotion_note"] = note

        # 降级提示：配置的是克隆音色，但实际用了备用声音。
        # 为什么必须说：FallbackTTS 在克隆服务不通时会回退到免费 TTS，
        # 这时 ok=True、error 为空，用户完全看不出声音换了人——
        # 会以为听到的是爱丽丝，其实是另一个声音。
        # 和"媒体失败要如实说明"是同一条原则。
        provider = str(getattr(result, "provider", "") or "")
        if "(备用)" in provider:
            outcome.debug["voice_fallback"] = provider
            if self._should_note_fallback(message.user_id, now):
                outcome.reply = (outcome.reply or "") + (
                    "\n\n（克隆音色服务没连上，这次先用备用声音念的。）")

        path = self._store_generated("voice", result.data,
                                     f".{getattr(self.config.media, 'tts_format', 'mp3')}",
                                     f"语音：{spoken[:20]}", tags=[])
        if path is None:
            return False
        consume_feature_quota(self.storage, self.config, "voice", message.user_id,
                              message.group_id, message.is_private, now)
        outcome.media.append(MediaAttachment(kind="voice", path=str(path),
                                             origin="generated", desc=spoken[:20]))
        outcome.debug["media"] = "voice:generated"
        return True

    def _generate_image(self, message: IncomingMessage, outcome: Outcome,
                        reply_text: str, now: float) -> None:
        if self.image_client is None or not getattr(self.image_client, "available", False):
            # 与语音路径保持一致：用户明确要了图却给不出，就如实说明（V4 要求不假装成功）
            outcome.debug["media_blocked"] = "未配置图片生成，且素材库没有匹配图片"
            self._note_media_failure(outcome, "图片", "未配置图片生成，素材库里也没有匹配的图")
            return
        # 生成用的描述：优先用户原话，避免把整段回复塞进去
        prompt = (message.text or "").strip()[:200]
        result = self.image_client.generate(prompt)
        if not result.ok:
            outcome.debug["media_blocked"] = f"图片生成失败：{result.error}"
            self._note_media_failure(outcome, "图片", result.error)
            return
        # 存成 kind="image" + origin="generated"。
        # 早期存成独立的 kind="generated" 会导致它不出现在素材清单里，
        # 既看不见也无法被再次选中——origin 字段已经足够表达"这是生成的"。
        path = self._store_generated("image", result.data, ".png",
                                     f"生成图：{prompt[:20]}", tags=[])
        if path is None:
            return
        consume_feature_quota(self.storage, self.config, "image", message.user_id,
                              message.group_id, message.is_private, now)
        # origin=generated：V4 要求不能把 AI 生成图说成官方素材
        outcome.media.append(MediaAttachment(kind="image", path=str(path),
                                             origin="generated", desc=prompt[:20]))
        outcome.debug["media"] = "image:generated"

    def _should_note_fallback(self, user_id: str, now: float, interval: float = 1800.0) -> bool:
        """降级提示按用户限频，避免每条消息都刷同一句。

        用 kv 存上次提示时间。提示本身是重要的（声音换了人），
        但每条都提醒就成了噪音——所以给一个时间窗。
        """
        key = f"voice_fallback_noted:{user_id}"
        try:
            last = self.storage.kv_get(key)
        except Exception:
            return True
        try:
            last_at = float(last) if last is not None else 0.0
        except (TypeError, ValueError):
            last_at = 0.0
        if now - last_at < interval:
            return False
        try:
            self.storage.kv_set(key, now)
        except Exception:
            pass
        return True

    # 这个装饰器原本在 _should_note_fallback 上面，插入新方法时被"抢"走了，
    # 导致 _should_note_fallback 变成静态方法、self 参数错位。
    @staticmethod
    def _note_media_failure(outcome: Outcome, what: str, reason: str) -> None:
        """工具失败时在回复里如实说明，并提供文字替代。

        V4 明确要求："说明没有成功完成。提供文字替代方案。不要编造工具结果。"
        所以这里直接在回复末尾追加一句，而不是留给模型自由发挥
        （模型很容易写成"图片已发送"）。
        """
        note = f"\n\n（{what}没能生成成功：{reason}。先用文字说给你听。）"
        outcome.reply = (outcome.reply or "") + note
        outcome.debug["media_failure_note"] = what

    def _store_generated(self, kind: str, data: bytes, suffix: str,
                         desc: str, tags) -> Optional[str]:
        """把 AI 生成物写成**临时文件**，不登记进素材库。

        为什么要这样（实测踩到的坑）：
        早期实现把生成物 add() 进素材库，于是
        1) 生成物无限堆积，磁盘和 manifest 持续增长；
        2) 它会被当成"已有素材"列进提示词，里面还塞着上一轮的回复原文，
           等于把模型自己过去的输出又喂回去，既浪费上下文又干扰判断。
        生成物是一次性的（语音念的是某一句特定的话，本来也不可复用），
        所以只落盘、不入库。
        """
        # 空数据不要落盘：落盘后会被当作音频发出去（CQ:record），
        # 用户收到的是一个坏文件，不如直接不发。
        if not data:
            self._last_block_reason = "生成的音频为空"
            return None
        try:
            directory = Path(self.config.media.path).expanduser().absolute() / "transient"
            directory.mkdir(parents=True, exist_ok=True)
            self._prune_transient(directory)
            name = f"{kind}_{uuid.uuid4().hex[:10]}{suffix}"
            target = directory / name
            target.write_bytes(data)
            return str(target)
        except Exception as exc:
            self._last_block_reason = f"生成物写入失败：{exc}"
            return None

    @staticmethod
    def _prune_transient(directory: Path, keep_hours: int = 6, keep_max: int = 50) -> None:
        """清理旧的临时生成物，避免磁盘无限增长。"""
        try:
            files = sorted((p for p in directory.iterdir() if p.is_file()),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        cutoff = time.time() - keep_hours * 3600
        for index, path in enumerate(files):
            try:
                if index >= keep_max or path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def _asr_guard(self) -> bool:
        """语音识别的额度闸门（供检索层回调）。

        按时长计费，所以用"每日总次数"限制，并且**在真正调用前**扣减——
        被拒时返回 False，检索层会跳过识别并如实说明。
        """
        try:
            verdict = check_asr_quota(self.storage, self.config)
        except Exception:
            return False
        if not verdict.allowed:
            return False
        try:
            consume_asr_quota(self.storage)
        except Exception:
            pass
        return True

    def _feature_allowed(self, message: IncomingMessage, feature: str) -> bool:
        """判断本轮是否允许使用某能力：先看群级/全局开关，再看该能力自己的额度。

        V3 要求"额度不足时不要继续调用高成本工具，也不要假装调用成功"，
        所以这里直接拦在调用之前，并记下原因供 debug 与提示词使用。
        """
        label = {"search": "联网检索", "image": "图片", "voice": "语音"}.get(feature, feature)
        settings = self._current_settings if not message.is_private else None
        if settings is not None and feature in ("search", "image", "voice"):
            if getattr(settings, feature, None) is False:
                self._last_block_reason = f"本群已关闭{label}"
                return False
        if feature == "search" and not self.config.features.web_search:
            self._last_block_reason = "全局已关闭联网检索"
            return False
        if feature == "image" and not self.config.features.image:
            self._last_block_reason = "全局已关闭图片能力"
            return False
        if feature == "voice" and not self.config.features.voice:
            self._last_block_reason = "全局已关闭语音能力"
            return False
        verdict = check_feature_quota(self.storage, self.config, feature, message.user_id,
                                      message.group_id, message.is_private)
        if not verdict.allowed:
            self._last_block_reason = verdict.reason
            return False
        self._last_block_reason = ""
        return True

    def _consume_feature(self, message: IncomingMessage, feature: str, now: float) -> None:
        consume_feature_quota(self.storage, self.config, feature, message.user_id,
                              message.group_id, message.is_private, now)

    def _recent_topic(self, message: IncomingMessage) -> str:
        """从最近的对话里找"当前在聊什么话题"。

        用户的追问常常省略主语，这里从最近几轮消息里找一个专名补进查询。
        """
        history = self.storage.recent_messages(
            message.user_id, 6, group_id=message.group_id if not message.is_private else "",
            exclude_current=True)
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            text = str(item.get("content") or "")
            if len(text) < 4 or text.startswith("/"):
                continue
            # 注意：不能用"按空格切词"来挖话题——中文原文没有空格，
            # 整句会被当成一个词，什么都挖不到（这是最初版本的缺陷）。
            topic = search_mod_mine_topic(text, exclude=message.text or "")
            if topic:
                return topic
        return ""

    def _enrich_query(self, message: IncomingMessage) -> str:
        """补全检索词；补不出来返回空串，表示"这次不要搜"。"""
        text = message.text or ""
        if search_mod_has_topic(text):
            return text
        topic = self._recent_topic(message)
        if topic:
            return f"{topic} {text}"
        return ""

    def _search_block(self, message: IncomingMessage, mode: str) -> str:
        """给提示词用的检索结果区块。"""
        # 学术检索结果优先展示（本轮走的是文献链路）
        academic = getattr(self, "_pending_academic", None)
        if academic is not None:
            owner = getattr(self, "_pending_search_owner", None)
            if owner is None or (str(owner[0]) == str(message.user_id)
                                 and str(owner[1]) == str(message.scene_id)):
                note = ("（这是学术文献检索结果。请给出标题与链接；"
                        "不要编造卷号、页码、引用次数。若限流导致失败，如实说明。）")
                return note + "\n" + format_academic(academic)

        if self.search_service is None:
            return "（当前没有接入联网检索，涉及最新信息时必须说明无法确认）"
        result = getattr(self, "_pending_search", None)
        if result is None:
            return "（本次没有进行检索。若问题涉及最新信息而你无法确认，请直接说明并建议检索）"
        # 检索结果只对"发起检索的那个用户、且同一场景"可见。
        # 注意必须带上 user_id：私聊的 scene_id 都是 "private"，
        # 只按场景判定会让 A 的检索结果出现在 B 的私聊里（test_search_result_not_leaked_across_users 抓到的）。
        owner = getattr(self, "_pending_search_owner", None)
        if owner is not None and (str(owner[0]) != str(message.user_id)
                                  or str(owner[1]) != str(message.scene_id)):
            return "（本次没有进行检索。若问题涉及最新信息而你无法确认，请直接说明并建议检索）"
        text = self.search_service.format_for_model(result)
        character = characters_mod.get(mode)
        if character.is_roleplay:
            return (f"（以下是刚检索到的真实结果，请用{character.name}的口吻转述，"
                    "但事实部分必须准确，并附上来源名称或链接；"
                    "不要改动数字、日期与专有名词）\n" + text)
        return "（以下是刚检索到的真实结果，请据此回答并标注来源）\n" + text

    def _media_catalog(self, message: Optional[IncomingMessage] = None) -> str:
        """把"当前真正可用的媒体能力"如实告诉模型。

        V4 的关键要求：能力不可用时不能声称已发送。
        所以这里必须反映**真实状态**（素材库有没有东西、生成客户端有没有配 key），
        不能写成固定的"支持图片/语音"。
        """
        parts: List[str] = []
        settings = self._current_settings if (message is not None
                                             and not message.is_private) else None

        def enabled(feature: str, global_flag: bool) -> bool:
            if settings is not None and getattr(settings, feature, None) is False:
                return False
            return global_flag

        sticker_ok = enabled("image", self.config.features.emoji) and self.config.media.sticker_enabled
        image_ok = enabled("image", self.config.features.image)
        voice_ok = enabled("voice", self.config.features.voice)

        total = 0
        if self.media_library is not None:
            catalog = self.media_library.catalog_text(
                allow_sticker=sticker_ok, allow_image=image_ok, allow_voice=voice_ok)
            total = len(self.media_library.all())
        else:
            catalog = "（素材库未启用）"

        if sticker_ok:
            parts.append("QQ 表情包：可用（会按话题自动挑选，同一会话有间隔限制，不要每条都发）。")
        else:
            parts.append("QQ 表情包：当前不可用，不要声称已发送表情包。")

        if image_ok:
            gen = ("已接入" if (self.image_client is not None
                               and getattr(self.image_client, "available", False))
                   else "未接入（没有可用素材时无法生成，要如实说明）")
            parts.append(f"图片：素材库可用；AI 生成{gen}。"
                         "AI 生成图片必须说明是生成的，不得称为官方素材。")
        else:
            parts.append("图片：当前不可用（额度或群配置限制），不要声称已发送图片。")

        if voice_ok:
            tts = ("已接入" if (self.tts_client is not None
                               and getattr(self.tts_client, "available", False))
                   else "未接入，要如实说明无法发送语音")
            parts.append(f"语音：{tts}。音色为爱丽丝风格的合成音，"
                         "不得声称使用官方配音或复制了官方演员声音。")
        else:
            parts.append("语音：当前不可用，不要承诺发送语音。")

        if self.search_service is not None:
            parts.append("联网检索：已接入，可查询最新信息并附来源。")
        else:
            parts.append("联网检索：未接入，涉及最新信息时必须说明无法确认。")

        parts.append(f"素材库（共 {total} 项，只列标签与说明，不列文件路径）：")
        parts.append(catalog)
        parts.append("优先级：现成素材 > AI 生成；每次回复最多附带一个媒体；"
                     "不要发送与当前话题无关的图片。")
        return "\n".join(parts)

    def _is_owner(self, user_id: str) -> bool:
        owners = {str(i).strip() for i in self.config.bot.owner_ids if str(i).strip()}
        if str(user_id) in owners:
            return True
        # 首次使用兜底：没配 OWNER_IDS 时，第一个开口的人被认领为 owner，
        # 避免"谁都授权不了"的死局。认领后写库，不会每次换人。
        if not owners or any(o.lower() in ("", "0", "none", "your_qq") for o in owners):
            claimed = self.storage.kv_get("claimed_owner")
            if not claimed:
                self.storage.kv_set("claimed_owner", str(user_id))
                return True
            return str(claimed) == str(user_id)
        return False

    def _is_group_admin(self, user_id: str, group_id: str = "") -> bool:
        """群管理权限。默认只认 owner 列表。

        适配层如果拿到了群角色（owner/admin），会写进 extra.admins；
        这样群主/管理员也能执行 /授权、/群规则，不必等 owner 到场。
        """
        if self._is_owner(user_id):
            return True
        if not group_id:
            return False
        record = self.storage.get_group(group_id)
        if record is None:
            return False
        admins = {str(a) for a in record.extra.get("admins", [])}
        return str(user_id) in admins

    def _owner_hint(self, user_id: str) -> str:
        owners = [str(i) for i in self.config.bot.owner_ids if str(i).strip()]
        return (f"这个操作需要机器人所有者权限。当前你的 QQ 是 {user_id}，"
                f"而 OWNER_IDS 配置为 {('、'.join(owners)) if owners else '（空）'}。\n"
                "把 .env 里的 OWNER_IDS 改成你自己的 QQ 号，重启后就能执行了。")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def reply_to(config: Optional[AppConfig], text: str, user_id: str = "1", *, is_private: bool = True,
             group_id: str = "", user_name: str = "", mentioned: bool = False,
             storage: Optional[Storage] = None, llm: Optional[BaseLLM] = None) -> Outcome:
    """一次性调用：方便脚本和测试使用。"""
    engine = Engine(config, storage, llm)
    message = IncomingMessage(user_id=str(user_id), text=text, is_private=is_private, group_id=str(group_id),
                              user_name=user_name, mentioned=mentioned)
    return engine.handle(message)
