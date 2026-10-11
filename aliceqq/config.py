"""配置加载：环境变量 / .env / config.json，三层合并。

优先级：进程环境变量 > .env 文件 > config.json > 内置默认值
命名同时兼容文档里的 AGENT_NAME 这类大写写法和 snake_case。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUTHY = {"1", "true", "yes", "y", "on", "是", "开启"}
_FALSY = {"0", "false", "no", "n", "off", "否", "关闭"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_env_file(path: Path) -> Dict[str, str]:
    """极简 .env 解析，避免引入 python-dotenv 依赖。"""
    data: Dict[str, str] = {}
    if not path.is_file():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            data[key] = value
    return data


def _split_words(raw: Any) -> List[str]:
    """把「词1,词2,词3」拆成列表（中英文逗号、顿号、分号、竖线都认）。

    过滤掉空项并去重——管理员手写时很容易多打一个逗号，
    空字符串会变成"命中任何消息"的规则，那是最危险的误伤来源。
    """
    text = str(raw or "").strip()
    if not text:
        return []
    parts = re.split(r"[,，、;；|]+", text)
    kept: List[str] = []
    for part in parts:
        word = part.strip()
        if word and word not in kept:
            kept.append(word)
    return kept


# 这些词太通用，出现在任何群聊里都不代表"在说爱丽丝"。# 早期把它们列进了默认关键词，结果群里聊到"老师""游戏"就全被触发，
# 表现为"发什么都回复"。这里做兜底剔除，用户在 .env 里写了也会被过滤。
TOO_GENERIC_KEYWORDS = {
    "老师", "老师好", "游戏", "玩家", "任务", "副本", "升级", "经验值",
    "ba", "alice", "爱丽丝梦游仙境",
}


def filter_keywords(keywords: List[str]) -> List[str]:
    """剔除过于通用的关键词，并去掉过短的英文（容易误命中）。"""
    kept: List[str] = []
    for word in keywords:
        text = str(word).strip()
        if not text:
            continue
        if text.lower() in TOO_GENERIC_KEYWORDS:
            continue
        # 纯 ASCII 且长度 < 3（如 "ba"）会在任意英文里误命中
        if text.isascii() and text.isalpha() and len(text) < 3:
            continue
        if text not in kept:
            kept.append(text)
    return kept


def _pick(source: Dict[str, Any], *names: str, default: Any = None,
          allow_empty: bool = False) -> Any:
    """按名字取值。

    allow_empty=True 时，显式写成的空值也会被采纳（而不是回落到 default）。
    例如 PROACTIVE_QUIET_HOURS= 表示"没有安静时段"，
    如果按"空值即未设置"处理，用户就永远关不掉安静时段。
    """
    def acceptable(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "" and not allow_empty:
            return False
        return True

    for name in names:
        if name in source and acceptable(source[name]):
            return source[name]
        lower = name.lower()
        for key, value in source.items():
            if key.lower() == lower and acceptable(value):
                return value
    return default


@dataclass
class LLMConfig:
    provider: str = "mock"                 # mock | openai
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.85
    max_tokens: int = 500
    timeout_seconds: int = 60
    context_messages: int = 10
    history_retention: int = 200

    @property
    def is_real(self) -> bool:
        return self.provider != "mock" and bool(self.api_key)


@dataclass
class FeatureConfig:
    web_search: bool = True                # WEB_SEARCH_IN_ALICE_MODE
    memory: bool = True
    memory_sensitive_data: bool = False
    emoji: bool = True
    image: bool = True
    voice: bool = False
    group_management: bool = True
    knowledge_upload: bool = True           # 所有者上传资料
    proactive_private: bool = True
    proactive_group: bool = True
    # 唱歌（RVC 音色转换）。**默认关闭**，理由有三：
    #   1. 一次要跑 60~90 秒并吃满 CPU，只有本机自用时才合适；
    #   2. 涉及翻唱，音乐著作权风险比"声音人格权"更高
    #      （见 docs 项目状态交接 第十五节），不该默认打开；
    #   3. 依赖 D:\DSHWORK\GPT-SoVITS-setup 那套独立环境，别人 clone
    #      这个仓库时不一定有。
    singing: bool = False


@dataclass
class GroupPolicyConfig:
    enabled: bool = True                   # GROUP_CHAT_ENABLED
    access: str = "approved_groups"        # controlled | approved_groups | open
    triggers: List[str] = field(default_factory=lambda: ["mention", "keyword", "command", "relevant_topic"])
    # 只保留"确实指向爱丽丝/本作"的词。像"老师""游戏"这种日常高频词绝不能在这里，
    # 否则群聊里几乎每句话都会被判定为相关。
    keywords: List[str] = field(default_factory=lambda: [
        "爱丽丝", "アリス", "蔚蓝档案", "碧蓝档案", "blue archive",
        "游戏开发部", "千年科学学园", "无名众神",
    ])
    # 话题延续是否要求"同一个人连续说"：默认要求。
    # 关掉的话，机器人一旦在群里开口，窗口期内任何人说话都会被当成延续话题。
    relevant_topic_same_user_only: bool = True
    rate_limit_per_minute: int = 3          # 每个群每分钟最多回复次数
    rate_limit_per_user_per_minute: int = 2
    cooldown_seconds: int = 5
    max_reply_chars: int = 300              # 群聊回复上限，避免刷屏
    reply_probability: float = 1.0          # 命中关键词后的回复概率
    continuity_seconds: int = 90            # 最近参与过的群话题窗口


@dataclass
class PrivatePolicyConfig:
    enabled: bool = True
    rate_limit_per_minute: int = 6
    cooldown_seconds: int = 2
    max_reply_chars: int = 1200


@dataclass
class ProactiveConfig:
    enabled: bool = True
    frequency: str = "low"                  # off | low | normal
    private_only: bool = True
    min_idle_minutes: int = 360
    max_per_user_per_day: int = 1
    quiet_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6, 7])
    check_interval_seconds: int = 900
    templates: List[str] = field(default_factory=lambda: [
        "……{name}，好久没有收到你的信号了。爱丽丝在待机中，状态是「随时可以出发」。",
        "记录完成：距离上次对话已经过了 {hours} 小时。{name} 最近在玩什么？",
    ])
    # ---- V4：群聊低频主动发起话题 ----
    # 默认关闭。群聊主动发言打扰的是所有人，所以必须显式开启，
    # 且有三重约束：群配置允许、群内安静够久、每日/每群上限。
    group_templates: List[str] = field(default_factory=lambda: [
        "（爱丽丝从待机中醒来）这个群好像安静了很久……要不要聊聊最近在玩的游戏？",
        "……记录一下：本群已经安静了一段时间。爱丽丝想问，最近有什么新活动或者新番值得关注吗？",
        "爱丽丝发现一个待解决的支线任务——本群好像没人说话了。有人想聊点什么吗？不需要勉强。",
    ])
    group_min_idle_minutes: int = 240      # 群内安静多久才考虑发言
    group_max_per_day: int = 1             # 每个群每天最多主动几次


@dataclass
class QuotaConfig:
    enabled: bool = True
    daily_private: int = 200
    daily_group: int = 500
    daily_global: int = 5000
    # V3：高成本能力各自独立额度（0 表示不限）。
    # 联网检索是当前唯一已接入的高成本能力；图片/语音先留开关位。
    daily_search_private: int = 60
    daily_search_group: int = 120
    daily_search_global: int = 1000
    daily_image: int = 20
    daily_voice: int = 20
    # 图片理解按张计费（和"生成图片"是两件事，所以额度也分开）。
    # 默认比其他高成本能力宽松一点：看图是**理解用户已经发来的东西**，
    # 不给看会直接影响对话体验；而生成图片是"创造新内容"，可以更保守。
    daily_vision: int = 100
    # 语音理解（用户发来的语音转文字）。
    # 和 daily_asr 分开算：那个是"转写视频音轨"，一段几分钟；
    # 这个是"群友说了一句话"，十几秒。用同一个额度会导致
    # 转两段视频就把语音对话的额度吃光。本地 whisper 不花钱，
    # 所以默认值给得比视频转写宽松。
    daily_asr_voice: int = 200


@dataclass
class MemoryConfig:
    enabled: bool = True
    scope: str = "per_user"
    per_user_limit: int = 80
    conversation_retention: int = 200
    explicit_save: bool = True


@dataclass
class SearchConfig:
    """V2 联网检索配置。默认零配置可用：DuckDuckGo HTML 端点，无需 API Key。"""
    enabled: bool = True                 # 与 WEB_SEARCH_IN_ALICE_MODE 联动
    provider: str = "so360"              # so360 | bing | duckduckgo | searxng | off
    searxng_url: str = ""                # provider=searxng 时必填
    timeout_seconds: int = 15
    max_results: int = 4
    fetch_pages: int = 3                 # 抓取前 N 条的正文；摘要往往滞后，正文才是新信息的来源
    cache_ttl_seconds: int = 600
    # provider=chain 时的回退顺序。实测：360 命中率最好，Bing 作为后备。
    # 可加 "baidu"，但百度网页抓取会撞安全验证（需官方 API Key）。
    # 链首是 B 站：中文游戏/动漫信息最全、pubdate 精确到天，而且免费（不消耗博查额度）。
    # 注意它需要先访问首页取 cookie，否则搜索接口返回 412 request was banned。
    # 博查放在第二位：覆盖 B 站薄弱的领域（新闻、事实、价格等），并作为兜底。
    chain: List[str] = field(default_factory=lambda: ["bilibili", "bocha", "so360", "bing"])
    baidu_api_key: str = ""              # 百度官方 AI 搜索 API 的 key（可选）
    # 博查 AI 搜索 API（官方接口，摘要完整且带发布时间）——配上 key 会成为链首
    bocha_api_key: str = ""
    bocha_summary: bool = True           # 是否请求长文本摘要
    # B 站：视频内信息与评论。
    # 匿名状态下评论只有 3 条、字幕完全不返回；配置 BILIBILI_SESSDATA（自己账号的
    # 登录 Cookie）后可翻页取更多评论并读取 AI/CC 字幕。
    bilibili_sessdata: str = ""
    bilibili_enrich_top: int = 2         # 对前几个视频做富化（0 = 关闭）
    bilibili_comments: int = 8           # 每个视频最多取多少条评论
    bilibili_subtitle_chars: int = 1200  # 字幕节选长度上限
    bilibili_comments_chars: int = 600   # 评论节选长度上限
    auto_search: bool = True             # 时效性问题自动检索，无需模型主动调工具
    announce: bool = True                # 结果前附一行"已检索"提示
    # 查询消歧：把"蔚蓝档案"补成"蔚蓝档案 手游"。
    # 对 360 不需要；换用 bing 等会把系列名拆词匹配的源时应打开。
    disambiguate: bool = False


@dataclass
class VisionConfig:
    """图片理解。

    **这是项目从"能发图"走到"看得懂图"的那一步。**
    在此之前用户发的图只会被存进素材库，从来没送给模型看过，
    所以模型只能靠文件名和用户的话去猜图里有什么。

    走 OpenAI 兼容的视觉接口，不绑定厂商：
    qwen-vl / glm-4v / kimi / 本地 ollama 都是同一个协议，
    换 base_url 和 model 即可。默认关闭——它按张计费，
    而且没配好时应该明确"看不到"，而不是静默失效。
    """
    enabled: bool = False
    base_url: str = ""                   # 例：https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: str = ""
    model: str = ""                      # 例：qwen-vl-max
    timeout: int = 60
    max_tokens: int = 400
    # 一次最多看几张。群聊里连发多图很常见，不设上限成本会翻倍。
    max_images_per_turn: int = 3
    # 视频抽几帧。
    #
    # **这个字段以前不存在，而 engine 已经在 getattr 它**——
    # `getattr(self.config.vision, "video_frames", 3)` 永远拿到默认值 3，
    # 也就是说"调抽帧数量"这件事根本调不动（死代码）。
    # 抽得越多越贵（视觉按图计费），但短视频画面变化快，1 帧常常看不出在做什么。
    video_frames: int = 3
    # 是否把"这一次看不到图"如实告诉模型（关掉的话它会瞎猜）
    tell_model_when_unavailable: bool = True


@dataclass
class MediaConfig:
    """多媒体（V4）：表情包、图片、AI 生成图、语音。"""
    enabled: bool = True
    path: str = "media"
    # 表情包：默认低频，避免刷屏
    sticker_enabled: bool = True
    sticker_probability: float = 0.35   # 命中标签时发送的概率
    sticker_min_interval: int = 180     # 同一会话两次表情的最小间隔（秒）
    # AI 图片生成：按张计费，默认关闭
    image_gen_enabled: bool = False
    image_gen_base_url: str = "https://api.openai.com/v1"
    image_gen_api_key: str = ""
    image_gen_model: str = "dall-e-3"
    image_gen_size: str = "1024x1024"
    image_gen_daily_calls: int = 10
    # 语音合成。provider 可选：
    #   baidu  = 百度翻译的免费 TTS（无需 Key，实测本机可用，默认）
    #   google = Google 翻译的免费 TTS（无需 Key，实测本机可用）
    #   openai_compatible = 任意 OpenAI 兼容 /audio/speech（需要 Key）
    # 免费 provider 不需要 Key，所以这一项默认开启——否则"语音"就是个摆设。
    # 只在用户明确要求语音时才合成，且有每日次数上限。
    tts_enabled: bool = True
    tts_provider: str = "baidu"
    tts_language: str = "zh"
    tts_speed: int = 5                  # 仅百度端点使用，1~9
    tts_base_url: str = "https://api.openai.com/v1"
    tts_api_key: str = ""
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_format: str = "mp3"
    tts_max_chars: int = 120
    tts_daily_calls: int = 30
    # 自建克隆音色服务（TTS_PROVIDER=gptsovits/custom）。
    # ref_audio 是**服务端可见**的参考音频路径；prompt_text 是参考音频里说的话。
    tts_clone_url: str = "http://127.0.0.1:9880"
    tts_clone_ref_audio: str = ""
    # 第二个参考音频：GPT-SoVITS 会把两个参考音频一起用，相当于混合音色
    tts_clone_ref_audio_secondary: str = ""
    tts_clone_prompt_text: str = ""
    # GPT-SoVITS 的语速倍率（1.0 = 原速，>1 更快）。
    # 单独一个键而不是复用 tts_speed：后者是百度端点的 1~9 整数刻度，语义不同。
    # 客户端一直支持 speed_factor，但以前没人传，配置了也不生效。
    tts_clone_speed: float = 1.0
    # 变调（半音，正数更高）。用于把音色推得更"夹/元气"。
    # 为什么需要它：GPT-SoVITS 的输出音调跟随参考有上限——实测参考 F0 提到
    # 526 Hz 时输出只到 399 Hz。想再往上是模型能力之外，只能直接变调。
    tts_pitch: float = 0.0
    # 逐句独立合成再拼接。解决"一次请求里合成多句，前后句演绎不一致"
    # （实测同一段文本第1句与第2句 F0 差 35 Hz，听起来像换了个人）。
    # 随机种子。**固定值才能保证同一个角色声音稳定**。
    # -1 是 GPT-SoVITS 的"随机"，实测会让时长在 4.54~5.72 秒间摆动（±13%）、
    # 音调在 ±15 Hz 晃 —— 听感上就是"语速忽快忽慢、前后像两个人"。
    # 之前客户端默认 -1 而配置里没有这个键，等于每次都在随机采样。
    tts_clone_seed: int = 42
    tts_split_sentences: bool = False
    # 句间插入的静音秒数（仅逐句合成时生效）
    tts_sentence_gap: float = 0.25
    # ffmpeg 可执行文件路径。留空则依次找 PATH、GPT-SoVITS venv 里的
    # imageio-ffmpeg 自带二进制。找不到就不变调并如实说明。
    tts_ffmpeg: str = ""
    # 音色转换（RVC / so-vits-svc 等）。这是"中文发音 + 目标音色"的正确组合：
    # 中文 TTS 负责发音，转换模型负责音色；混合两个 CV 音色靠 vc_blend 的权重。
    tts_vc_url: str = ""
    tts_vc_model: str = ""
    tts_vc_blend: str = ""              # 例如 "alice_ja:0.7,alice_zh:0.3"
    tts_vc_pitch: int = 0
    # 情感：{情感名: 情感参考音频路径}，供 GPT-SoVITS 换语气用。
    # 例：happy=/voices/happy.wav,sad=/voices/sad.wav
    # 注意：百度/Google 免费端点没有情感参数，配了也不会生效（会如实标注）。
    tts_emotion_refs: str = ""
    tts_emotion_alpha: float = 0.7
    tts_emotion_enabled: bool = True
    # 主用克隆音色、不灵时退回哪个 provider（留空表示不退回）。
    # 本地克隆服务是"你自己的一个进程"，没开机或崩了都会让语音直接消失，
    # 所以自己玩时建议设成 baidu。
    tts_fallback: str = ""


@dataclass
class AsrConfig:
    """语音识别（音频转文字）。

    两条路：
    1. B 站自带 AI 字幕——免费，只需 BILIBILI_SESSDATA（推荐，先走这条）；
    2. 本配置（OpenAI 兼容的 /audio/transcriptions）——B 站没给字幕时兜底。

    默认关闭：按时长计费（一个 20 分钟视频约 $0.12），公开服务里必须显式启用，
    并且有体积/时长/每日次数三重上限，避免被刷成账单事故。
    """
    # 后端选择：
    #   openai_compatible —— 云端接口，按量计费，适合公开服务（有额度上限）
    #   local             —— 本地 faster-whisper，不花钱不联网，适合自用
    provider: str = "openai_compatible"
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "whisper-1"
    language: str = "zh"
    timeout_seconds: int = 120
    # 本地 faster-whisper 后端（provider=local 时生效）
    local_model: str = "small"
    local_device: str = "cpu"
    local_cache: str = ""
    max_bytes: int = 8 * 1024 * 1024     # 单次上传音频体积上限
    max_seconds: int = 300               # 只识别前 N 秒（控制成本）
    daily_calls: int = 20                # 每日识别次数上限
    # 长视频分段（只在 provider=local 时生效：本地识别不花钱，值得多花时间）：
    # 超过 max_seconds 的视频不再直接放弃，而是用 ffmpeg 切成 segment_seconds
    # 一段、顺序识别，最多 max_segments 段。宁可覆盖一部分并**写清覆盖范围**，
    # 也不假装整条视频都识别过了；云端按时长计费的路径不会自动分段。
    segment_seconds: int = 300
    max_segments: int = 24
    # 识别提示词（Whisper 的 initial_prompt）。
    #
    # **这是实测最有效的单项改动。** 12 条已知文本的语音做基准：
    #
    #     不加提示        平均字符错误率 40.6%，完全正确  0/12
    #     加简体提示      平均字符错误率 14.2%，完全正确  5/12
    #     加专名+简体提示  平均字符错误率  1.0%，完全正确 11/12
    #
    # 关键发现：**主导错误是输出繁体字**（"今天天气看起来还不错" ->
    # "今天天氣看起來還不錯"），不是一开始以为的同音字。
    # 提示里指定简体 + 列出角色专名，两个问题一起解决。
    #
    # 另外实测对比过两种替代方案，都不如它：
    #   * 事后做简繁转换：7.0%（比直接提示差，还多一层机制）
    #   * temperature=0：40.6%（完全没用）
    #
    # 留空则不加提示。
    initial_prompt: str = (
        "以下是普通话的句子，请用简体中文转写。"
        "数字也用汉字写，例如「三点」而不是「3点」。"
    )
    # 额外的专名列表，会拼到 initial_prompt 后面。
    # 放在这里是因为**专名是每个角色/项目各自不同的**，
    # 不该写死在代码里；实测把角色名加进去能显著降低同音字错误。
    vocabulary: str = ""
    # 是否自动把 `CHARACTER` / `AGENT_NAME` 里的名字也加进专名表。
    #
    # 实测：光有"请用简体中文转写"的提示，错误率从 40.6% 降到 15.9%，
    # 但"爱丽丝"仍然会被写成"艾利斯"；**把角色名加进提示之后降到 1.0%**。
    # 而项目自己明明知道角色叫什么（CHARACTER=爱丽丝），
    # 不该指望用户再去 vocabulary 里重复抄一遍。
    auto_vocabulary: bool = True


@dataclass
class AcademicConfig:
    """学术文献检索（CrossRef / Semantic Scholar / arXiv）。

    实测定位：
    - CrossRef 元数据权威但**基本没有摘要**；
    - Semantic Scholar 摘要最好，但**无 key 时几乎必然 429**；
    - arXiv 免费稳定、摘要完整。
    所以默认启用 arXiv + CrossRef，S2 需要 key 才作为首选。
    """
    enabled: bool = True
    crossref_enabled: bool = True
    arxiv_enabled: bool = True
    semantic_scholar_enabled: bool = True
    semantic_scholar_api_key: str = ""
    # 没配 key 时是否仍尝试 S2（会大概率被限流，但偶尔能成功）
    semantic_scholar_without_key: bool = False
    s2_retries: int = 2
    timeout_seconds: int = 20
    cache_ttl_seconds: int = 1800


@dataclass
class GroupSettings:
    """单个群的独立配置（V3）。

    群主/管理员可以按群调整：是否允许主动参与、语音/图片/搜索、
    群消息额度、触发词、群管理规则。
    存在 group_config.extra（JSON）里，避免频繁改表结构。
    所有字段都是"可选覆盖"：None 表示跟随全局配置。
    """

    proactive: Optional[bool] = None
    search: Optional[bool] = None
    image: Optional[bool] = None
    voice: Optional[bool] = None
    quota_daily: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None
    reply_probability: Optional[float] = None
    max_reply_chars: Optional[int] = None
    # ---- V5：模式开关与群内文本 ----
    # 群主/管理员可以关掉某个模式（例如"本群只做助手，不角色扮演"）。
    # None = 跟随全局（即允许）。
    allow_alice: Optional[bool] = None
    allow_assistant: Optional[bool] = None
    # 入群欢迎 / 退群告别 / 公告。空串表示不发送。
    # 支持占位符 {name}（成员显示名）、{group}（群号）。
    welcome: Optional[str] = None
    farewell: Optional[str] = None
    notice: Optional[str] = None
    # ---- V5：群管理 / 反广告 ----
    # moderation：是否启用审核判定（关掉就完全不看消息内容）
    # moderation_dry_run：只记录不动作。**默认开**——一上来就自动处罚
    #   很容易误伤正常群友，而误伤的代价远大于漏掉一条广告。
    # moderation_auto_warn：中等置信度是否自动警告（高风险动作仍需管理员确认）
    # warn_threshold：累计警告几次后升级为禁言（0 = 不自动升级）
    # ban_words / allow_words：管理员自定义敏感词与白名单（逗号分隔）
    moderation: Optional[bool] = None
    moderation_dry_run: Optional[bool] = None
    moderation_auto_warn: Optional[bool] = None
    warn_threshold: Optional[int] = None
    ban_words: Optional[str] = None
    allow_words: Optional[str] = None
    # 本群额外触发关键词（逗号分隔）。**是在全局关键词之外追加**，
    # 不是替换——替换语义很容易让管理员一不小心把"爱丽丝"本身关掉。
    group_keywords: Optional[str] = None

    # 可通过 /群设置 修改的键：{用户输入: (字段, 类型, 说明)}
    EDITABLE = {
        "主动参与": ("proactive", bool, "机器人是否可在此群主动参与"),
        "proactive": ("proactive", bool, "同上（英文键名）"),
        "搜索": ("search", bool, "是否允许在此群联网检索"),
        "search": ("search", bool, "同上（英文键名）"),
        "图片": ("image", bool, "是否允许图片能力"),
        "image": ("image", bool, "同上（英文键名）"),
        "语音": ("voice", bool, "是否允许语音能力"),
        "voice": ("voice", bool, "同上（英文键名）"),
        "爱丽丝": ("allow_alice", bool, "是否允许本群使用爱丽丝模式"),
        "助手": ("allow_assistant", bool, "是否允许本群使用普通助手模式"),
        "群额度": ("quota_daily", int, "本群每日消息额度"),
        "quota": ("quota_daily", int, "同上（英文键名）"),
        "每分钟上限": ("rate_limit_per_minute", int, "本群每分钟最多回复次数"),
        "回复概率": ("reply_probability", float, "命中触发后回复的概率 0~1"),
        "字数上限": ("max_reply_chars", int, "本群单条回复最大字数"),
        "欢迎语": ("welcome", str, "新成员入群时的欢迎语，支持 {name}、{group}；留「关」则不发送"),
        "告别语": ("farewell", str, "成员退群时的告别语，同上；默认不发送"),
        "公告": ("notice", str, "本群公告文本，成员可用 /公告 查看；留「关」则清空"),
        "审核": ("moderation", bool, "是否启用广告/刷屏审核"),
        "审核试运行": ("moderation_dry_run", bool,
                       "只记录不动作（建议先开一段时间）；填「关」表示真的执行警告"),
        "自动警告": ("moderation_auto_warn", bool,
                     "中等置信度是否自动警告；填「关」表示只记录不警告"),
        "警告上限": ("warn_threshold", int, "累计警告几次后升级为禁言，0=不升级"),
        "敏感词": ("ban_words", str, "本群自定义敏感词，逗号分隔；留「关」清空"),
        "白名单": ("allow_words", str, "永不判定的词，逗号分隔；留「关」清空"),
        "关键词": ("group_keywords", str,
                   "本群额外触发关键词，逗号分隔（在全局关键词之外追加）"),
    }

    # 文本类字段：值里可能有空格、标点，不能按 bool/int 解析
    _TEXT_FIELDS = ("welcome", "farewell", "notice", "ban_words", "allow_words",
                    "group_keywords")
    # 文本字段里表示"清空/不发送"的写法
    _TEXT_OFF = ("关", "关闭", "清空", "清除", "无", "off", "clear", "none", "reset", "默认")

    _TRUE = ("1", "true", "on", "是", "开", "开启", "允许", "yes")
    _FALSE = ("0", "false", "off", "否", "关", "关闭", "不允许", "no")
    _RESET = ("默认", "跟随", "reset", "default")

    @classmethod
    def from_extra(cls, extra: Optional[Dict[str, Any]]) -> "GroupSettings":
        data = dict(extra or {})
        return cls(
            proactive=data.get("proactive"),
            search=data.get("search"),
            image=data.get("image"),
            voice=data.get("voice"),
            quota_daily=data.get("quota_daily"),
            rate_limit_per_minute=data.get("rate_limit_per_minute"),
            reply_probability=data.get("reply_probability"),
            max_reply_chars=data.get("max_reply_chars"),
            allow_alice=data.get("allow_alice"),
            allow_assistant=data.get("allow_assistant"),
            welcome=data.get("welcome"),
            farewell=data.get("farewell"),
            notice=data.get("notice"),
            moderation=data.get("moderation"),
            moderation_dry_run=data.get("moderation_dry_run"),
            moderation_auto_warn=data.get("moderation_auto_warn"),
            warn_threshold=data.get("warn_threshold"),
            ban_words=data.get("ban_words"),
            allow_words=data.get("allow_words"),
            group_keywords=data.get("group_keywords"),
        )

    def to_extra(self) -> Dict[str, Any]:
        """导出全部配置项，None 表示"跟随全局"。

        注意：必须把 None 也带上。upsert_group 的 extra 是**合并**语义，
        且"值为 None = 删除该键"。如果这里省略 None，
        用户执行 /群设置 搜索 默认 时旧值会残留在库里，看起来"恢复默认没生效"。
        """
        return {
            "proactive": self.proactive,
            "search": self.search,
            "image": self.image,
            "voice": self.voice,
            "quota_daily": self.quota_daily,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "reply_probability": self.reply_probability,
            "max_reply_chars": self.max_reply_chars,
            "allow_alice": self.allow_alice,
            "allow_assistant": self.allow_assistant,
            "welcome": self.welcome,
            "farewell": self.farewell,
            "notice": self.notice,
            "moderation": self.moderation,
            "moderation_dry_run": self.moderation_dry_run,
            "moderation_auto_warn": self.moderation_auto_warn,
            "warn_threshold": self.warn_threshold,
            "ban_words": self.ban_words,
            "allow_words": self.allow_words,
            "group_keywords": self.group_keywords,
        }

    def keyword_list(self) -> List[str]:
        """本群额外触发关键词（在全局关键词之外追加）。"""
        return _split_words(self.group_keywords)

    def ban_word_list(self) -> List[str]:
        """敏感词/白名单存成一整行文本，用逗号分隔；这里拆成列表。"""
        return _split_words(self.ban_words)

    def allow_word_list(self) -> List[str]:
        return _split_words(self.allow_words)

    def apply(self, key: str, raw: str):
        """解析并设置一项，返回 (是否成功, 说明)。"""
        spec = self.EDITABLE.get(str(key).strip())
        if spec is None:
            return False, f"不支持设置「{key}」"
        field, caster, _desc = spec
        text = str(raw).strip()
        try:
            if caster is str:
                # 文本类字段（欢迎语/告别语/公告）。
                # 这里必须**原样保留**，不能 strip 掉内部空格、也不能当成数值解析——
                # 用户写的是一句话。空值或"关/清空"表示清除。
                if not text or text.lower() in [w.lower() for w in self._TEXT_OFF]:
                    value = None
                else:
                    value = text
            elif caster is bool:
                lowered = text.lower()
                if lowered in self._TRUE:
                    value = True
                elif lowered in self._FALSE:
                    value = False
                elif lowered in self._RESET:
                    value = None
                else:
                    return False, f"「{text}」不是有效的开关值（用 开/关/默认）"
            elif caster is int:
                value = None if text in self._RESET else int(text)
            else:
                value = None if text in self._RESET else float(text)
        except ValueError:
            return False, f"「{text}」不是合法数值"

        if value is not None:
            if field == "reply_probability" and not (0 <= value <= 1):
                return False, "回复概率需在 0~1 之间"
            if field in ("quota_daily", "rate_limit_per_minute", "max_reply_chars") and value < 0:
                return False, "数值不能为负"
            if field in self._TEXT_FIELDS and len(str(value)) > 200:
                return False, "这段文字太长了（上限 200 字）"
        setattr(self, field, value)
        if field in self._TEXT_FIELDS:
            shown = "已清空" if value is None else f"已设为「{value}」"
        else:
            shown = "跟随全局" if value is None else value
        return True, f"{field} = {shown}"

    def describe(self, config: "AppConfig") -> str:
        policy = config.group

        def flag(value: Optional[bool], fallback: bool) -> str:
            if value is None:
                return f"跟随全局（{'开' if fallback else '关'}）"
            return "开" if value else "关"

        quota = self.quota_daily if self.quota_daily is not None else config.quota.daily_group
        rate = (self.rate_limit_per_minute if self.rate_limit_per_minute is not None
                else policy.rate_limit_per_minute)
        chars = self.max_reply_chars if self.max_reply_chars is not None else policy.max_reply_chars
        prob = self.reply_probability if self.reply_probability is not None else policy.reply_probability

        def text_line(label: str, value: Optional[str], off_hint: str) -> str:
            if not value:
                return f"- {label}：{off_hint}"
            shown = value if len(value) <= 40 else value[:40] + "…"
            return f"- {label}：{shown}"

        def flag_default_on(value: Optional[bool], label: str) -> str:
            """默认即为"开"的开关：None 也要显示成"开"，不能让管理员困惑。"""
            return f"- {label}：{'开' if (value is None or value) else '关'}"

        moderation_lines = [
            flag_default_on(self.moderation, "审核"),
            flag_default_on(self.moderation_dry_run, "审核试运行（只记录不动作）"),
            flag_default_on(self.moderation_auto_warn, "自动警告"),
            f"- 升级禁言阈值：{self.warn_threshold if self.warn_threshold is not None else '不自动升级'}",
            text_line("自定义敏感词", self.ban_words, "未设置"),
            text_line("白名单词", self.allow_words, "未设置"),
            text_line("本群额外关键词", self.group_keywords, "未设置（只用全局关键词）"),
        ]

        return "\n".join([
            f"- 主动参与：{flag(self.proactive, config.features.proactive_group)}",
            f"- 联网检索：{flag(self.search, config.features.web_search)}",
            f"- 图片能力：{flag(self.image, config.features.image)}",
            f"- 语音能力：{flag(self.voice, config.features.voice)}",
            f"- 爱丽丝模式：{flag(self.allow_alice, True)}",
            f"- 助手模式：{flag(self.allow_assistant, True)}",
            f"- 每日群额度：{quota}",
            f"- 每分钟回复上限：{rate}",
            f"- 触发后回复概率：{prob}",
            f"- 单条回复字数上限：{chars}",
            text_line("入群欢迎语", self.welcome, "未设置（不发送）"),
            text_line("退群告别语", self.farewell, "未设置（不发送）"),
            text_line("群公告", self.notice, "未设置"),
        ] + moderation_lines)


@dataclass
class BotConfig:
    agent_name: str = "爱丽丝 QQ 伙伴"
    primary_type: str = "角色陪伴型 Agent"
    character: str = "爱丽丝"
    relationship: str = "朋友或队友"
    romance_allowed: bool = False
    default_mode: str = "alice"            # alice | assistant
    assistant_mode_enabled: bool = True
    owner_ids: List[str] = field(default_factory=lambda: ["10001", "10086"])
    command_prefix: str = "/"
    mode_switch_alice: str = "/爱丽丝"
    mode_switch_assistant: str = "/助手"
    timezone: str = "Asia/Shanghai"
    time_format: str = "%Y-%m-%d %H:%M"
    max_input_chars: int = 1500
    knowledge_dir: str = "knowledge"
    knowledge_max_chars: int = 12000      # 注入提示词的知识总量上限（V2 资料变大后必须限流）
    knowledge_max_file_chars: int = 6000  # 单篇上限
    knowledge_upload_prefix: str = "90"   # 上传资料的优先级前缀，数字越小越优先
    prompts_dir: str = "prompts"
    data_dir: str = "data"
    db_path: str = "data/alice.db"
    group_config_path: str = "data/groups.json"


@dataclass
class AppConfig:
    bot: BotConfig = field(default_factory=BotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    private: PrivatePolicyConfig = field(default_factory=PrivatePolicyConfig)
    group: GroupPolicyConfig = field(default_factory=GroupPolicyConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    academic: AcademicConfig = field(default_factory=AcademicConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    raw_env: Dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 路径 ----------
    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def db_file(self) -> Path:
        return self.resolve(self.bot.db_path)

    @property
    def knowledge_path(self) -> Path:
        return self.resolve(self.bot.knowledge_dir)

    @property
    def prompts_path(self) -> Path:
        return self.resolve(self.bot.prompts_dir)

    @property
    def groups_file(self) -> Path:
        return self.resolve(self.bot.group_config_path)

    def env(self, name: str, default: str = "") -> str:
        return self.raw_env.get(name, default)

    # ---------- 构造 ----------
    @classmethod
    def load(cls, config_file: Optional[Union[str, Path]] = None) -> "AppConfig":
        file_data: Dict[str, Any] = {}
        path = Path(config_file) if config_file else (PROJECT_ROOT / "config.json")
        if path.is_file():
            try:
                file_data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                file_data = {}
        env_file = load_env_file(PROJECT_ROOT / ".env")
        merged: Dict[str, Any] = {}
        merged.update(file_data)
        merged.update(env_file)
        merged.update(os.environ)

        bot = BotConfig(
            agent_name=str(_pick(merged, "AGENT_NAME", "agent_name", default=BotConfig.agent_name)),
            primary_type=str(_pick(merged, "PRIMARY_TYPE", "primary_type", default=BotConfig.primary_type)),
            character=str(_pick(merged, "CHARACTER", "character", default=BotConfig.character)),
            relationship=str(_pick(merged, "RELATIONSHIP", "relationship", default=BotConfig.relationship)),
            romance_allowed=_as_bool(_pick(merged, "ROMANCE_ALLOWED", "romance_allowed"), False),
            default_mode=str(_pick(merged, "DEFAULT_MODE", "default_mode", default="alice")).strip().lower(),
            assistant_mode_enabled=_as_bool(_pick(merged, "ASSISTANT_MODE_ENABLED", "assistant_mode_enabled"), True),
            owner_ids=_as_list(_pick(merged, "OWNER_IDS", "owner_ids"), ["10001"]),
            command_prefix=str(_pick(merged, "COMMAND_PREFIX", "command_prefix", default="/")),
            timezone=str(_pick(merged, "TIMEZONE", "timezone", default="Asia/Shanghai")),
            knowledge_dir=str(_pick(merged, "KNOWLEDGE_DIR", "knowledge_dir", default="knowledge")),
            knowledge_max_chars=_as_int(_pick(merged, "KNOWLEDGE_MAX_CHARS", "knowledge_max_chars"), 12000),
            knowledge_max_file_chars=_as_int(_pick(merged, "KNOWLEDGE_MAX_FILE_CHARS", "knowledge_max_file_chars"), 6000),
            knowledge_upload_prefix=str(_pick(merged, "KNOWLEDGE_UPLOAD_PREFIX",
                                              "knowledge_upload_prefix", default="90")),
            prompts_dir=str(_pick(merged, "PROMPTS_DIR", "prompts_dir", default="prompts")),
            data_dir=str(_pick(merged, "DATA_DIR", "data_dir", default="data")),
            db_path=str(_pick(merged, "DB_PATH", "db_path", default="data/alice.db")),
            group_config_path=str(_pick(merged, "GROUP_CONFIG_PATH", "group_config_path", default="data/groups.json")),
        )
        if bot.default_mode not in ("alice", "assistant"):
            bot.default_mode = "alice"

        llm = LLMConfig(
            provider=str(_pick(merged, "LLM_PROVIDER", "llm_provider", default="mock")).strip().lower(),
            base_url=str(_pick(merged, "LLM_BASE_URL", "OPENAI_BASE_URL", "llm_base_url", default=LLMConfig.base_url)).rstrip("/"),
            api_key=str(_pick(merged, "LLM_API_KEY", "OPENAI_API_KEY", "llm_api_key", default="")),
            model=str(_pick(merged, "LLM_MODEL", "OPENAI_MODEL", "llm_model", default=LLMConfig.model)),
            temperature=_as_float(_pick(merged, "LLM_TEMPERATURE", "llm_temperature"), 0.85),
            max_tokens=_as_int(_pick(merged, "LLM_MAX_TOKENS", "llm_max_tokens"), 500),
            timeout_seconds=_as_int(_pick(merged, "LLM_TIMEOUT", "llm_timeout_seconds"), 60),
            context_messages=_as_int(_pick(merged, "LLM_CONTEXT_MESSAGES", "context_messages"), 10),
        )
        if llm.provider == "auto":
            llm.provider = "openai" if llm.api_key else "mock"

        memory = MemoryConfig(
            enabled=_as_bool(_pick(merged, "MEMORY_ENABLED", "memory_enabled"), True),
            scope=str(_pick(merged, "MEMORY_SCOPE", "memory_scope", default="per_user")),
            per_user_limit=_as_int(_pick(merged, "MEMORY_PER_USER_LIMIT", "memory_per_user_limit"), 80),
            conversation_retention=_as_int(_pick(merged, "CONVERSATION_RETENTION", "conversation_retention"), 200),
            explicit_save=_as_bool(_pick(merged, "MEMORY_EXPLICIT_SAVE", "memory_explicit_save"), True),
        )

        private = PrivatePolicyConfig(
            enabled=_as_bool(_pick(merged, "PRIVATE_CHAT_ENABLED", "private_chat_enabled"), True),
            rate_limit_per_minute=_as_int(_pick(merged, "PRIVATE_RATE_LIMIT_PER_MINUTE", "private_rate_limit_per_minute"), 6),
            cooldown_seconds=_as_int(_pick(merged, "PRIVATE_COOLDOWN_SECONDS", "private_cooldown_seconds"), 2),
            max_reply_chars=_as_int(_pick(merged, "PRIVATE_MAX_REPLY_CHARS", "private_max_reply_chars"), 1200),
        )

        group = GroupPolicyConfig(
            enabled=_as_bool(_pick(merged, "GROUP_CHAT_ENABLED", "group_chat_enabled"), True),
            access=str(_pick(merged, "GROUP_ACCESS", "group_access", default="approved_groups")),
            triggers=_as_list(_pick(merged, "GROUP_TRIGGER", "group_triggers"), GroupPolicyConfig().triggers),
            keywords=filter_keywords(_as_list(_pick(merged, "GROUP_KEYWORDS", "group_keywords"),
                                               GroupPolicyConfig().keywords)) or filter_keywords(GroupPolicyConfig().keywords),
            relevant_topic_same_user_only=_as_bool(
                _pick(merged, "GROUP_RELEVANT_TOPIC_SAME_USER", "relevant_topic_same_user_only"), True),
            rate_limit_per_minute=_as_int(_pick(merged, "GROUP_RATE_LIMIT_PER_MINUTE", "group_rate_limit_per_minute"), 3),
            rate_limit_per_user_per_minute=_as_int(_pick(merged, "GROUP_RATE_LIMIT_PER_USER", "group_rate_limit_per_user_per_minute"), 2),
            max_reply_chars=_as_int(_pick(merged, "GROUP_MAX_REPLY_CHARS", "group_max_reply_chars"), 300),
            reply_probability=_as_float(_pick(merged, "GROUP_REPLY_PROBABILITY", "group_reply_probability"), 1.0),
        )
        if group.access not in ("controlled", "approved_groups", "open"):
            group.access = "approved_groups"

        proactive = ProactiveConfig(
            enabled=_as_bool(_pick(merged, "PROACTIVE_ENABLED", "proactive_enabled"), True),
            frequency=str(_pick(merged, "PROACTIVE_FREQUENCY", "proactive_frequency", default="low")).strip().lower(),
            private_only=_as_bool(_pick(merged, "PROACTIVE_PRIVATE_ONLY", "proactive_private_only"), True),
            min_idle_minutes=_as_int(_pick(merged, "PROACTIVE_MIN_IDLE_MINUTES", "proactive_min_idle_minutes"), 360),
            max_per_user_per_day=_as_int(_pick(merged, "PROACTIVE_MAX_PER_USER_PER_DAY", "proactive_max_per_user_per_day"), 1),
            # allow_empty：显式写成空值表示"没有安静时段"，不能回落到默认时段
            quiet_hours=[_as_int(h, 0) for h in _as_list(
                _pick(merged, "PROACTIVE_QUIET_HOURS", "proactive_quiet_hours",
                      default=[str(h) for h in ProactiveConfig().quiet_hours], allow_empty=True),
                [])],
            check_interval_seconds=_as_int(_pick(merged, "PROACTIVE_CHECK_INTERVAL", "proactive_check_interval_seconds"), 900),
            group_min_idle_minutes=_as_int(
                _pick(merged, "PROACTIVE_GROUP_MIN_IDLE_MINUTES",
                      "proactive_group_min_idle_minutes"), 240),
            group_max_per_day=_as_int(
                _pick(merged, "PROACTIVE_GROUP_MAX_PER_DAY", "proactive_group_max_per_day"), 1),
        )
        group_templates = _as_list(
            _pick(merged, "PROACTIVE_GROUP_TEMPLATES", "proactive_group_templates",
                  default=ProactiveConfig().group_templates, allow_empty=True), [])
        proactive.group_templates = group_templates
        if proactive.frequency not in ("off", "low", "normal"):
            proactive.frequency = "low"

        quota = QuotaConfig(
            enabled=_as_bool(_pick(merged, "QUOTA_ENABLED", "quota_enabled"), True),
            daily_private=_as_int(_pick(merged, "QUOTA_DAILY_PRIVATE", "quota_daily_private"), 200),
            daily_group=_as_int(_pick(merged, "QUOTA_DAILY_GROUP", "quota_daily_group"), 500),
            daily_global=_as_int(_pick(merged, "QUOTA_DAILY_GLOBAL", "quota_daily_global"), 5000),
            daily_search_private=_as_int(
                _pick(merged, "QUOTA_SEARCH_PRIVATE", "quota_daily_search_private"), 60),
            daily_search_group=_as_int(
                _pick(merged, "QUOTA_SEARCH_GROUP", "quota_daily_search_group"), 120),
            daily_search_global=_as_int(
                _pick(merged, "QUOTA_SEARCH_GLOBAL", "quota_daily_search_global"), 1000),
            daily_image=_as_int(_pick(merged, "QUOTA_IMAGE", "quota_daily_image"), 20),
            daily_voice=_as_int(_pick(merged, "QUOTA_VOICE", "quota_daily_voice"), 20),
        )

        search = SearchConfig(
            enabled=_as_bool(_pick(merged, "SEARCH_ENABLED", "search_enabled"), True),
            provider=str(_pick(merged, "SEARCH_PROVIDER", "search_provider", default="so360")).strip().lower(),
            searxng_url=str(_pick(merged, "SEARXNG_URL", "searxng_url", default="") or ""),
            timeout_seconds=_as_int(_pick(merged, "SEARCH_TIMEOUT", "search_timeout_seconds"), 15),
            max_results=_as_int(_pick(merged, "SEARCH_MAX_RESULTS", "search_max_results"), 4),
            fetch_pages=_as_int(_pick(merged, "SEARCH_FETCH_PAGES", "search_fetch_pages"), 3),
            cache_ttl_seconds=_as_int(_pick(merged, "SEARCH_CACHE_TTL", "search_cache_ttl_seconds"), 600),
            chain=_as_list(_pick(merged, "SEARCH_CHAIN", "search_chain"),
                           ["bilibili", "bocha", "so360", "bing"]),
            baidu_api_key=str(_pick(merged, "BAIDU_API_KEY", "baidu_api_key", default="") or ""),
            bocha_api_key=str(_pick(merged, "BOCHA_API_KEY", "bocha_api_key", default="") or ""),
            bocha_summary=_as_bool(_pick(merged, "BOCHA_SUMMARY", "bocha_summary"), True),
            bilibili_sessdata=str(_pick(merged, "BILIBILI_SESSDATA", "bilibili_sessdata", default="") or ""),
            bilibili_enrich_top=_as_int(_pick(merged, "BILIBILI_ENRICH_TOP", "bilibili_enrich_top"), 2),
            bilibili_comments=_as_int(_pick(merged, "BILIBILI_COMMENTS", "bilibili_comments"), 8),
            bilibili_subtitle_chars=_as_int(
                _pick(merged, "BILIBILI_SUBTITLE_CHARS", "bilibili_subtitle_chars"), 1200),
            bilibili_comments_chars=_as_int(
                _pick(merged, "BILIBILI_COMMENTS_CHARS", "bilibili_comments_chars"), 600),
            auto_search=_as_bool(_pick(merged, "SEARCH_AUTO", "search_auto_search"), True),
            announce=_as_bool(_pick(merged, "SEARCH_ANNOUNCE", "search_announce"), True),
            disambiguate=_as_bool(_pick(merged, "SEARCH_DISAMBIGUATE", "search_disambiguate"), False),
        )
        if search.provider in ("none", "off", "disabled", "false"):
            search.enabled = False
        if search.provider == "searxng" and not search.searxng_url:
            # 配了 searxng 却没给地址：退回默认 provider，避免静默不可用
            search.provider = "duckduckgo"

        academic = AcademicConfig(
            enabled=_as_bool(_pick(merged, "ACADEMIC_ENABLED", "academic_enabled"), True),
            crossref_enabled=_as_bool(_pick(merged, "ACADEMIC_CROSSREF", "crossref_enabled"), True),
            arxiv_enabled=_as_bool(_pick(merged, "ACADEMIC_ARXIV", "arxiv_enabled"), True),
            semantic_scholar_enabled=_as_bool(
                _pick(merged, "ACADEMIC_S2", "semantic_scholar_enabled"), True),
            semantic_scholar_api_key=str(
                _pick(merged, "SEMANTIC_SCHOLAR_API_KEY", "semantic_scholar_api_key", default="") or ""),
            semantic_scholar_without_key=_as_bool(
                _pick(merged, "ACADEMIC_S2_WITHOUT_KEY", "semantic_scholar_without_key"), False),
            s2_retries=_as_int(_pick(merged, "ACADEMIC_S2_RETRIES", "s2_retries"), 2),
            timeout_seconds=_as_int(_pick(merged, "ACADEMIC_TIMEOUT", "academic_timeout_seconds"), 20),
            cache_ttl_seconds=_as_int(
                _pick(merged, "ACADEMIC_CACHE_TTL", "academic_cache_ttl_seconds"), 1800),
        )

        vision = VisionConfig(
            enabled=_as_bool(_pick(merged, "VISION_ENABLED", "vision_enabled"), False),
            base_url=str(_pick(merged, "VISION_BASE_URL", "vision_base_url",
                               default="") or "").rstrip("/"),
            api_key=str(_pick(merged, "VISION_API_KEY", "vision_api_key",
                               default="") or ""),
            model=str(_pick(merged, "VISION_MODEL", "vision_model",
                            default="") or ""),
            timeout=_as_int(_pick(merged, "VISION_TIMEOUT", "vision_timeout"), 60),
            max_tokens=_as_int(_pick(merged, "VISION_MAX_TOKENS", "vision_max_tokens"), 400),
            max_images_per_turn=_as_int(
                _pick(merged, "VISION_MAX_IMAGES", "vision_max_images"), 3),
            video_frames=_as_int(
                _pick(merged, "VISION_VIDEO_FRAMES", "vision_video_frames"), 3),
            tell_model_when_unavailable=_as_bool(
                _pick(merged, "VISION_TELL_WHEN_UNAVAILABLE",
                      "vision_tell_when_unavailable"), True),
        )

        media = MediaConfig(
            enabled=_as_bool(_pick(merged, "MEDIA_ENABLED", "media_enabled"), True),
            path=str(_pick(merged, "MEDIA_PATH", "media_path", default="media")),
            sticker_enabled=_as_bool(_pick(merged, "STICKER_ENABLED", "sticker_enabled"), True),
            sticker_probability=_as_float(
                _pick(merged, "STICKER_PROBABILITY", "sticker_probability"), 0.35),
            sticker_min_interval=_as_int(
                _pick(merged, "STICKER_MIN_INTERVAL", "sticker_min_interval"), 180),
            image_gen_enabled=_as_bool(
                _pick(merged, "IMAGE_GEN_ENABLED", "image_gen_enabled"), False),
            image_gen_base_url=str(_pick(merged, "IMAGE_GEN_BASE_URL", "image_gen_base_url",
                                         default="https://api.openai.com/v1")).rstrip("/"),
            image_gen_api_key=str(_pick(merged, "IMAGE_GEN_API_KEY", "image_gen_api_key",
                                        default="") or ""),
            image_gen_model=str(_pick(merged, "IMAGE_GEN_MODEL", "image_gen_model",
                                      default="dall-e-3")),
            image_gen_size=str(_pick(merged, "IMAGE_GEN_SIZE", "image_gen_size",
                                     default="1024x1024")),
            image_gen_daily_calls=_as_int(
                _pick(merged, "IMAGE_GEN_DAILY_CALLS", "image_gen_daily_calls"), 10),
            tts_enabled=_as_bool(_pick(merged, "TTS_ENABLED", "tts_enabled"), True),
            tts_provider=str(_pick(merged, "TTS_PROVIDER", "tts_provider",
                                   default="baidu")).strip().lower(),
            tts_language=str(_pick(merged, "TTS_LANGUAGE", "tts_language", default="zh")),
            tts_speed=_as_int(_pick(merged, "TTS_SPEED", "tts_speed"), 5),
            tts_base_url=str(_pick(merged, "TTS_BASE_URL", "tts_base_url",
                                   default="https://api.openai.com/v1")).rstrip("/"),
            tts_api_key=str(_pick(merged, "TTS_API_KEY", "tts_api_key", default="") or ""),
            tts_model=str(_pick(merged, "TTS_MODEL", "tts_model", default="tts-1")),
            tts_voice=str(_pick(merged, "TTS_VOICE", "tts_voice", default="alloy")),
            tts_format=str(_pick(merged, "TTS_FORMAT", "tts_format", default="mp3")),
            tts_max_chars=_as_int(_pick(merged, "TTS_MAX_CHARS", "tts_max_chars"), 120),
            tts_daily_calls=_as_int(_pick(merged, "TTS_DAILY_CALLS", "tts_daily_calls"), 30),
            tts_clone_url=str(_pick(merged, "TTS_CLONE_URL", "tts_clone_url",
                                    default="http://127.0.0.1:9880")).rstrip("/"),
            tts_clone_ref_audio=str(_pick(merged, "TTS_CLONE_REF_AUDIO",
                                          "tts_clone_ref_audio", default="") or ""),
            tts_clone_ref_audio_secondary=str(_pick(
                merged, "TTS_CLONE_REF_AUDIO_2", "tts_clone_ref_audio_secondary",
                default="") or ""),
            tts_vc_url=str(_pick(merged, "TTS_VC_URL", "tts_vc_url", default="") or "").rstrip("/"),
            tts_vc_model=str(_pick(merged, "TTS_VC_MODEL", "tts_vc_model", default="") or ""),
            tts_vc_blend=str(_pick(merged, "TTS_VC_BLEND", "tts_vc_blend", default="") or ""),
            tts_vc_pitch=_as_int(_pick(merged, "TTS_VC_PITCH", "tts_vc_pitch"), 0),
            tts_emotion_refs=str(_pick(merged, "TTS_EMOTION_REFS",
                                       "tts_emotion_refs", default="") or ""),
            tts_emotion_alpha=_as_float(_pick(merged, "TTS_EMOTION_ALPHA",
                                              "tts_emotion_alpha"), 0.7),
            tts_emotion_enabled=_as_bool(_pick(merged, "TTS_EMOTION_ENABLED",
                                               "tts_emotion_enabled"), True),
            tts_fallback=str(_pick(merged, "TTS_FALLBACK", "tts_fallback",
                                   default="")).strip().lower(),
            tts_clone_prompt_text=str(_pick(merged, "TTS_CLONE_PROMPT_TEXT",
                                            "tts_clone_prompt_text", default="") or ""),
            tts_clone_speed=_as_float(_pick(merged, "TTS_CLONE_SPEED",
                                            "tts_clone_speed"), 1.0),
            tts_pitch=_as_float(_pick(merged, "TTS_PITCH", "tts_pitch"), 0.0),
            tts_clone_seed=_as_int(_pick(merged, "TTS_CLONE_SEED",
                                         "tts_clone_seed"), 42),
            tts_split_sentences=_as_bool(_pick(merged, "TTS_SPLIT_SENTENCES",
                                               "tts_split_sentences"), False),
            tts_sentence_gap=_as_float(_pick(merged, "TTS_SENTENCE_GAP",
                                             "tts_sentence_gap"), 0.25),
            tts_ffmpeg=str(_pick(merged, "TTS_FFMPEG", "tts_ffmpeg", default="") or ""),
        )
        # 开了却没有 key：明确关闭，避免每次白跑必然失败的调用
        if media.image_gen_enabled and not media.image_gen_api_key:
            media.image_gen_enabled = False
        # 只有需要 Key 的 provider 才因缺 Key 而关闭。
        # 早期写法是"没 key 就关"，会把 baidu/google 这类免费 provider 一起关掉，
        # 结果"语音"永远不可用。
        if (media.tts_enabled
                and media.tts_provider in ("openai_compatible", "openai")
                and not media.tts_api_key):
            media.tts_enabled = False

        asr = AsrConfig(
            provider=str(_pick(merged, "ASR_PROVIDER", "asr_provider",
                               default="openai_compatible") or "").strip().lower(),
            enabled=_as_bool(_pick(merged, "ASR_ENABLED", "asr_enabled"), False),
            base_url=str(_pick(merged, "ASR_BASE_URL", "asr_base_url",
                               default="https://api.openai.com/v1")).rstrip("/"),
            api_key=str(_pick(merged, "ASR_API_KEY", "asr_api_key", default="") or ""),
            model=str(_pick(merged, "ASR_MODEL", "asr_model", default="whisper-1")),
            language=str(_pick(merged, "ASR_LANGUAGE", "asr_language", default="zh")),
            timeout_seconds=_as_int(_pick(merged, "ASR_TIMEOUT", "asr_timeout_seconds"), 120),
            max_bytes=_as_int(_pick(merged, "ASR_MAX_BYTES", "asr_max_bytes"), 8 * 1024 * 1024),
            max_seconds=_as_int(_pick(merged, "ASR_MAX_SECONDS", "asr_max_seconds"), 300),
            daily_calls=_as_int(_pick(merged, "ASR_DAILY_CALLS", "asr_daily_calls"), 20),
            segment_seconds=_as_int(
                _pick(merged, "ASR_SEGMENT_SECONDS", "asr_segment_seconds"), 300),
            max_segments=_as_int(
                _pick(merged, "ASR_MAX_SEGMENTS", "asr_max_segments"), 24),
            local_model=str(_pick(merged, "ASR_LOCAL_MODEL", "asr_local_model",
                                  default="small") or "small").strip(),
            local_device=str(_pick(merged, "ASR_LOCAL_DEVICE", "asr_local_device",
                                   default="cpu") or "cpu").strip().lower(),
            local_cache=str(_pick(merged, "ASR_LOCAL_CACHE", "asr_local_cache",
                                  default="") or "").strip(),
            # 提示词：默认值在 dataclass 里，这里只在用户显式配了才覆盖。
            # 用 _pick 不带 default，好区分"没配"和"配成空串"
            # （配成空串是"故意不要提示"的合法用法）。
            initial_prompt=str(_pick(merged, "ASR_INITIAL_PROMPT", "asr_initial_prompt",
                                     default=AsrConfig.initial_prompt) or ""),
            vocabulary=str(_pick(merged, "ASR_VOCABULARY", "asr_vocabulary",
                                 default="") or "").strip(),
            auto_vocabulary=_as_bool(
                _pick(merged, "ASR_AUTO_VOCABULARY", "asr_auto_vocabulary"), True),
        )
        # 自动把角色名并进专名表。放在这里而不是 build_client 里，
        # 是因为配置层才知道 CHARACTER / AGENT_NAME 是什么。
        if asr.auto_vocabulary:
            extra = [part.strip() for part in
                     (bot.character, bot.agent_name) if part and part.strip()]
            if extra:
                existing = [w.strip() for w in asr.vocabulary.split("、") if w.strip()]
                for word in extra:
                    if word not in existing:
                        existing.append(word)
                asr.vocabulary = "、".join(existing)
        # 只有**需要 Key 的云端后端**才因缺 Key 而关闭。
        # 本地后端不需要 Key，被一起关掉的话"开了 ASR 却永远不可用"（TTS 那边踩过同样的坑）。
        if asr.enabled and asr.provider in ("openai", "openai_compatible") and not asr.api_key:
            asr.enabled = False

        features = FeatureConfig(
            web_search=_as_bool(_pick(merged, "WEB_SEARCH_IN_ALICE_MODE", "web_search", "web_search_in_alice_mode"), True),
            memory=_as_bool(_pick(merged, "MEMORY_ENABLED", "memory_enabled"), True),
            memory_sensitive_data=_as_bool(_pick(merged, "MEMORY_SENSITIVE_DATA", "memory_sensitive_data"), False),
            emoji=_as_bool(_pick(merged, "EMOJI_ENABLED", "emoji_enabled"), True),
            image=_as_bool(_pick(merged, "IMAGE_ENABLED", "image_enabled"), True),
            voice=_as_bool(_pick(merged, "VOICE_ENABLED", "voice_enabled"), False),
            group_management=_as_bool(_pick(merged, "GROUP_MANAGEMENT_ENABLED", "group_management_enabled"), True),
            knowledge_upload=_as_bool(_pick(merged, "KNOWLEDGE_UPLOAD_ENABLED", "knowledge_upload"), True),
            proactive_private=_as_bool(_pick(merged, "PROACTIVE_PRIVATE", "proactive_private"), True),
            proactive_group=_as_bool(_pick(merged, "PROACTIVE_GROUP", "proactive_group"), True),
            singing=_as_bool(_pick(merged, "SINGING_ENABLED", "singing_enabled"), False),
        )

        # 检索总开关跟随 WEB_SEARCH_IN_ALICE_MODE：关掉后者就彻底不联网
        if not features.web_search:
            search.enabled = False

        return cls(bot=bot, llm=llm, memory=memory, private=private, group=group,
                   proactive=proactive, quota=quota, search=search, academic=academic,
                   asr=asr, media=media, features=features, raw_env=merged,
                   vision=vision)

    # 所有凭据字段的统一脱敏表：{配置段: [字段名]}
    SECRET_FIELDS = {
        "llm": ["api_key"],
        "search": ["bocha_api_key", "baidu_api_key", "bilibili_sessdata"],
        "academic": ["semantic_scholar_api_key"],
        "asr": ["api_key"],
        "media": ["image_gen_api_key", "tts_api_key"],
    }

    def public_dict(self) -> Dict[str, Any]:
        """对外可见的配置快照。

        注意：**所有**凭据字段都必须脱敏。早期只处理了 llm.api_key，
        导致后来新增的博查/百度/S2 的 key 会原样出现在输出里。
        """
        data = asdict(self)
        data.pop("raw_env", None)
        for section, fields in self.SECRET_FIELDS.items():
            block = data.get(section)
            if not isinstance(block, dict):
                continue
            for name in fields:
                if name in block:
                    block[name] = "***" if block.get(name) else ""
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.public_dict(), ensure_ascii=False, indent=indent)


def write_example_config(path: Path) -> None:
    path.write_text(AppConfig().to_json(), encoding="utf-8")
