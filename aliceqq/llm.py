"""LLM 客户端：mock（离线确定性）与 OpenAI 兼容接口（纯标准库 urllib 实现）。

之所以不依赖 openai/httpx：本机与用户部署环境都可能装不上第三方包，
V1 只需要一个 POST 请求，标准库足够，且方便离线自测。
"""
from __future__ import annotations

from . import net
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import LLMConfig

Message = Dict[str, str]


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    provider: str = "mock"
    model: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseLLM:
    name = "base"

    def chat(self, messages: Sequence[Message], *, config: LLMConfig,
             tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "ready": True}


# ---------------------------------------------------------------------------
# 离线 mock：保证 V1 在没有 API Key 的情况下也能完整体验流程
# ---------------------------------------------------------------------------

_INTENTS: List[tuple] = [
    ("greeting", r"^(在吗|在么|你好|您好|嗨|哈喽|hi|hello|早上好|晚上好|下午好|早安|晚安)"),
    ("mode_alice", r"(爱丽丝模式|成为爱丽丝|扮演爱丽丝|切换到爱丽丝)"),
    ("mode_assistant", r"(助手模式|普通模式|别扮演|不要扮演|退出角色)"),
    ("identity", r"(你是谁|你叫什么|自我介绍|你是真人吗|你是人吗)"),
    ("game", r"(游戏|打游戏|开黑|段位|抽卡|氪金|手游|端游|steam|原神|蔚蓝档案|碧蓝档案|blue archive)"),
    ("anime", r"(动漫|番剧|追番|漫画|动画|二次元|轻小说)"),
    ("study", r"(作业|考试|学习|复习|论文|题目|怎么学|讲解|公式)"),
    ("coding", r"(代码|编程|bug|报错|函数|脚本|python|java|js|c\+\+)"),
    ("feeling", r"(好累|累死|难受|emo|抑郁|焦虑|不开心|压力|失恋|难过|哭)"),
    ("praise", r"(谢谢|感谢|厉害|好棒|可爱|喜欢你|辛苦)"),
    ("memory", r"(记住|记得|别忘|记忆|忘了吗)"),
    ("ability", r"(能做什么|会什么|有什么功能|怎么用|帮助|help|指令)"),
]


class MockLLM(BaseLLM):
    """用规则 + 模板模拟爱丽丝的回复风格，用于离线演示与自动化测试。

    注意：它只是占位实现，真实人设由 prompts/alice.md 交给真实模型表达。
    """

    name = "mock"

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    # -- 内部工具 --
    @staticmethod
    def _pick(options: Sequence[str], key: str) -> str:
        index = sum(ord(ch) for ch in key) % len(options)
        return options[index]

    @staticmethod
    def _intent(text: str) -> str:
        low = (text or "").strip().lower()
        for name, pattern in _INTENTS:
            if re.search(pattern, low, re.IGNORECASE):
                return name
        return "chat"

    @staticmethod
    def _extract_user_name(messages: Sequence[Message]) -> str:
        for msg in reversed(list(messages)):
            if msg.get("role") != "system":
                continue
            match = re.search(r"希望被称呼为[：:]\s*\**([^\n*（(，,。]{1,16})", msg.get("content", ""))
            if match:
                name = match.group(1).strip(" *_")
                # 模板里的默认昵称没有区分度，直接用"你"
                if name and name not in ("用户", "测试用户"):
                    return name
        return "你"

    # -- 主入口 --
    def chat(self, messages: Sequence[Message], *, config: LLMConfig,
             tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        user_text = ""
        for msg in reversed(list(messages)):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
        system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        # 用模板里的显式标记判断模式：不能靠"是否出现某个词"来猜，
        # 因为助手模板里同样会出现"爱丽丝模式"/"角色扮演"这类字样。
        if "MODE-MARKER:ASSISTANT" in system_text:
            mode = "assistant"
        elif "MODE-MARKER:ALICE" in system_text:
            mode = "alice"
        else:
            mode = "alice"
        name = self._extract_user_name(messages)
        if mode == "assistant":
            text = self._assistant_reply(user_text)
        else:
            text = self._alice_reply(user_text, name)
        return Reply(text=text, provider=self.name, model="mock-rules",
                     usage={"prompt_tokens": len(system_text) // 4, "completion_tokens": len(text) // 2},
                     raw={"intent": self._intent(user_text)})

    def _assistant_reply(self, text: str) -> str:
        intent = self._intent(text)
        if intent == "greeting":
            return "你好，有什么可以帮你的吗？"
        if intent == "ability":
            return ("当前是普通助手模式，我可以做知识问答、翻译、总结、写作建议和代码解释。"
                    "想和爱丽丝聊天的话，发送 /爱丽丝 就能切回去。")
        return ("（离线模拟回复）我收到了你的问题。当前没有配置真实模型，"
                "所以我只能确认消息已经送达并被正确处理。")

    def _alice_reply(self, text: str, name: str) -> str:
        intent = self._intent(text)
        table = {
            "greeting": [
                f"{name}，信号确认。爱丽丝在待机中，随时可以出发。",
                f"嗯，{name}来了。今天的任务日志要写点什么？",
            ],
            "identity": [
                "爱丽丝。千年科学学园，游戏开发部所属。记录应该没有错误。",
                "爱丽丝，游戏开发部的一员。目前的目标是成为勇者……大概是这样。",
            ],
            "game": [
                "游戏的话，爱丽丝很有兴趣！……不过抽卡的部分，经验值经常不够用。",
                "说到游戏，上次那个关卡我卡了三次。这次应该能通关了……大概。",
            ],
            "anime": [
                "动漫吗？爱丽丝会认真记下来的。有推荐的话请直接说。",
                "二次元的资料，爱丽丝的数据库还在持续更新中。",
            ],
            "study": [
                "这个……爱丽丝可以一起想。先把它拆成几个小任务，一个一个来。",
                "嗯，爱丽丝试着理解一下。如果哪里不对，请直接指出来。",
            ],
            "coding": [
                "代码吗？爱丽丝可以帮忙看。……不过爱丽丝的判断不一定完全可靠。",
                "bug 就像是没被发现的隐藏任务。把报错信息发过来，我们一起看。",
            ],
            "feeling": [
                "……嗯。爱丽丝在这里。想说的话，慢慢说也可以。",
                "收到。现在不需要勉强自己，先休息一下也没问题。",
            ],
            "praise": [
                "谢谢。爱丽丝会把这句话保存到重要记录里。",
                "被这样说了。爱丽丝的士气，上升了。",
            ],
            "memory": [
                f"爱丽丝记下了。关于{name}的事情，会保存在{name}自己的记录里。",
                "嗯，这条记录已经写入。需要删除的话，随时可以说。",
            ],
            "ability": [
                "现在的功能是：日常聊天、游戏和动漫话题、群里的互动。"
                "另外可以用 /记忆 查看爱丽丝记住了什么，用 /助手 切换到普通助手模式。",
                "爱丽丝的能力清单：聊天、协助、以及一点点勇者的直觉。太复杂的事情可能要之后才会。",
            ],
            "chat": [
                "……嗯，爱丽丝在听。然后呢？",
                f"{name}说的这个，爱丽丝想再多知道一点。",
                "记录完毕。爱丽丝的回答是：继续这个话题也可以。",
            ],
        }
        options = table.get(intent) or table["chat"]
        return self._pick(options, text or name)

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "ready": True, "note": "离线模拟，不产生网络请求"}


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容的 /chat/completions 客户端（DeepSeek、Qwen、Moonshot 等都适用）。"""

    name = "openai"

    def chat(self, messages: Sequence[Message], *, config: LLMConfig,
             tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        if not config.api_key:
            raise LLMError("缺少 LLM_API_KEY，无法调用真实模型")
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": list(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {config.api_key}")
        try:
            with net.open(request, timeout=config.timeout_seconds or 60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:500]
            raise LLMError(f"模型接口返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"无法连接模型接口 {url}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise LLMError(f"模型接口调用失败：{exc}") from exc

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        finish = choice.get("finish_reason", "")
        if not text and not tool_calls and finish == "length":
            raise LLMError("模型输出被长度限制截断，且没有任何内容")
        return Reply(text=text, provider=self.name, model=config.model,
                     tool_calls=list(tool_calls), usage=data.get("usage") or {}, raw=data)

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "ready": True, "note": "真实 HTTP 调用"}


def build_llm(config: LLMConfig) -> BaseLLM:
    if config.provider in ("openai", "openai_compatible", "compatible", "deepseek", "qwen", "moonshot"):
        return OpenAICompatibleLLM()
    return MockLLM()
