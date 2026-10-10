"""角色注册表。

为什么要有这张表：模式逻辑原来是硬编码的两分支
（MODE_ASSISTANT if ... else MODE_ALICE），加第三个角色就得改好几处判断，
很容易漏改一处（比如提示词选了新角色、但 context 里的 mode 字段还是旧的）。
把"有哪些角色、各自什么行为"集中到一处，引擎只查表。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MODE_ALICE = "alice"
MODE_ASSISTANT = "assistant"
MODE_WHALE = "whale"


@dataclass(frozen=True)
class Character:
    id: str
    name: str                       # 给用户看的名字
    template: str                   # 提示词文件名
    switch_commands: Tuple[str, ...] = ()
    switch_reply: str = ""
    is_roleplay: bool = True        # 是否角色扮演（影响检索结果的口吻包装）
    # 分条发送：小鲸鱼的核心习惯是"一句话一条消息"，
    # 而不是把三句话塞进一条。只有它需要这个行为。
    split_messages: bool = False
    max_split_parts: int = 3
    # 风格护栏名（见 aliceqq/style.py）。空串表示不加护栏。
    style: str = ""
    memory_block: bool = True
    knowledge_block: bool = True


CHARACTERS: Dict[str, Character] = {
    MODE_ALICE: Character(
        id=MODE_ALICE,
        name="爱丽丝",
        template="alice_system.md",
        switch_commands=("/爱丽丝",),
        switch_reply="收到！爱丽丝模式启动。",
    ),
    MODE_WHALE: Character(
        id=MODE_WHALE,
        name="DeepSeek 小鲸鱼",
        template="deepseek_whale.md",
        switch_commands=("/小鲸鱼", "/鲸鱼", "/deepseek", "/ds"),
        switch_reply="行，我来了。别叫我大肥鱼。",
        is_roleplay=True,
        split_messages=True,
        max_split_parts=3,
        style="whale",
    ),
    MODE_ASSISTANT: Character(
        id=MODE_ASSISTANT,
        name="普通助手",
        template="assistant_system.md",
        switch_commands=("/助手",),
        switch_reply="好的，现在切换为普通助手模式。想聊天的发送 /爱丽丝。",
        is_roleplay=False,
        memory_block=False,
        knowledge_block=False,
    ),
}

DEFAULT_MODE = MODE_ALICE


def get(mode: Optional[str]) -> Character:
    """按 id 取角色；未知 id 回落到默认角色（不抛异常，避免一条脏数据把回复搞挂）。"""
    if mode and mode in CHARACTERS:
        return CHARACTERS[mode]
    return CHARACTERS[DEFAULT_MODE]


def exists(mode: Optional[str]) -> bool:
    return bool(mode) and mode in CHARACTERS


def by_command(command: str) -> Optional[Character]:
    """按切换指令找角色（大小写不敏感）。"""
    lowered = (command or "").strip().lower()
    for character in CHARACTERS.values():
        if lowered in (c.lower() for c in character.switch_commands):
            return character
    return None


def switch_help() -> str:
    """把所有切换指令拼成一行，用于帮助文本。"""
    parts = []
    for character in CHARACTERS.values():
        if character.switch_commands:
            parts.append("%s -> %s" % (character.switch_commands[0], character.name))
    return "；".join(parts)


def all_ids() -> List[str]:
    return list(CHARACTERS.keys())
