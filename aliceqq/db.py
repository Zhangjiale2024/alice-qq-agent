"""SQLite 存储层：对话记录、用户档案、长期记忆、群配置、额度计数。

所有用户相关数据都以 user_id 为分区键，群数据以 group_id 为分区键，
从存储层就杜绝"A 的记忆被 B 看到"。标准库实现，无第三方依赖。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 会话作用域：私聊统一用这个字符串，群聊直接用群号。
SCOPE_PRIVATE = "private"

SCHEMA = """CREATE TABLE IF NOT EXISTS conversation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    group_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_log(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_conv_group ON conversation_log(group_id, id DESC);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    nickname TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'alice',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_fact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',
    confidence REAL NOT NULL DEFAULT 0.6,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(user_id, content)
);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_fact(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS group_config (
    group_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    mode TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    rules TEXT NOT NULL DEFAULT '',
    extra TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_counter (
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    period TEXT NOT NULL,
    counter INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, scope_id, period)
);

CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- V5：模式按**会话**隔离（scope = 'private' 或群号）。
-- 为什么单独一张表而不是复用 user_profile.mode：
-- 同一个用户在 A 群可以是助手模式、在 B 群是爱丽丝模式、私聊又是另一种，
-- 而 user_profile 是每用户一行的。旧的 user_profile.mode 保留为"全局默认"，
-- 已有数据在迁移时灌进这张表，行为不变。
CREATE TABLE IF NOT EXISTS session_mode (
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    mode TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, scope)
);
CREATE INDEX IF NOT EXISTS idx_session_mode_scope ON session_mode(scope, mode);

-- V5 群管理：警告计数（每个群对每个成员单独累计，群与群之间不共享）
CREATE TABLE IF NOT EXISTS member_warning (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_reason TEXT NOT NULL DEFAULT '',
    last_at REAL NOT NULL,
    PRIMARY KEY (group_id, user_id)
);

-- V5 群管理：待管理员确认的高风险操作。
-- 为什么必须有这张表：V5 要求"禁言、踢出和批量撤回等高影响操作，默认需要管理员确认"，
-- 而确认是跨越两条消息的（机器人说"要禁言 X，确认吗" -> 管理员答"确认"），
-- 所以状态必须落库，不能只放内存（重启就没了，管理员会一脸茫然）。
CREATE TABLE IF NOT EXISTS pending_action (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_id TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    requested_by TEXT NOT NULL DEFAULT '',
    requested_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'   -- pending | done | cancelled | expired
);
CREATE INDEX IF NOT EXISTS idx_pending_group ON pending_action(group_id, status, id DESC);

-- V5 群管理：动作审计日志。
-- V5 明确"不得伪造已经撤回、禁言或踢出成功"——所以每次动作都要留下
-- 尝试与结果的记录，失败也要记（含原因）。
CREATE TABLE IF NOT EXISTS moderation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_id TEXT NOT NULL DEFAULT '',
    operator_id TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',          -- ok | failed | skipped | pending
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_modlog_group ON moderation_log(group_id, id DESC);
"""


@dataclass
class UserProfile:
    user_id: str
    display_name: str = ""
    nickname: str = ""
    mode: str = "alice"
    first_seen: float = 0.0
    last_seen: float = 0.0
    message_count: int = 0
    notes: str = ""

    @property
    def preferred_name(self) -> str:
        return self.nickname or self.display_name or f"用户{self.user_id[-4:]}"


@dataclass
class MemoryFact:
    id: int
    user_id: str
    kind: str
    content: str
    source: str
    confidence: float
    created_at: float
    updated_at: float


@dataclass
class GroupConfig:
    group_id: str
    enabled: bool = True
    mode: str = ""
    keywords: List[str] = field(default_factory=list)
    rules: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0


class Storage:
    """线程安全的 SQLite 封装。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # 对话记录
    # ------------------------------------------------------------------
    def recent_user_group_messages(self, group_id: str, user_id: str,
                                   within_seconds: int = 120, limit: int = 12) -> List[str]:
        """取某成员在某群最近说过的话（只取 user 角色的内容）。

        用途：V5 反广告的"重复刷屏"判定。只看**时间窗内**的消息，
        否则几小时前说过一句一样的话也会被算成刷屏。
        """
        cutoff = time.time() - max(1, int(within_seconds))
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM conversation_log WHERE group_id = ? AND user_id = ? "
                "AND role = 'user' AND created_at >= ? ORDER BY id DESC LIMIT ?",
                (str(group_id), str(user_id), cutoff, int(limit))).fetchall()
        return [str(row["content"]) for row in reversed(rows)]

    def append_message(self, user_id: str, role: str, content: str, group_id: str = "",
                       created_at: Optional[float] = None) -> None:
        stamp = time.time() if created_at is None else created_at
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_log (user_id, group_id, role, content, created_at) VALUES (?,?,?,?,?)",
                (str(user_id), str(group_id), role, content, stamp),
            )

    def recent_messages(self, user_id: str, limit: int = 10, group_id: Optional[str] = None,
                        exclude_current: bool = False) -> List[Dict[str, Any]]:
        """读取用户最近的对话。group_id 为 None 时取该用户全部场景；传空串只取私聊。

        exclude_current=True 时丢弃最后一条 user 消息——用于组装模型上下文时排除"正在处理
        的这一条"（它已经写入日志了）。
        """
        sql = "SELECT role, content, group_id, created_at FROM conversation_log WHERE user_id = ?"
        params: List[Any] = [str(user_id)]
        if group_id is not None:
            sql += " AND group_id = ?"
            params.append(str(group_id))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit) + (1 if exclude_current else 0))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        items = [dict(row) for row in reversed(rows)]
        if exclude_current:
            for index in range(len(items) - 1, -1, -1):
                if items[index]["role"] == "user":
                    items.pop(index)
                    break
        return items

    def recent_group_messages(self, group_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, role, content, created_at FROM conversation_log "
                "WHERE group_id = ? ORDER BY id DESC LIMIT ?",
                (str(group_id), int(limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def trim_conversation(self, user_id: str, keep: int) -> int:
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM conversation_log WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM conversation_log WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
                (str(user_id), str(user_id), int(keep)),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # 用户档案
    # ------------------------------------------------------------------
    def touch_user(self, user_id: str, display_name: str = "", now: Optional[float] = None) -> UserProfile:
        stamp = time.time() if now is None else now
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM user_profile WHERE user_id = ?", (str(user_id),)).fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO user_profile (user_id, display_name, mode, first_seen, last_seen, message_count) "
                    "VALUES (?,?,?,?,?,1)",
                    (str(user_id), display_name, "alice", stamp, stamp),
                )
            else:
                cur.execute(
                    "UPDATE user_profile SET last_seen = ?, message_count = message_count + 1, "
                    "display_name = CASE WHEN ? <> '' THEN ? ELSE display_name END WHERE user_id = ?",
                    (stamp, display_name, display_name, str(user_id)),
                )
        return self.get_profile(str(user_id))  # type: ignore[return-value]

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (str(user_id),)).fetchone()
        if row is None:
            return None
        return UserProfile(
            user_id=row["user_id"], display_name=row["display_name"], nickname=row["nickname"],
            mode=row["mode"], first_seen=row["first_seen"], last_seen=row["last_seen"],
            message_count=row["message_count"], notes=row["notes"],
        )

    def ensure_profile(self, user_id: str, display_name: str = "", default_mode: str = "alice") -> UserProfile:
        profile = self.get_profile(user_id)
        if profile is not None:
            return profile
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO user_profile (user_id, display_name, mode, first_seen, last_seen, message_count) "
                "VALUES (?,?,?,?,?,0)",
                (str(user_id), display_name, default_mode, now, now),
            )
        return self.get_profile(str(user_id))  # type: ignore[return-value]

    def set_mode(self, user_id: str, mode: str, scope: str = SCOPE_PRIVATE) -> None:
        """设置某用户**在某个会话里**的模式。

        scope：'private' 表示私聊，群聊传群号。默认私聊。

        **只写会话记录，不碰 user_profile.mode（全局默认）**。
        这一点很关键：如果每次切换都顺手改全局默认，那"在 A 群切成助手"
        就会让 B 群和私聊也变成助手——等于没隔离。
        全局默认只在**私聊**切换时更新（私聊是用户自己的主场景），
        这样他第一次去的新群会沿用最近一次的选择，而不是突然跳回爱丽丝。
        """
        scope = str(scope or SCOPE_PRIVATE)
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO session_mode (user_id, scope, mode, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, scope) DO UPDATE SET mode = excluded.mode, "
                "updated_at = excluded.updated_at",
                (str(user_id), scope, mode, now),
            )
            if scope == SCOPE_PRIVATE:
                cur.execute(
                    "INSERT INTO user_profile (user_id, mode, first_seen, last_seen) VALUES (?,?,?,?) "
                    "ON CONFLICT(user_id) DO UPDATE SET mode = excluded.mode",
                    (str(user_id), mode, now, now),
                )

    def get_session_mode(self, user_id: str, scope: str) -> str:
        """取某会话里的模式；没设过返回空串（由上层决定回落）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT mode FROM session_mode WHERE user_id = ? AND scope = ?",
                (str(user_id), str(scope or SCOPE_PRIVATE)),
            ).fetchone()
        return str(row["mode"]) if row else ""

    def reset_session_mode(self, user_id: str, scope: str) -> None:
        """清除某会话的模式覆盖，让它回落到全局默认。"""
        with self._cursor() as cur:
            cur.execute("DELETE FROM session_mode WHERE user_id = ? AND scope = ?",
                        (str(user_id), str(scope or SCOPE_PRIVATE)))

    def migrate_profile_modes_to_sessions(self, default_scope: str = SCOPE_PRIVATE) -> int:
        """把老库里的 user_profile.mode 灌进 session_mode，避免升级后"模式全忘了"。

        只处理**还没有任何会话记录**的用户——已有会话记录的用户说明他升级后
        已经切过模式，不能覆盖。返回迁移条数。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.user_id, p.mode FROM user_profile p "
                "WHERE NOT EXISTS (SELECT 1 FROM session_mode s WHERE s.user_id = p.user_id)"
            ).fetchall()
        migrated = 0
        now = time.time()
        with self._cursor() as cur:
            for row in rows:
                mode = str(row["mode"] or "")
                if not mode:
                    continue
                cur.execute(
                    "INSERT OR IGNORE INTO session_mode (user_id, scope, mode, updated_at) "
                    "VALUES (?,?,?,?)",
                    (str(row["user_id"]), str(default_scope), mode, now),
                )
                migrated += 1
        return migrated

    def set_display_name(self, user_id: str, display_name: str) -> None:
        if not display_name:
            return
        with self._cursor() as cur:
            cur.execute("UPDATE user_profile SET display_name = ? WHERE user_id = ?", (display_name, str(user_id)))

    def set_nickname(self, user_id: str, nickname: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO user_profile (user_id, nickname, first_seen, last_seen) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET nickname = excluded.nickname",
                (str(user_id), nickname, time.time(), time.time()),
            )

    def iterate_profiles(self) -> Iterable[UserProfile]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM user_profile").fetchall()
        for row in rows:
            yield UserProfile(
                user_id=row["user_id"], display_name=row["display_name"], nickname=row["nickname"],
                mode=row["mode"], first_seen=row["first_seen"], last_seen=row["last_seen"],
                message_count=row["message_count"], notes=row["notes"],
            )

    # ------------------------------------------------------------------
    # 长期记忆
    # ------------------------------------------------------------------
    def add_memory(self, user_id: str, content: str, kind: str = "fact", source: str = "auto",
                   confidence: float = 0.6, now: Optional[float] = None) -> bool:
        stamp = time.time() if now is None else now
        content = content.strip()
        if not content:
            return False
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO memory_fact (user_id, kind, content, source, confidence, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id, content) DO UPDATE SET updated_at = excluded.updated_at, "
                "kind = excluded.kind, confidence = MAX(memory_fact.confidence, excluded.confidence)",
                (str(user_id), kind, content, source, float(confidence), stamp, stamp),
            )
            return cur.rowcount > 0

    def list_memory(self, user_id: str, limit: int = 50) -> List[MemoryFact]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_fact WHERE user_id = ? ORDER BY confidence DESC, updated_at DESC LIMIT ?",
                (str(user_id), int(limit)),
            ).fetchall()
        return [MemoryFact(row["id"], row["user_id"], row["kind"], row["content"], row["source"],
                           row["confidence"], row["created_at"], row["updated_at"]) for row in rows]

    def count_memory(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM memory_fact WHERE user_id = ?", (str(user_id),)).fetchone()
        return int(row["c"])

    def delete_memory(self, user_id: str, keyword: str = "") -> int:
        with self._cursor() as cur:
            if keyword:
                cur.execute("DELETE FROM memory_fact WHERE user_id = ? AND content LIKE ?",
                            (str(user_id), f"%{keyword}%"))
            else:
                cur.execute("DELETE FROM memory_fact WHERE user_id = ?", (str(user_id),))
            return cur.rowcount

    def enforce_memory_limit(self, user_id: str, limit: int) -> int:
        """超出上限时淘汰置信度最低、最久未更新的记忆。"""
        with self._cursor() as cur:
            count = cur.execute("SELECT COUNT(*) AS c FROM memory_fact WHERE user_id = ?", (str(user_id),)).fetchone()["c"]
            overflow = int(count) - int(limit)
            if overflow <= 0:
                return 0
            cur.execute(
                "DELETE FROM memory_fact WHERE id IN (SELECT id FROM memory_fact WHERE user_id = ? "
                "ORDER BY confidence ASC, updated_at ASC LIMIT ?)",
                (str(user_id), overflow),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # 群配置
    # ------------------------------------------------------------------
    def get_group(self, group_id: str) -> Optional[GroupConfig]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM group_config WHERE group_id = ?", (str(group_id),)).fetchone()
        if row is None:
            return None
        try:
            extra = json.loads(row["extra"] or "{}")
        except json.JSONDecodeError:
            extra = {}
        try:
            keywords = json.loads(row["keywords"] or "[]")
        except json.JSONDecodeError:
            keywords = []
        return GroupConfig(group_id=row["group_id"], enabled=bool(row["enabled"]), mode=row["mode"],
                           keywords=list(keywords), rules=row["rules"], extra=dict(extra),
                           updated_at=row["updated_at"])

    def upsert_group(self, group_id: str, enabled: Optional[bool] = None, mode: Optional[str] = None,
                     keywords: Optional[List[str]] = None, rules: Optional[str] = None,
                     extra: Optional[Dict[str, Any]] = None) -> GroupConfig:
        current = self.get_group(group_id) or GroupConfig(group_id=str(group_id))
        if enabled is not None:
            current.enabled = enabled
        if mode is not None:
            current.mode = mode
        if keywords is not None:
            current.keywords = keywords
        if rules is not None:
            current.rules = rules
        # 注意：extra 采用"合并"而不是"替换"。
        # 早期实现是整字典替换，导致 /授权 写入 approved 时把之前记录的管理员列表
        # 一并抹掉——这类"多来源写同一个 json 字段"的场景必须合并。
        # 想删除某个键，显式传 None。
        if extra is not None:
            merged = dict(current.extra)
            for key, value in extra.items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            current.extra = merged
        current.updated_at = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO group_config (group_id, enabled, mode, keywords, rules, extra, updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(group_id) DO UPDATE SET enabled = excluded.enabled, "
                "mode = excluded.mode, keywords = excluded.keywords, rules = excluded.rules, "
                "extra = excluded.extra, updated_at = excluded.updated_at",
                (str(group_id), 1 if current.enabled else 0, current.mode,
                 json.dumps(current.keywords, ensure_ascii=False), current.rules,
                 json.dumps(current.extra, ensure_ascii=False), current.updated_at),
            )
        return current

    # ------------------------------------------------------------------
    # 群级设置（V3）：存在 group_config.extra 里，避免频繁改表结构
    # ------------------------------------------------------------------
    def get_group_settings(self, group_id: str):
        from .config import GroupSettings
        record = self.get_group(group_id)
        return GroupSettings.from_extra(record.extra if record else {})

    def save_group_settings(self, group_id: str, settings) -> None:
        """把设置合并进 extra（upsert_group 的 extra 是合并语义，不会抹掉其它键）。"""
        self.upsert_group(group_id, extra=settings.to_extra())

    def list_groups(self) -> List[GroupConfig]:
        with self._lock:
            rows = self._conn.execute("SELECT group_id FROM group_config ORDER BY group_id").fetchall()
        return [cfg for cfg in (self.get_group(row["group_id"]) for row in rows) if cfg is not None]

    # ------------------------------------------------------------------
    # V5 群管理：警告计数 / 待确认操作 / 审计日志
    # ------------------------------------------------------------------
    def add_warning(self, group_id: str, user_id: str, reason: str = "") -> int:
        """给某成员加一次警告，返回累计次数。

        注意计数是 **(群, 用户)** 维度的：同一个人在不同群的警告互不影响。
        """
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO member_warning (group_id, user_id, count, last_reason, last_at) "
                "VALUES (?,?,1,?,?) "
                "ON CONFLICT(group_id, user_id) DO UPDATE SET count = count + 1, "
                "last_reason = excluded.last_reason, last_at = excluded.last_at",
                (str(group_id), str(user_id), str(reason)[:200], now),
            )
            row = cur.execute("SELECT count FROM member_warning WHERE group_id = ? AND user_id = ?",
                              (str(group_id), str(user_id))).fetchone()
        return int(row["count"]) if row else 0

    def get_warning_count(self, group_id: str, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM member_warning WHERE group_id = ? AND user_id = ?",
                (str(group_id), str(user_id))).fetchone()
        return int(row["count"]) if row else 0

    def list_warnings(self, group_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, count, last_reason, last_at FROM member_warning "
                "WHERE group_id = ? ORDER BY count DESC, last_at DESC LIMIT ?",
                (str(group_id), int(limit))).fetchall()
        return [dict(row) for row in rows]

    def clear_warnings(self, group_id: str, user_id: str = "") -> int:
        with self._cursor() as cur:
            if user_id:
                cur.execute("DELETE FROM member_warning WHERE group_id = ? AND user_id = ?",
                            (str(group_id), str(user_id)))
            else:
                cur.execute("DELETE FROM member_warning WHERE group_id = ?", (str(group_id),))
            return cur.rowcount or 0

    def create_pending_action(self, group_id: str, action: str, target_id: str = "",
                              payload: Optional[Dict[str, Any]] = None, reason: str = "",
                              requested_by: str = "") -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO pending_action (group_id, action, target_id, payload, reason, "
                "requested_by, requested_at) VALUES (?,?,?,?,?,?,?)",
                (str(group_id), str(action), str(target_id),
                 json.dumps(payload or {}, ensure_ascii=False), str(reason)[:200],
                 str(requested_by), time.time()),
            )
            return int(cur.lastrowid or 0)

    def get_pending_action(self, group_id: str, action_id: int = 0) -> Optional[Dict[str, Any]]:
        """取本群**最近一条待确认**的动作；给了 action_id 就精确取。"""
        with self._lock:
            if action_id:
                row = self._conn.execute(
                    "SELECT * FROM pending_action WHERE id = ? AND group_id = ? AND status = 'pending'",
                    (int(action_id), str(group_id))).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM pending_action WHERE group_id = ? AND status = 'pending' "
                    "ORDER BY id DESC LIMIT 1", (str(group_id),)).fetchone()
        return dict(row) if row else None

    def resolve_pending_action(self, action_id: int, status: str = "done") -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE pending_action SET status = ? WHERE id = ?",
                        (str(status), int(action_id)))

    def log_moderation(self, group_id: str, action: str, result: str, target_id: str = "",
                       operator_id: str = "", reason: str = "", detail: str = "") -> None:
        """记一条管理动作。失败也必须记（V5：不得伪造成功）。"""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO moderation_log (group_id, action, target_id, operator_id, reason, "
                "result, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (str(group_id), str(action), str(target_id), str(operator_id),
                 str(reason)[:200], str(result), str(detail)[:400], time.time()),
            )

    def recent_moderation(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT action, target_id, operator_id, reason, result, detail, created_at "
                "FROM moderation_log WHERE group_id = ? ORDER BY id DESC LIMIT ?",
                (str(group_id), int(limit))).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 额度 / 频率
    # ------------------------------------------------------------------
    def incr_counter(self, scope: str, scope_id: str, period: str, amount: int = 1) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO usage_counter (scope, scope_id, period, counter, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(scope, scope_id, period) DO UPDATE SET counter = counter + excluded.counter, "
                "updated_at = excluded.updated_at",
                (scope, str(scope_id), period, int(amount), time.time()),
            )
            row = cur.execute("SELECT counter FROM usage_counter WHERE scope = ? AND scope_id = ? AND period = ?",
                              (scope, str(scope_id), period)).fetchone()
        return int(row["counter"])

    def get_counter(self, scope: str, scope_id: str, period: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT counter FROM usage_counter WHERE scope = ? AND scope_id = ? AND period = ?",
                                    (scope, str(scope_id), period)).fetchone()
        return int(row["counter"]) if row else 0

    def set_counter(self, scope: str, scope_id: str, period: str, value: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO usage_counter (scope, scope_id, period, counter, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(scope, scope_id, period) DO UPDATE SET counter = excluded.counter, updated_at = excluded.updated_at",
                (scope, str(scope_id), period, int(value), time.time()),
            )

    # ------------------------------------------------------------------
    # 通用 KV（如"上次主动发言时间"）
    # ------------------------------------------------------------------
    def kv_set(self, key: str, value: Any) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO kv_store (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )

    def kv_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def stats(self) -> Dict[str, int]:
        with self._lock:
            def one(sql: str) -> int:
                return int(self._conn.execute(sql).fetchone()[0])
            return {
                "users": one("SELECT COUNT(*) FROM user_profile"),
                "groups": one("SELECT COUNT(*) FROM group_config"),
                "memory": one("SELECT COUNT(*) FROM memory_fact"),
                "messages": one("SELECT COUNT(*) FROM conversation_log"),
            }
