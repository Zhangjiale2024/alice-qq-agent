"""V2 知识库测试：优先级、容量控制、上传/删除、权限。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aliceqq import knowledge
from aliceqq.config import PROJECT_ROOT
from aliceqq.engine import IncomingMessage, MODE_ALICE
from tests.test_engine import group, make_engine, private


class KnowledgeLoadTest(unittest.TestCase):
    def _tmpdir(self):
        # 不能用系统临时目录：沙箱下那里不允许 sqlite/文件操作
        base = PROJECT_ROOT / "data" / "test-kb" / f"case-{id(self)}"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def test_priority_order(self):
        base = self._tmpdir()
        for name, body in (("20_world.md", "# 世界\n世界观内容"),
                           ("10_alice.md", "# 角色\n角色内容"),
                           ("90_upload.md", "# 上传\n上传内容")):
            (base / name).write_text(body, encoding="utf-8")
        loaded = knowledge.load(base)
        names = [d.name for d in loaded.docs]
        self.assertEqual(names, ["10_alice.md", "20_world.md", "90_upload.md"])

    def test_char_limit_drops_low_priority(self):
        base = self._tmpdir()
        (base / "10_alice.md").write_text("# A\n" + "甲" * 200, encoding="utf-8")
        (base / "20_world.md").write_text("# B\n" + "乙" * 200, encoding="utf-8")
        loaded = knowledge.load(base, max_chars=260, max_file_chars=1000)
        self.assertEqual([d.name for d in loaded.docs], ["10_alice.md"])
        self.assertIn("20_world.md", loaded.dropped)

    def test_file_truncation(self):
        base = self._tmpdir()
        (base / "10_big.md").write_text("# 大\n" + "字" * 500, encoding="utf-8")
        loaded = knowledge.load(base, max_chars=5000, max_file_chars=50)
        self.assertTrue(loaded.docs[0].truncated)
        self.assertIn("已截断", loaded.docs[0].content)

    def test_render_marks_source_kind(self):
        base = self._tmpdir()
        (base / "90_up.md").write_text("# 上传标题\n内容", encoding="utf-8")
        text = knowledge.render(knowledge.load(base))
        self.assertIn("所有者上传", text)

    def test_safe_filename(self):
        self.assertEqual(knowledge.safe_filename("活动规则", prefix="90"), "90_活动规则.md")
        self.assertTrue(knowledge.safe_filename("../etc/passwd", prefix="90").endswith(".md"))
        self.assertNotIn("/", knowledge.safe_filename("../etc/passwd"))

    def test_delete_refuses_path_traversal(self):
        base = self._tmpdir()
        (base / "10_x.md").write_text("x", encoding="utf-8")
        self.assertFalse(knowledge.delete_doc(base, "../config.py"))
        self.assertFalse(knowledge.delete_doc(base, "不存在.md"))
        self.assertTrue(knowledge.delete_doc(base, "10_x.md"))


class KnowledgeUploadCommandTest(unittest.TestCase):
    def test_owner_can_upload_and_it_reaches_prompt(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["10001"]
        _, outcome = private("/资料上传 活动规则｜本期活动到 5 月 20 日结束", engine=engine)
        self.assertIn("已保存资料", outcome.reply)
        files = list(engine.config.knowledge_path.glob("90_*.md"))
        self.assertTrue(files)
        # 上传内容必须出现在后续提示词里
        _, _, variables = engine.build_messages(
            IncomingMessage(user_id="10001", text="你好", is_private=True, user_name="阿星"), MODE_ALICE)
        self.assertIn("本期活动到 5 月 20 日结束", variables["knowledge_base"])
        for path in files:
            path.unlink()

    def test_non_owner_cannot_upload(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["99999"]
        _, outcome = private("/资料上传 标题｜内容内容内容", user="10001", engine=engine)
        self.assertIn("OWNER_IDS", outcome.reply)

    def test_upload_rejects_sensitive_content(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["10001"]
        _, outcome = private("/资料上传 账号｜我的支付密码是 123456", engine=engine)
        self.assertIn("敏感信息", outcome.reply)

    def test_upload_requires_content(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["10001"]
        _, outcome = private("/资料上传", engine=engine)
        self.assertIn("用法", outcome.reply)

    def test_knowledge_list_shows_loaded_docs(self):
        engine = make_engine()
        engine.config.bot.owner_ids = ["10001"]
        _, outcome = private("/资料", engine=engine)
        self.assertIn("10_alice.md", outcome.reply)
        self.assertIn("合计", outcome.reply)

    def test_upload_disabled(self):
        engine = make_engine(features__knowledge_upload=False)
        engine.config.bot.owner_ids = ["10001"]
        _, outcome = private("/资料上传 标题｜内容内容内容", engine=engine)
        self.assertIn("未启用", outcome.reply)


class OutputRedactionTest(unittest.TestCase):
    def test_phone_number_redacted_in_reply(self):
        engine = make_engine()

        class PhoneLLM:
            name = "phone"

            def chat(self, messages, *, config, tools=None):
                from aliceqq.llm import Reply
                return Reply(text="联系我 13800138000 就行", provider="phone", model="fake")

        engine.llm = PhoneLLM()
        _, outcome = private("给我个联系方式", engine=engine)
        self.assertNotIn("13800138000", outcome.reply)
        self.assertIn("隐藏", outcome.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
