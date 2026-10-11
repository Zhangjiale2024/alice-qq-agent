# -*- coding: utf-8 -*-
"""图片理解（多模态）测试。

**全部离线**：视觉接口用假的 transport 替代，不发真实请求。
这一点和项目其它测试一致——核心逻辑必须能在没有网络、
没有 API Key 的机器上验证。

覆盖的是那次改动的核心：用户发的图要真的到达模型。
在此之前，图片段被适配层丢掉、消息里没有 images 字段、
`Message` 类型也装不下多模态内容，所以"理解图片"是做不到的。
"""
from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aliceqq import policy
from aliceqq import vision
from aliceqq.config import VisionConfig
from aliceqq.engine import IncomingMessage
from tests.test_engine import make_engine

# 带魔数的假图片，用来验证 MIME 判断与 data URI 编码
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_net(target, payload=None, error=None):
    """替换 vision 模块里的 net.open，让视觉调用离线可控。返回记录到的调用列表。

    **注意不要用 lambda 再包一层**（`lambda req, t: fake_open(req, t)`）——
    那样第一个位置参数会被当成 `fake_open` 传进去，`m.open` 就成了签名
    对不上的函数，一调用就 TypeError。第一版正是这么写的，
    结果一半用例红在这里。
    """
    calls = []

    def fake_open(request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        calls.append({"url": request.full_url, "body": body,
                      "headers": dict(request.headers), "timeout": timeout})
        if error is not None:
            raise error
        data = payload if payload is not None else {
            "choices": [{"message": {"content": "一只猫趴在键盘上"}}]}
        return _FakeResponse(json.dumps(data).encode("utf-8"))

    target.open = fake_open
    return calls


class ImageRefTest(unittest.TestCase):
    """图片引用的编码与类型判断。"""

    def test_mime_is_detected_from_magic_bytes(self):
        # 不能靠扩展名：QQ 给的 url 常常没有扩展名，file 名也未必对
        for data, expect in ((PNG, "image/png"), (JPG, "image/jpeg"), (GIF, "image/gif")):
            uri = vision.ImageRef(blob=data).to_data_uri()
            self.assertTrue(uri.startswith("data:%s;base64," % expect), uri[:40])

    def test_unknown_format_falls_back_to_png(self):
        uri = vision.ImageRef(blob=b"whatever-not-an-image").to_data_uri()
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_data_uri_round_trips_the_bytes(self):
        uri = vision.ImageRef(blob=PNG).to_data_uri()
        decoded = base64.b64decode(uri.split(",", 1)[1])
        self.assertEqual(decoded, PNG)

    def test_existing_data_uri_is_passed_through(self):
        given = "data:image/webp;base64,AAAA"
        self.assertEqual(vision.ImageRef(url=given).to_data_uri(), given)

    def test_http_url_is_preferred_over_re_encoding(self):
        # 有远程地址时不该先下载再转 base64：省一次流量和时间
        ref = vision.ImageRef(url="https://example.com/a.jpg", blob=PNG)
        self.assertEqual(ref.to_url_or_uri(), "https://example.com/a.jpg")

    def test_local_path_is_read_and_encoded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            path.write_bytes(PNG)
            uri = vision.ImageRef(path=str(path)).to_data_uri()
            self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_missing_file_yields_empty_uri(self):
        ref = vision.ImageRef(path=r"D:\definitely\not\here.png")
        self.assertEqual(ref.to_data_uri(), "")

    def test_oversized_image_is_rejected(self):
        # 超过上限就不编码：base64 后还要涨 1/3，硬发容易超时又费钱
        big = PNG + b"\x00" * (vision.MAX_IMAGE_BYTES + 10)
        self.assertEqual(vision.ImageRef(blob=big).to_data_uri(), "")


class VisionClientTest(unittest.TestCase):
    """调用视觉接口。用假的 net.open，不发真实请求。"""

    def setUp(self):
        self._real_open = vision.net.open

    def tearDown(self):
        vision.net.open = self._real_open

    def _client(self, **kw):
        args = {"base_url": "https://example.com/v1", "api_key": "k", "model": "m"}
        args.update(kw)
        return vision.VisionClient(**args)

    def test_unconfigured_client_reports_instead_of_calling(self):
        calls = _patch_net(vision.net)
        client = vision.VisionClient("", "", "")
        result = client.describe([vision.ImageRef(blob=PNG)])
        self.assertFalse(result.ok)
        self.assertIn("没有配置", result.error)
        self.assertEqual(calls, [], "没配置时不该发出任何请求")

    def test_successful_describe(self):
        _patch_net(vision.net)
        result = self._client().describe([vision.ImageRef(blob=PNG)])
        self.assertTrue(result.ok)
        self.assertIn("猫", result.description)
        self.assertEqual(result.images_seen, 1)

    def test_request_uses_multimodal_content_shape(self):
        # 这是接线的关键：content 必须是数组，且带 image_url。
        # 写成纯字符串的话，视觉模型会当普通文本处理、完全不看图。
        calls = _patch_net(vision.net)
        self._client().describe([vision.ImageRef(blob=PNG)])
        content = calls[0]["body"]["messages"][0]["content"]
        self.assertIsInstance(content, list)
        kinds = [part["type"] for part in content]
        self.assertEqual(kinds[0], "text")
        self.assertIn("image_url", kinds)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png"))

    def test_api_key_is_sent_as_bearer(self):
        calls = _patch_net(vision.net)
        self._client(api_key="secret123").describe([vision.ImageRef(blob=PNG)])
        self.assertEqual(calls[0]["headers"].get("Authorization"), "Bearer secret123")

    def test_multiple_images_are_all_sent(self):
        calls = _patch_net(vision.net)
        self._client().describe([vision.ImageRef(blob=PNG), vision.ImageRef(blob=JPG)])
        content = calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(sum(1 for p in content if p["type"] == "image_url"), 2)

    def test_too_many_images_are_capped_and_noted(self):
        # 群聊里连发多图很常见，不设上限成本会翻倍。
        # 关键是被截掉时要**说出来**，否则会以为九张都看过了。
        calls = _patch_net(vision.net)
        many = [vision.ImageRef(blob=PNG) for _ in range(vision.MAX_IMAGES_PER_TURN + 4)]
        result = self._client().describe(many)
        content = calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(sum(1 for p in content if p["type"] == "image_url"),
                         vision.MAX_IMAGES_PER_TURN)
        self.assertEqual(result.skipped, 4)
        self.assertIn("没看", result.note)

    def test_all_images_unreadable_reports_failure(self):
        calls = _patch_net(vision.net)
        result = self._client().describe([vision.ImageRef(path=r"D:\nope.png")])
        self.assertFalse(result.ok)
        self.assertEqual(calls, [], "读不到图就不该发请求")

    def test_http_error_is_reported_not_raised(self):
        import urllib.error

        _patch_net(vision.net, error=urllib.error.HTTPError(
            "u", 400, "bad", {}, None))
        result = self._client().describe([vision.ImageRef(blob=PNG)])
        self.assertFalse(result.ok)
        self.assertIn("400", result.error)

    def test_connection_error_is_reported_not_raised(self):
        import urllib.error

        _patch_net(vision.net, error=urllib.error.URLError("no route"))
        result = self._client().describe([vision.ImageRef(blob=PNG)])
        self.assertFalse(result.ok)
        self.assertIn("连不上", result.error)

    def test_empty_model_reply_is_treated_as_failure(self):
        _patch_net(vision.net, payload={"choices": [{"message": {"content": "  "}}]})
        result = self._client().describe([vision.ImageRef(blob=PNG)])
        self.assertFalse(result.ok)


class BuildClientTest(unittest.TestCase):
    """按配置造客户端。这里踩过一个坑，专门写了用例守住。"""

    def test_disabled_returns_none(self):
        self.assertIsNone(vision.build_client(VisionConfig(enabled=False)))

    def test_enabled_but_incomplete_returns_none(self):
        self.assertIsNone(vision.build_client(
            VisionConfig(enabled=True, base_url="https://x/v1", api_key="", model="m")))

    def test_fully_configured_returns_client(self):
        client = vision.build_client(VisionConfig(
            enabled=True, base_url="https://x/v1", api_key="k", model="m"))
        self.assertIsNotNone(client)
        self.assertTrue(client.configured)

    def test_reads_lowercase_field_names(self):
        # **这是实测踩到的 bug**：build_client 第一版用
        # `getattr(config, "vision_enabled")` 去取，但传进来的是
        # VisionConfig 子配置，字段叫 `enabled`。getattr 带了默认值，
        # 取不到就静默返回 False —— 于是永远返回 None，**没有任何报错**。
        # 这个用例就是为了让那种写法再出现时立刻红掉。
        cfg = VisionConfig(enabled=True, base_url="https://x/v1",
                           api_key="k", model="m")
        self.assertFalse(hasattr(cfg, "vision_enabled"),
                         "VisionConfig 不该有 vision_ 前缀的字段")
        self.assertIsNotNone(vision.build_client(cfg))


class RenderForModelTest(unittest.TestCase):
    """注入提示词的文本。看不见时**必须也注入**，这是护栏的关键。"""

    def test_success_includes_description(self):
        text = vision.render_for_model(vision.VisionResult(True, description="一只猫"))
        self.assertIn("一只猫", text)

    def test_failure_still_injects_telling_model_it_cannot_see(self):
        # 如果不注入任何东西，模型会以为没有图片这回事，
        # 转而根据用户的话瞎猜图里有什么。
        text = vision.render_for_model(vision.VisionResult(False, error="没有配置视觉模型"))
        self.assertIn("没有看到图片内容", text)
        self.assertIn("不要猜测", text)

    def test_note_about_skipped_images_is_included(self):
        text = vision.render_for_model(vision.VisionResult(
            True, description="画面", skipped=2, note="（一次最多看 3 张，后面 2 张没看）"))
        self.assertIn("没看", text)


class EngineVisionTest(unittest.TestCase):
    """引擎侧的接线：图片要进消息、进上下文、进配额。"""

    def _engine(self):
        engine = make_engine()
        engine.config.vision = VisionConfig(
            enabled=True, base_url="https://example.com/v1",
            api_key="k", model="m")
        return engine

    def setUp(self):
        self._real_open = vision.net.open

    def tearDown(self):
        vision.net.open = self._real_open

    def test_message_can_carry_images(self):
        msg = IncomingMessage(user_id="1", text="看看", images=[vision.ImageRef(blob=PNG)])
        self.assertTrue(msg.has_images)

    def test_message_without_images_is_normal(self):
        self.assertFalse(IncomingMessage(user_id="1", text="你好").has_images)

    def test_no_images_means_no_vision_block(self):
        engine = self._engine()
        msg = IncomingMessage(user_id="1", text="你好", is_private=True, user_name="x")
        self.assertEqual(engine._vision_block(msg), "")

    def test_engine_without_client_tells_model_it_cannot_see(self):
        # 没配视觉模型时不能静默：要让模型知道"有图但我看不到"
        engine = make_engine()
        engine.vision_client = None
        msg = IncomingMessage(user_id="1", text="看看", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        engine._pending_vision = None
        outcome_stub = type("O", (), {"debug": {}})()
        engine._analyze_images(msg, outcome_stub)
        block = engine._vision_block(msg)
        self.assertIn("没有看到图片内容", block)
        self.assertFalse(outcome_stub.debug["vision"]["ok"])

    def test_engine_describes_images_and_injects_result(self):
        _patch_net(vision.net, payload={
            "choices": [{"message": {"content": "红色边框，写着 ALICE-7391"}}]})
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)
        msg = IncomingMessage(user_id="1", text="看看", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        outcome = engine.handle(msg)
        self.assertTrue(outcome.debug["vision"]["ok"])
        self.assertIn("ALICE-7391", engine._vision_block(msg))

    def test_vision_result_does_not_leak_across_turns(self):
        # 和检索那边踩过的串轮坑同类。图片更敏感：
        # 把 A 发的图描述给 B 看，等于泄漏内容。
        _patch_net(vision.net, payload={
            "choices": [{"message": {"content": "某人的私密截图"}}]})
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)

        first = IncomingMessage(user_id="1", text="看图", is_private=True, user_name="a",
                                images=[vision.ImageRef(blob=PNG)])
        engine.handle(first)
        self.assertIn("私密截图", engine._vision_block(first))

        second = IncomingMessage(user_id="2", text="你好", is_private=True, user_name="b")
        engine.handle(second)
        self.assertEqual(engine._vision_block(second), "",
                         "第二轮没有图片时不能带上上一轮的图片描述")

    def test_vision_failure_does_not_break_the_reply(self):
        import urllib.error

        _patch_net(vision.net, error=urllib.error.URLError("boom"))
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)
        msg = IncomingMessage(user_id="1", text="看看这张", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        outcome = engine.handle(msg)          # 不该抛异常
        self.assertFalse(outcome.debug["vision"]["ok"])
        self.assertTrue(outcome.reply, "视觉失败也必须照常回复")

    def test_vision_quota_is_counted_separately(self):
        # 图片理解按张计费，和"生成图片"是两件事，额度要分开算——
        # 否则群聊里有人刷图，会把生成图片的额度一起吃掉。
        _patch_net(vision.net)
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)
        engine.config.quota.daily_vision = 5
        msg = IncomingMessage(user_id="1", text="看图", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        engine.handle(msg)
        period = policy._period_day()
        used = engine.storage.get_counter("quota_vision_user", "1", period)
        self.assertEqual(used, 1, "看过图之后应当消耗 1 次 vision 配额")
        # 而且不该串到别的能力上去
        self.assertEqual(engine.storage.get_counter("quota_image_user", "1", period), 0)
        self.assertEqual(engine.storage.get_counter("quota_search_user", "1", period), 0)

    def test_failed_vision_does_not_consume_quota(self):
        # 调用失败还扣额度，用户会觉得"什么都没看到却少了次数"
        import urllib.error

        _patch_net(vision.net, error=urllib.error.URLError("boom"))
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)
        msg = IncomingMessage(user_id="1", text="看图", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        engine.handle(msg)
        period = policy._period_day()
        self.assertEqual(engine.storage.get_counter("quota_vision_user", "1", period), 0)

    def test_vision_disabled_by_quota_says_so(self):
        engine = self._engine()
        engine.vision_client = vision.build_client(engine.config.vision)
        engine.config.quota.daily_vision = 0
        # 额度为 0 时 feature_quota_cap 返回 0，通常表示"不限"，
        # 这里改成用一个必定拦下的方式：直接伪造门禁
        engine._feature_allowed = lambda message, feature: False
        engine._last_block_reason = "测试用：额度不足"
        msg = IncomingMessage(user_id="1", text="看图", is_private=True, user_name="x",
                              images=[vision.ImageRef(blob=PNG)])
        stub = type("O", (), {"debug": {}})()
        engine._analyze_images(msg, stub)
        self.assertTrue(stub.debug["vision"].get("blocked"))
        self.assertIn("看不到", engine._vision_block(msg))


class PromptTemplateTest(unittest.TestCase):
    """模板里必须有占位符，否则注入的变量会被静默丢掉。"""

    def test_roleplay_template_has_vision_placeholder(self):
        root = Path(__file__).resolve().parent.parent
        text = (root / "prompts" / "alice_system.md").read_text(encoding="utf-8")
        self.assertIn("{{vision_result}}", text)

    def test_assistant_template_has_vision_placeholder(self):
        root = Path(__file__).resolve().parent.parent
        text = (root / "prompts" / "assistant_system.md").read_text(encoding="utf-8")
        self.assertIn("{{vision_result}}", text)

    def test_templates_forbid_guessing_when_image_is_unavailable(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("alice_system.md", "assistant_system.md"):
            text = (root / "prompts" / name).read_text(encoding="utf-8")
            # 两个模板措辞不同（一个写"不要去猜"、一个写"不要猜测"），
            # 所以断言共同的部分：都要求"看不到就说看不到"
            self.assertIn("看不到", text, "%s 缺少『看不到要说出来』的约束" % name)
            self.assertIn("{{vision_result}}", text)


if __name__ == "__main__":
    unittest.main()
