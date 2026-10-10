"""唱歌功能的离线测试。

覆盖三块：
  1. **意图识别**——最容易出错的地方。第一版正则太宽，
     "我想听你唱《虫儿飞》"会抽出 `你唱《虫儿飞`、"爱丽丝唱首歌"会抽出 `首歌`，
     所以这里用一批真实说法卡住它。原则是**宁可漏判也不要乱唱**：
     一次重活要跑 60~90 秒并吃满 CPU，误触发比不触发更糟。
  2. **缓存**——同一首歌第二次点必须秒回，不能重跑。
  3. **任务执行**——通过注入假 runner，不真的跑 90 秒流水线。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

from aliceqq import singing


class DetectSongTest(unittest.TestCase):
    def test_clear_requests_are_detected(self):
        cases = {
            "唱一首反方向的钟": "反方向的钟",
            "我想听你唱《虫儿飞》": "虫儿飞",
            "来一首 千本桜": "千本桜",
            "给我唱个温柔的理由": "温柔的理由",
            "爱丽丝，唱《打上花火》给我听": "打上花火",
            "点歌 群青": "群青",
            "唱一首 反方向的钟 吧": "反方向的钟",
            "唱 群青": "群青",
        }
        for text, expect in cases.items():
            self.assertEqual(singing.detect_song(text), expect, f"输入：{text!r}")

    def test_chat_about_singing_is_not_a_request(self):
        """聊"唱歌"这件事不该触发一次 90 秒的重活。"""
        for text in ("你会唱歌吗", "唱歌好听吗", "唱歌怎么样", "喜欢唱歌",
                     "你唱得真好听", "有什么歌单", "推荐点歌", "这首歌叫什么名字",
                     "我们来合唱吧", "唱歌技巧是什么"):
            self.assertIsNone(singing.detect_song(text), f"不该触发：{text!r}")

    def test_requests_without_a_song_name_are_ignored(self):
        """说了"唱"但没给歌名——不知道该唱什么，不猜。"""
        for text in ("爱丽丝唱首歌", "唱一段吧", "唱首歌来听听", "爱丽丝唱一首歌",
                     "能不能唱首歌", "我想唱歌", "唱一个", ""):
            self.assertIsNone(singing.detect_song(text), f"不该触发：{text!r}")

    def test_unrelated_text_is_ignored(self):
        for text in ("今天天气不错", "你好", "帮我查一下资料"):
            self.assertIsNone(singing.detect_song(text))

    def test_overlong_input_is_rejected(self):
        """超长"歌名"多半是把整句话当歌名了，拒绝掉。"""
        self.assertIsNone(singing.detect_song("唱" + "啊" * 60))

    # ------------------------------------------------------------------
    # 以下四条来自 2026-10-10 的 QQ 截图——用户真实用词暴露的漏判。
    # 单测当时全绿，因为没想到"出处前缀"和"纠正句"这两种说法。
    # ------------------------------------------------------------------

    def test_origin_prefix_is_stripped(self):
        """「作品名 + 的」前缀不是歌名。

        截图实录：用户说"唱一首蔚蓝档案的《准备出发》"，
        机器人把"蔚蓝档案的"也当歌名去搜，结果放出《钱是万恶之源吗》。
        """
        cases = {
            "唱一首蔚蓝档案的《准备出发》": "准备出发",
            "唱一首星塔旅人的《Memory of Stella》": "Memory of Stella",
            "唱一首蔚蓝档案的《全速向光》": "全速向光",
            "再唱一首蔚蓝档案的《准备出发》": "准备出发",
        }
        for text, expect in cases.items():
            self.assertEqual(singing.detect_song(text), expect, f"输入：{text!r}")

    def test_correction_sentence_is_not_a_song_name(self):
        """带判断词的纠正句不是歌名。

        截图实录：用户说"让你唱全速向光不是反乌托邦"（在**纠正**机器人），
        整句当歌名去搜必然搜错——应当直接不触发。
        """
        for text in ("让你唱全速向光不是反乌托邦",
                     "唱群青不是这首",
                     "唱一首应该叫打上花火的",
                     "唱 温柔的理由 换成 群青"):
            self.assertIsNone(singing.detect_song(text), f"不该触发：{text!r}")

    def test_nested_brackets_take_the_inner_name(self):
        self.assertEqual(singing.detect_song("唱《蔚蓝档案的《准备出发》》"), "准备出发")


class SongCacheTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.cache = singing.SongCache(os.path.join(self._dir.name, "songs"))
        self._src_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._dir.cleanup()
        self._src_dir.cleanup()

    def _make_audio(self, name: str = "out.mp3") -> str:
        path = os.path.join(self._src_dir.name, name)
        with open(path, "wb") as fh:
            fh.write(b"ID3fake-audio-bytes")
        return path

    def test_miss_then_hit(self):
        self.assertIsNone(self.cache.lookup("反方向的钟"))
        stored = self.cache.store("反方向的钟", self._make_audio())
        self.assertIsNotNone(stored)
        self.assertTrue(os.path.isfile(stored))
        self.assertEqual(self.cache.lookup("反方向的钟"), stored)

    def test_key_ignores_spaces_and_case(self):
        """空格和大小写不该造成两次缓存未命中（否则会重复跑 90 秒）。"""
        stored = self.cache.store("千本桜", self._make_audio())
        self.assertEqual(self.cache.lookup(" 千本桜 "), stored)
        stored2 = self.cache.store("Blue Bird", self._make_audio("b.mp3"))
        self.assertEqual(self.cache.lookup("blue bird"), stored2)

    def test_dangerous_song_name_is_sanitised(self):
        """歌名里带路径分隔符不能写到缓存目录外面去。"""
        stored = self.cache.store("../../evil", self._make_audio())
        self.assertIsNotNone(stored)
        root = os.path.abspath(self.cache.root)
        self.assertTrue(os.path.abspath(stored).startswith(root),
                        f"缓存文件跑到目录外了：{stored}")

    def test_missing_source_is_not_cached(self):
        self.assertIsNone(self.cache.store("不存在", "/no/such/file.mp3"))


class RunnerTest(unittest.TestCase):
    """任务执行：注入假的 sing.py，不真的跑流水线。"""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.script = os.path.join(self._dir.name, "sing.py")
        self.audio = os.path.join(self._dir.name, "song_成品.mp3")
        with open(self.audio, "wb") as fh:
            fh.write(b"ID3audio")

    def tearDown(self):
        self._dir.cleanup()

    def _write_script(self, body: str) -> None:
        with open(self.script, "w", encoding="utf-8") as fh:
            fh.write(body)

    def _runner(self):
        return singing.make_runner(self.script, "python", timeout=30)

    def test_success_parses_audio_path_and_title(self):
        # 用 repr() 写路径：Windows 路径里的反斜杠如果直接塞进字符串字面量，
        # `\D`/`\a`/`\t` 会被当成转义序列，路径当场被改写
        # （第一次跑这个用例时就踩到了：D:\...\alice -> D:...lice + 响铃）。
        self._write_script(
            "import sys\n"
            "print('选中： 《虫儿飞》爱丽丝翻唱')\n"
            f"print('成品：' + {self.audio!r})\n")
        result = self._runner()("虫儿飞")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.audio, self.audio)
        self.assertIn("虫儿飞", result.title)
        self.assertGreaterEqual(result.elapsed, 0.0)

    def test_failure_reports_real_reason(self):
        """失败必须带回脚本输出的**真实**原因，不能只说"失败了"。"""
        self._write_script(
            "import sys\n"
            "print('没找到符合条件的爱丽丝翻唱。')\n"
            "print('（只接受标题里同时有『爱丽丝』和『翻唱』的视频）')\n"
            "sys.exit(1)\n")
        result = self._runner()("不存在的歌")
        self.assertFalse(result.ok)
        self.assertIn("没找到", result.error)

    def test_missing_output_file_is_failure(self):
        """脚本没报错但没产出文件，同样算失败（不能假装成功）。"""
        self._write_script("print('成品：/no/such/file.mp3')\n")
        result = self._runner()("歌")
        self.assertFalse(result.ok)

    def test_missing_script_is_reported(self):
        # 不删文件，直接把 runner 指向一个不存在的脚本路径——
        # 删文件那版会被 setUp 的临时目录行为干扰（第一版就是这么挂的）。
        runner = singing.make_runner(
            os.path.join(self._dir.name, "根本没有这个脚本.py"), "python")
        result = runner("歌")
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error)

    def test_timeout_is_reported(self):
        self._write_script("import time\ntime.sleep(30)\n")
        result = singing.make_runner(self.script, "python", timeout=1)("歌")
        self.assertFalse(result.ok)
        self.assertIn("超时", result.error)

    def test_parse_failure_does_not_raise(self):
        """脚本输出乱七八糟也不能抛异常——后台任务必须稳。"""
        self._write_script("print('完全不是约定的格式')\n")
        result = self._runner()("歌")
        self.assertFalse(result.ok)
        self.assertTrue(result.error)


class CancelTest(unittest.TestCase):
    """取消：用户说"停"时要真的停。

    截图实录：用户发了"停"，机器人还是把歌发出来了——
    因为结果已经在返回路上。所以取消要做两件事：
    **杀掉子进程** 和 **让已跑完的结果失效**。
    """

    def setUp(self):
        singing.reset_cancel()

    def tearDown(self):
        singing.reset_cancel()

    def test_request_cancel_marks_previous_token_stale(self):
        token = singing.current_token()
        self.assertFalse(singing.is_stale(token))
        singing.request_cancel()
        self.assertTrue(singing.is_stale(token),
                        "取消后，先前拿到的 token 必须失效（否则结果照样发出去）")

    def test_new_job_supersedes_old(self):
        """用户改主意点了另一首：旧结果也要失效。"""
        old = singing.current_token()
        singing.request_cancel()             # 新请求会先作废旧任务
        new = singing.current_token()
        self.assertTrue(singing.is_stale(old))
        self.assertFalse(singing.is_stale(new))

    def test_cancel_flag_is_visible_to_worker(self):
        self.assertFalse(singing._CANCEL.is_set())
        singing.request_cancel()
        self.assertTrue(singing._CANCEL.is_set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
