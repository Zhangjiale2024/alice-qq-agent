# -*- coding: utf-8 -*-
"""语音识别质量的配置与提示词测试（全部离线）。

**这个文件守着一组实测数据换来的结论。**

基准：12 条已知文本的语音（TTS 合成），用本地 faster-whisper `small` 转写，
按字符编辑距离算错误率：

    不加提示              错误率 40.6%   完全正确  0/12
    加"简体+数字"提示      错误率 15.9%   完全正确  5/12
    再加角色专名           错误率  1.0%   完全正确 11/12

**关键发现：主导错误是输出繁体字**（"今天天气看起来还不错" ->
"今天天氣看起來還不錯"），不是一开始以为的同音字。
一开始只观察到一次"爱丽丝 -> 艾利斯"就想去修同音字，方向是错的。

另外实测否决了两个候选方案：
  * 事后做简繁转换：7.0% —— 比直接提示（1.0%）差，还多一层机制
  * 自己写代码把阿拉伯数字转汉字：把"10点"改成了"一零点"，
    因为逐字替换不懂量级。提示模型就够（"3点" -> "三点"）。

还有一条：专名表里混进 ASCII（自动并入的 AGENT_NAME 里含 "QQ"）
担心会带坏中文输出，实测 1.0% / ASCII 混入 0 条，**没有影响**，
所以不加过滤——不加没有被证实需要的机制。

这里只测**配置与提示词拼接**（确定、离线、快）；
真实识别质量靠上面的实测数据说明，不放进单元测试（那需要模型和网络）。
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aliceqq import asr as asr_mod
from aliceqq.config import AppConfig, AsrConfig


class DefaultPromptTest(unittest.TestCase):
    def test_default_prompt_forces_simplified(self):
        # 只要求"简体"就能把错误率从 40.6% 压到 15.9%
        self.assertIn("简体", AsrConfig.initial_prompt)

    def test_default_prompt_asks_for_chinese_numerals(self):
        # 残留的一例是"三点" -> "3点"；提示模型比用代码转换可靠
        self.assertIn("汉字", AsrConfig.initial_prompt)

    def test_default_is_not_empty(self):
        self.assertTrue(AsrConfig.initial_prompt.strip())


class ClientPromptTest(unittest.TestCase):
    def _client(self, **kw):
        args = {"model_size": "small", "device": "cpu", "language": "zh"}
        args.update(kw)
        return asr_mod.LocalWhisperASR(**args)

    def test_prompt_is_carried_to_the_client(self):
        client = self._client(initial_prompt="请用简体中文")
        self.assertEqual(client.initial_prompt, "请用简体中文")

    def test_vocabulary_is_appended(self):
        client = self._client(initial_prompt="请用简体中文", vocabulary="爱丽丝")
        self.assertIn("请用简体中文", client.initial_prompt)
        self.assertIn("爱丽丝", client.initial_prompt)

    def test_vocabulary_alone_works_without_base_prompt(self):
        # 用户可能把 initial_prompt 显式配成空串（"不要提示"），
        # 但又配了专名——专名仍然要生效
        client = self._client(initial_prompt="", vocabulary="爱丽丝")
        self.assertIn("爱丽丝", client.initial_prompt)

    def test_no_prompt_no_vocabulary_is_empty(self):
        client = self._client(initial_prompt="", vocabulary="")
        self.assertEqual(client.initial_prompt, "")

    def test_whitespace_is_trimmed(self):
        client = self._client(initial_prompt="  请用简体中文  ", vocabulary="  爱丽丝  ")
        self.assertFalse(client.initial_prompt.startswith(" "))
        self.assertFalse(client.initial_prompt.endswith(" "))

    def test_ascii_in_vocabulary_is_kept(self):
        # 实测：混入 ASCII（如自动并入的 "爱丽丝 QQ 伙伴"）**没有**影响识别
        # （1.0% / 正确 11-12，ASCII 混入 0 条），所以不做过滤。
        # 这条用例把这个判断固定下来，免得以后有人"顺手"加一层过滤。
        client = self._client(vocabulary="爱丽丝、爱丽丝 QQ 伙伴")
        self.assertIn("QQ", client.initial_prompt)


class TranscribePassesPromptTest(unittest.TestCase):
    """静态检查：initial_prompt 真的传给了模型。

    和"工具 schema 没传给模型"那个 bug 同类——拼好了但没传，
    表现是完全静默（质量没变化，也不会报错）。
    """

    def test_transcribe_forwards_initial_prompt(self):
        src = inspect.getsource(asr_mod.LocalWhisperASR.transcribe)
        self.assertIn("initial_prompt", src)
        self.assertIn("self.initial_prompt", src)

    def test_prompt_is_only_added_when_non_empty(self):
        # 空串时不该传 initial_prompt=None 或空串给模型
        src = inspect.getsource(asr_mod.LocalWhisperASR.transcribe)
        self.assertIn("if self.initial_prompt:", src)


class BuildClientTest(unittest.TestCase):
    def test_build_client_forwards_prompt_and_vocabulary(self):
        src = inspect.getsource(asr_mod.build_client)
        self.assertIn("initial_prompt", src)
        self.assertIn("vocabulary", src)

    def test_missing_fields_do_not_break_build(self):
        """兼容只有部分字段的旧配置对象。

        **注意 `build_client` 收的是「外层 config」**（内部取 `config.asr`），
        不是 AsrConfig 本身——这和 `vision.build_client(config.vision)`
        收子配置的语义**正好相反**。我在调试时用一个套另一个，
        得到"静默返回 None"，一度以为引擎接线错了。
        """
        class MinimalSettings:
            enabled = True
            provider = "local"
            local_model = "small"
            local_device = "cpu"
            local_cache = ""
            timeout_seconds = 60
            language = "zh"

        class Outer:
            asr = MinimalSettings()

        client = asr_mod.build_client(Outer())
        self.assertIsNotNone(client)
        self.assertEqual(client.initial_prompt, "")

    def test_wrong_shape_returns_none_instead_of_raising(self):
        # 传错形态（直接传子配置）时不抛异常，但返回 None。
        # 这是既有行为；关键是**不要以为它一定会报错**。
        class NotOuter:
            enabled = True

        self.assertIsNone(asr_mod.build_client(NotOuter()))


class ConfigWiringTest(unittest.TestCase):
    """配置层：自动并入角色名。"""

    def test_character_name_is_auto_added(self):
        # 项目自己知道角色叫什么，不该指望用户去 vocabulary 里再抄一遍
        config = AppConfig.load()
        self.assertIn(config.bot.character, config.asr.vocabulary)

    def test_auto_vocabulary_can_be_disabled(self):
        src = inspect.getsource(AppConfig.load)
        self.assertIn("auto_vocabulary", src)

    def test_explicit_vocabulary_is_preserved(self):
        # 用户自己配的词不能被自动并入覆盖掉
        src = inspect.getsource(AppConfig.load)
        self.assertIn("ASR_VOCABULARY", src)

    def test_no_duplicate_words(self):
        config = AppConfig.load()
        words = [w.strip() for w in config.asr.vocabulary.split("、") if w.strip()]
        self.assertEqual(len(words), len(set(words)), "专名表里出现了重复词")


if __name__ == "__main__":
    unittest.main()
