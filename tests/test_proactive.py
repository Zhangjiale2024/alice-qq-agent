"""主动关心的约束测试：默认要足够保守。"""
from __future__ import annotations

import time
import unittest

from tests.test_engine import make_engine, private
from aliceqq import proactive


class ProactivePolicyTest(unittest.TestCase):
    # 真实的 QQ 号长度（6~12 位）。原来这里用 `private()` 的默认值 "10001"，
    # 而 10001 正是 README/.env.example 里的**示例号**——实测 QQ 侧对它一律回
    # retcode=1200「无法获取用户信息」，于是主动关心每轮重试、日志被刷满。
    # 生产代码现在会过滤掉这类短号（见 is_reachable_user），
    # 所以测试也**必须用真实长度的号**，否则测的是一个永远不会被发送的用户。
    QQ = "100010001"

    def _idle_engine(self, hours_ago: float = 10.0, user: str = "", **overrides):
        # 注意 1：make_engine 默认关闭主动功能（避免其它用例被后台行为干扰），所以这里必须显式打开。
        # 注意 2：默认安静时段包含 23 点，如果用例依赖真实钟点，晚上跑就会必然失败。
        #        测试必须与时间无关，所以这里强制清空安静时段；
        #        「安静时段会拦下发送」由 test_quiet_hours_block_sending 专门覆盖。
        options = {"proactive__enabled": True, "proactive__quiet_hours": []}
        options.update(overrides)
        engine = make_engine(**options)
        private("你好", user=user or self.QQ, engine=engine,
                at=time.time() - hours_ago * 3600)
        return engine

    def test_disabled_produces_nothing(self):
        engine = self._idle_engine(proactive__enabled=False)
        self.assertEqual(proactive.plan(engine.config, engine.storage), [])

    def test_frequency_off_produces_nothing(self):
        engine = self._idle_engine(proactive__frequency="off")
        self.assertEqual(proactive.plan(engine.config, engine.storage), [])

    def test_recent_user_is_not_disturbed(self):
        engine = self._idle_engine(hours_ago=0.1)
        self.assertEqual(proactive.plan(engine.config, engine.storage), [])

    def test_idle_user_gets_one_task(self):
        engine = self._idle_engine(hours_ago=10.0)
        tasks = proactive.plan(engine.config, engine.storage)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].user_id, self.QQ)
        self.assertIn("小星", tasks[0].text)

    def test_daily_cap_respected(self):
        engine = self._idle_engine(hours_ago=10.0, proactive__max_per_user_per_day=1)
        tasks = proactive.plan(engine.config, engine.storage)
        self.assertEqual(len(tasks), 1)
        proactive.mark_sent(engine.storage, tasks[0].user_id)
        self.assertEqual(proactive.plan(engine.config, engine.storage), [])

    def test_quiet_hours_block_sending(self):
        engine = self._idle_engine(hours_ago=10.0, proactive__quiet_hours=list(range(0, 24)))
        self.assertTrue(proactive.in_quiet_hours(engine.config))
        self.assertEqual(proactive.plan(engine.config, engine.storage), [])

    def test_current_hour_is_never_accidentally_in_quiet_hours(self):
        # 防止再次出现"用例依赖真实钟点"的缺陷：默认夹具不该落在安静时段里
        engine = self._idle_engine(hours_ago=10.0)
        self.assertFalse(proactive.in_quiet_hours(engine.config),
                         "测试夹具落进了安静时段，用例会随运行时间随机失败")

    def test_template_placeholders_are_filled(self):
        engine = self._idle_engine(hours_ago=10.0)
        tasks = proactive.plan(engine.config, engine.storage)
        self.assertNotIn("{name}", tasks[0].text)
        self.assertNotIn("{hours}", tasks[0].text)
        self.assertNotIn("{minutes}", tasks[0].text)

    # ------------------------------------------------------------------
    # 以下覆盖 2026-10-10 从真实日志里发现的两个缺陷。
    # 这些用例存在的意义：那两个 bug 都**只在真实运行时才暴露**——
    # 单测当时全绿，因为测试数据恰好用了会被 QQ 拒绝的示例号。
    # ------------------------------------------------------------------

    def test_placeholder_qq_ids_are_not_treated_as_reachable(self):
        """示例号不能被当成可发送用户。

        日志实录：主动关心反复给 10001/10002 发送，QQ 一律回
        `retcode=1200 无法获取用户信息`。它们是 README 里的示例号，
        因为开发时用 `--user 10001` 试过而进了档案库。
        """
        for bad in ("10001", "10002", "1", "123", "1234567890123", "abc", "", "10001a"):
            self.assertFalse(proactive.is_reachable_user(bad),
                             f"{bad!r} 不该被判定为可发送的 QQ 号")
        for good in ("100010001", "100020002", "123456789012"):
            self.assertTrue(proactive.is_reachable_user(good),
                            f"{good!r} 应该是可发送的 QQ 号")

    def test_placeholder_profile_produces_no_task(self):
        """档案库里就算有示例号，也不该产生主动关心任务。"""
        engine = self._idle_engine(hours_ago=10.0)      # 真实号，会产生任务
        private("你好", user="10001", engine=engine,
                at=time.time() - 10 * 3600)             # 再塞一个示例号
        tasks = proactive.plan(engine.config, engine.storage)
        self.assertEqual([t.user_id for t in tasks], [self.QQ],
                         "示例号 10001 不该出现在发送列表里")

    def test_failed_send_enters_backoff(self):
        """发送失败过的用户要退避，不能每轮都重撞。

        没有退避时：默认 15 分钟一轮 = 一天撞 96 次，
        而失败原因往往是永久的（对方不是有效用户 / 已拉黑）。
        """
        engine = self._idle_engine(hours_ago=10.0)
        self.assertEqual(len(proactive.plan(engine.config, engine.storage)), 1)

        proactive.mark_failed(engine.storage, self.QQ)
        self.assertEqual(proactive.plan(engine.config, engine.storage), [],
                         "失败后应进入退避")

        # 退避过期后应恢复尝试（不是永久放弃）
        proactive.mark_failed(engine.storage, self.QQ,
                              now=time.time() - proactive.FAILURE_BACKOFF_SECONDS - 60)
        self.assertEqual(len(proactive.plan(engine.config, engine.storage)), 1,
                         "退避过期后应该重新尝试")


if __name__ == "__main__":
    unittest.main(verbosity=2)
