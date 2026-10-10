# -*- coding: utf-8 -*-
import time
from tests.test_engine import make_engine, private
from aliceqq.engine import IncomingMessage, MODE_ALICE, MODE_ASSISTANT

eng = make_engine()
for label, text, t in [("/助手", "/助手", 0), ("问答", "你是谁", 200), ("/爱丽丝", "/爱丽丝", 400)]:
    out = eng.handle(IncomingMessage(user_id="10001", text=text, is_private=True,
                                     user_name="阿岚", timestamp=time.time()+t))
    print(f"{label!r}: mode={out.mode} decision={out.decision} reply={out.reply!r}")
print("profile.mode =", eng.storage.get_profile("10001").mode)

# 2) "第几次互动" 的 off-by-one
eng2 = make_engine()
out2 = eng2.handle(IncomingMessage(user_id="10002", text="你好", is_private=True,
                                   user_name="阿岚", timestamp=time.time()))
print("\n首次私聊的 conversation_summary:")
import re
print([l for l in out2.prompt_system.split("\n") if "第" in l and "次" in l])
print("profile.message_count =", eng2.storage.get_profile("10002").message_count)
