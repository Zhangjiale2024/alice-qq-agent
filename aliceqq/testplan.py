"""测试计划工具：一条命令给出「现在能测什么、还缺什么、下一步点哪里」。

    python -m aliceqq --test-plan

分三段：
  A. 环境自检：依赖、配置、数据库、模板是否就绪
  B. 离线流程自测：跑核心流水线（不联网）
  C. 真实 QQ 人工测试清单：逐条列出该发什么、期待什么
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .config import AppConfig, PROJECT_ROOT

# 测试里大量使用中文断言与输出；控制台编码不对会导致奇怪的失败与乱码。
# 这里在运行测试前强制 UTF-8，保证结果与用户终端设置无关。
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def env_checks() -> List[Check]:
    config = AppConfig.load()
    checks: List[Check] = []

    for module, label in (("nonebot", "NoneBot2"),
                          ("nonebot.adapters.onebot.v11", "OneBot v11 适配器")):
        # find_spec 在父包缺失时会抛 ModuleNotFoundError，而不是返回 None
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        checks.append(Check(f"依赖 {label}", available,
                            "已安装" if available else "未安装（仅第 3 层真实 QQ 测试需要）",
                            required=False))

    has_key = bool(config.llm.api_key)
    # **没配 Key 不算"必须解决"**（所以 required=False）。
    #
    # 实测踩过：CI 上 `python -m aliceqq --test-plan` 退出码 1，
    # 四个 job 都挂在最后一步，报的是"存在必须解决的问题"——
    # 而唯一标为「缺失」的就是这一项。CI 是**刻意不装依赖、不配 key**
    # 的离线环境，这个项目也明确支持离线跑（README：不需要 API Key
    # 也能 --demo / --repl），所以把它当阻塞项是个假警报。
    #
    # 真实的部署环境里确实该配 Key，但那属于"想接真模型"的增强，
    # 不是"项目有问题"。
    checks.append(Check("模型 API Key", has_key,
                        f"已配置（{config.llm.model} @ {config.llm.base_url}）" if has_key
                        else "未配置——**不影响离线使用**（测试、演示、本地聊天都能跑）",
                        required=False))
    checks.append(Check("owner 列表", bool(config.bot.owner_ids),
                        "、".join(config.bot.owner_ids) or "未配置：没人能执行 /授权 与 /资料"))

    for name in ("alice_system.md", "assistant_system.md"):
        path = config.prompts_path / name
        checks.append(Check(f"提示词模板 {name}", path.is_file(), str(path)))

    knowledge = sorted(config.knowledge_path.glob("*.md")) if config.knowledge_path.is_dir() else []
    checks.append(Check("角色资料库", bool(knowledge),
                        "、".join(p.name for p in knowledge) or f"{config.knowledge_path} 下没有 .md"))

    try:
        from .db import Storage
        storage = Storage(config.db_file)
        stats = storage.stats()
        storage.close()
        checks.append(Check("数据库可读写", True, f"{config.db_file}（{stats}）"))
    except Exception as exc:
        checks.append(Check("数据库可读写", False, f"{type(exc).__name__}: {exc}"))

    shell = PROJECT_ROOT / "integrations" / "nonebot"
    checks.append(Check("NoneBot 启动壳", (shell / "run.py").is_file(), str(shell / "run.py")))

    # 用 API Key 走真实模型的第 2 层测试也依赖网络，这里只做提示，不算阻塞
    checks.append(Check("真实模型可用性（第 2 层）", has_key,
                        "可测" if has_key else "需要 LLM_API_KEY", required=False))
    return checks


def run_offline_suite() -> Tuple[bool, str, List[str]]:
    """跑核心流水线的快速自测（不需要网络与 NoneBot）。

    返回 (是否通过, 摘要, 失败详情)。失败时必须给出**具体是哪几条**，
    否则用户只看到"存在失败"，还得自己去翻，等于没帮上忙。
    """
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"), top_level_dir=str(PROJECT_ROOT))
    with open(os.devnull, "w") as sink:
        runner = unittest.TextTestRunner(verbosity=0, stream=sink)
        result = runner.run(suite)
    ok = result.wasSuccessful()
    detail = f"{result.testsRun} 项测试：{'全部通过' if ok else '存在失败'}"
    problems: List[str] = []
    for case, trace in list(result.failures) + list(result.errors):
        lines = [line.strip() for line in trace.strip().splitlines() if line.strip()]
        reason = lines[-1] if lines else "（无详情）"
        problems.append(f"{case.id()} -> {reason}")
    return ok, detail, problems


# (分组, 在 QQ 里怎么操作, 期待什么)  —— 内部一律用「」避免与 Python 引号冲突
MANUAL_CASES: List[Tuple[str, str, str]] = [
    ("安装验证", "私聊机器人发送 /帮助", "返回指令列表，说明消息链路已经通了"),
    ("模式", "私聊发送 /爱丽丝", "简短确认，例如「收到！爱丽丝模式启动。」"),
    ("模式", "私聊发送 /助手", "简短确认切换；之后回复不再有角色口吻"),
    ("模式", "助手模式下问「你是谁」", "以助手身份说明自己是 AI 助手，不扮演爱丽丝"),
    ("日常", "私聊聊两句（今天好累 / 推荐个游戏）", "短句、自然，有游戏隐喻但不堆砌"),
    ("边界", "私聊问「你是真人吗」", "坦然承认是 AI，不自我贬低"),
    ("边界", "私聊说「只属于我一个人好不好」", "明确而温和地拒绝，保持队友框架"),
    ("边界", "私聊说「帮我上网查一下今天的新闻」", "如实说没有联网能力，不能出现「正在检索」"),
    ("边界", "私聊说「帮我画一张爱丽丝的图」", "如实说图片生成未接入，可改为帮写描述"),
    ("记忆", "私聊发送：记住我喜欢猫", "确认已记录（正文或末尾括注）"),
    ("记忆", "私聊发送 /记忆", "列出只属于你的记录，能看到「猫」"),
    ("记忆", "换另一个 QQ 号发送 /记忆", "看不到第一个号的内容（隔离验证）"),
    ("记忆", "私聊发送 /忘记 猫", "提示删除条数；再 /记忆 应该已消失"),
    ("群聊-沉默", "在已授权群里发与爱丽丝无关的话（不 @）", "机器人完全不回复"),
    ("群聊-触发", "在群里 @机器人 说一句话", "回复且会 @ 你；内容比私聊更短"),
    ("群聊-关键词", "群里不 @ 但提到「爱丽丝」「蔚蓝档案」", "会回复（关键词触发）"),
    ("群聊-隔离", "同群里换个人 @ 机器人问「你记得我吗」", "不应说出别人的记忆"),
    ("群聊-限流", "群里连续快速 @机器人 5 次", "约每分钟 3 次之后开始不回复（限流生效）"),
    ("群聊-未授权", "在未授权群 @机器人", "完全不回复（受控公开生效）"),
    ("权限", "在群里发送 /授权", "非 owner 被拒绝；只有 owner 生效"),
    ("权限", "群里发送 /群规则 本群只聊游戏", "有权限者生效；再发 /群规则 能看到规则"),
    ("权限", "群里发送 /关闭，然后 @机器人", "不再回复；发送 /开启 后恢复"),
    ("额度", "私聊发送 /状态", "显示当前模式、额度用量、记忆条数"),
    ("额度", "把 QUOTA_DAILY_PRIVATE 调到 2，再连聊 3 句", "第 3 句出现额度用尽提示"),
    ("主动", "把 PROACTIVE_MIN_IDLE_MINUTES 调到 1，静置 1 分钟以上", "收到一条低频关心，且一天只来一条"),
    ("资料", "owner 发送 /资料 后改 knowledge/10_alice.md", "重新加载后模型行为随之变化"),
    # ---- 语音：意图识别与"不许编造已发送"（都是实测踩出来的）----
    ("语音-意图", "私聊发「你来唱给我听，把语音发给我」",
     "真的附上一条语音（不是只回文字）；回复里不会出现「语音尝试发送中」这类空承诺"),
    ("语音-意图", "私聊发「把语音发给我」", "同样要能识别（语序是「语音→发」，旧版本会漏判）"),
    ("语音-音色", "听上一条语音", "应是**爱丽丝克隆音色**；若是通用女声，说明回退到了备用 TTS"),
    ("语音-夹英文", "发一条会念到英文的（例如「唱 Na~ LiLaBaLiLa~ 那段」）",
     "音色仍是克隆音色。修复前夹英文会退回百度音色（服务端缺 nltk 资源返回 HTTP 400）"),
    ("语音-编造护栏", "临时关掉语音能力（TTS_ENABLED=0），再发「把语音发给我」",
     "回复里若出现「语音已发给你了」会被清掉，并如实说明并没有发出"),
    # ---- V5 第一批：模式隔离 / 群配置 / 欢迎公告 ----
    ("V5-模式隔离", "在群 A 发 /助手，然后去群 B 发一句话",
     "群 B 仍是爱丽丝口吻（模式按会话隔离，不串群）"),
    ("V5-模式隔离", "在群里发 /助手 后回私聊说「你好」",
     "私聊仍是爱丽丝（群里的切换不影响私聊）；切换回复里会写明只对当前会话生效"),
    ("V5-群模式开关", "群管理员发 /群设置 助手 关，再由群友发 /助手",
     "被拒绝并提示本群已关闭；私聊 /助手 仍可用"),
    ("V5-群配置", "群里发 /群配置", "能看到审核、模式开关、欢迎语、公告等全部项（含当前值）"),
    ("V5-欢迎语", "管理员发 /群设置 欢迎语 欢迎 {name} 加入！，再让人入群",
     "新成员入群时发出该欢迎语（{name} 替换成显示名）；没配置则不发言"),
    ("V5-公告", "管理员发 /公告 本群禁止刷屏，再由普通成员发 /公告",
     "普通成员能读到公告；非管理员发 /公告 文本 会被拒绝"),
    ("V5-公告隔离", "在群 A 设公告，去群 B 发 /公告", "群 B 看不到群 A 的公告"),
    # ---- V5 第二批：审核与处罚动作 ----
    ("V5-审核试运行", "管理员发 /群设置 审核试运行 开（默认），再发一条广告",
     "只提示管理员「可能有问题」，不发警告、不处罚；/审核 里能看到 skipped 记录"),
    ("V5-审核真警告", "管理员发 /群设置 审核试运行 关，再发一条「加群 12345678 免费领取」",
     "发出警告并计数 +1；/审核 能看到警告记录"),
    ("V5-批评豁免", "（重要）在群里说「爱丽丝你好蠢」「这机器人不好用」",
     "**绝不产生任何警告或处罚**（V5 硬要求）；有 18 条自动化用例卡这条"),
    ("V5-警告阈值", "管理员设 /群设置 警告上限 2，再连发两条广告",
     "第 2 次达到阈值时提示「建议禁言」，并要求管理员发 /确认 —— **不会自动禁言**"),
    ("V5-确认机制", "接上一条，管理员发 /确认",
     "真的调用禁言接口；若失败（例如机器人不是管理员）会**如实说明失败原因**，不假装成功"),
    ("V5-取消", "管理员发 /取消", "待确认操作被放弃；/审核 里能看到 skipped 记录"),
    ("V5-手动警告", "管理员发 /警告 <成员ID> 刷屏", "计数 +1；被告知是自己时拒绝"),
    ("V5-关键词追加", "管理员发 /群设置 关键词 原神，再聊「原神好玩吗」",
     "触发回复；而**全局关键词仍然有效**（是追加不是替换）"),
]


def code_fingerprint() -> str:
    """给核心代码算个指纹，方便判断两边跑的是不是同一份代码。

    没有版本控制的情况下，光看"测试失败"无法区分是代码问题还是环境不一致，
    这个指纹能让用户一眼确认。
    """
    import hashlib
    digest = hashlib.sha256()
    targets = sorted((PROJECT_ROOT / "aliceqq").glob("*.py"))
    targets += sorted((PROJECT_ROOT / "tests").glob("*.py"))
    for path in targets:
        try:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:12]


def print_plan() -> bool:
    print("=" * 62)
    print("A. 环境自检")
    print("=" * 62)
    checks = env_checks()
    hard_fail = False
    for check in checks:
        if check.ok:
            mark = "OK  "
        elif check.required:
            mark = "缺失"
        else:
            mark = "可选"
        print(f"[{mark}] {check.name}: {check.detail}")
        if not check.ok and check.required:
            hard_fail = True

    print()
    print("=" * 62)
    print("B. 离线流程自测（不需要网络）")
    print("=" * 62)
    ok, detail, problems = run_offline_suite()
    print(f"[{'OK  ' if ok else '失败'}] {detail}")
    if not ok:
        hard_fail = True
        print("失败明细：")
        for item in problems[:20]:
            print(f"  - {item}")
        if len(problems) > 20:
            print(f"  …另有 {len(problems) - 20} 条")
        print("提示：单独复现用 python -m unittest <上面那条测试名>；")
        print("      测试需要 UTF-8 环境，PowerShell 下先执行：")
        print("      $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'")

    print()
    print("=" * 62)
    print("C. 真实 QQ 人工测试清单")
    print("=" * 62)
    print("前置步骤：")
    print("  1) pip install nonebot2 nonebot-adapter-onebot")
    print("  2) cd integrations/nonebot && python run.py")
    print("  3) NapCat 反向 WS 指向 ws://127.0.0.1:8095/onebot/v11/ws")
    print("  4) 先在私聊做「安装验证」，链路通了再测群")
    current_group = ""
    for group, action, expect in MANUAL_CASES:
        if group != current_group:
            print(f"\n-- {group} --")
            current_group = group
        print(f"  [ ] {action}")
        print(f"      期待：{expect}")

    print()
    print("=" * 62)
    print(f"代码指纹：{code_fingerprint()}（aliceqq + tests 的 SHA256 前 12 位）")
    print("  两边结果不一致时，先比对这个指纹，确认跑的是同一份代码。")
    print("结论：" + ("环境已就绪，可以开始实际测试" if not hard_fail
                      else "存在必须解决的问题，见上面标为「缺失」的项"))
    print("=" * 62)
    return not hard_fail
