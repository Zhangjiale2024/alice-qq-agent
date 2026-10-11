# Alice QQ Agent

一个运行在 QQ 上的 LLM Agent。

它可以根据消息决定是否调用工具，查询最新信息，保存用户主动要求记住的内容，也会在操作失败时直接说明失败原因。

目前有 715 项离线测试。核心逻辑约 14,400 行，不依赖第三方 Python 包。

<p>
<img alt="tests" src="https://img.shields.io/badge/tests-715%20passing-brightgreen">
<img alt="deps" src="https://img.shields.io/badge/core%20deps-none-blue">
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 现在能做什么

最初只是想做一个能在 QQ 上聊天的机器人。后来发现，聊天里有几件事不能只靠模型自己说：

- 用户说“记住这个”，确实要写进数据库；
- 用户问最新活动，确实要查资料；
- 群里出现广告，至少要先判断是否符合规则；
- 语音、撤回、禁言等操作，做没做必须和实际结果一致。

所以项目里把这些能力拆成了工具，由模型根据上下文决定是否调用。

| 工具 | 用途 |
|---|---|
| `remember` | 保存用户明确要求记住的信息 |
| `web_search` | 查询活动、价格、规则等时效性信息 |
| `academic_search` | 查询论文，使用 CrossRef、Semantic Scholar 和 arXiv |

模型调用工具后，执行结果会以 `role=tool` 的形式放回对话，再让模型根据真实结果生成回复。

例如，用户说：

```text
帮我记住，我不吃香菜
```

模型决定调用 `remember`，工具执行成功后，才会回复：

```text
已记录。以后点菜时我会帮你避开香菜。
```

这句话不是只由模型口头承诺，背后确实完成了一次写库。

使用 `--trace` 可以查看这条链路：

```text
决策      replied
模型      openai / deepseek-chat
工具目录  remember✓ web_search✓ academic_search✓
①         模型决定调用 remember
②         执行结果：成功，已记录
③         将工具结果回灌给模型
记忆落库  2 条
```

## 检索方式

项目保留两种检索路径。

第一种是模型自己调用 `web_search`。第二种是引擎先判断消息是否涉及最新信息，如果涉及，就先查资料，再把结果交给模型。

保留第二种路径是因为实际测试中，角色扮演模式下的模型有时更愿意顺着角色说话，而不是主动调用工具。用户问“今天有什么活动”时，不能因为模型没有调工具就直接凭记忆回答。

示例：

```text
检索路径  引擎判断为先检索
检索      provider=bilibili ok=True 条数=6 耗时=5.1
```

## 几个硬性限制

模型最容易造成的问题，不一定是回答内容错误，而是把没有发生的操作说成已经发生。

目前有几类情况由代码直接处理：

- 回复中出现“语音已经发给你了”，但实际没有附件时，删除这类表述并改为说明失败；
- 用户要求保存身份证号等敏感信息时，由 `looks_sensitive()` 拦截写入；
- 撤回、禁言等管理动作失败时，把实际错误返回给管理员，不让模型自行补一句“操作成功”。

这些规则不依赖模型自觉，换模型也有效。

## 测试

```bash
python -m unittest discover -s tests -t .
python -m aliceqq --test-plan
python -m aliceqq --simulate
```

三类测试覆盖范围不同：

| 测试层 | 需要 | 可以确认 | 不能确认 |
|---|---|---|---|
| 离线测试 | 无 | 模式隔离、限流、护栏、工具分发等逻辑 | 回复是否自然、QQ 是否收到消息 |
| 真实服务 | 网络和 API Key | 模型是否调用工具、检索是否返回 | QQ 是否实际收到 |
| 真实 QQ | NoneBot + NapCat | 消息送达、权限和管理动作 | 未覆盖的异常场景 |

目前第三层还没有充分验证。

CI 在 Ubuntu 和 Windows 上运行，覆盖两个 Python 版本，并且故意不安装核心依赖。这样可以确认“核心逻辑零第三方依赖”不是文档上的描述，而是每次提交都会检查的状态。

## 运行

初始化环境：

```bash
python -m aliceqq --setup
```

不需要 QQ 或 API Key，也可以先本地运行：

```bash
python -m aliceqq --demo
python -m aliceqq --repl
python -m aliceqq --show-prompt "你好"
```

接入真实模型前，在 `.env` 中填写 `LLM_API_KEY`：

```bash
python -m aliceqq "帮我记住我喜欢明日方舟" --live --trace
```

接入 QQ：

```bash
python -m aliceqq --serve
```

该模式需要 NoneBot2 和 NapCat。

## 代码结构

```text
aliceqq/
  engine.py
  tools.py
  llm.py
  memory.py
  search.py
  academic.py
  moderation.py
  media.py
  singing.py
  adapters/
    nonebot_plugin_alice.py

integrations/
  nonebot/
    run.py
```

`aliceqq/` 目录不直接导入第三方库。

- `engine.py`：组装上下文、请求模型、执行工具、运行护栏；
- `tools.py`：工具注册和 OpenAI 格式 schema；
- `llm.py`：使用 `urllib` 实现的 OpenAI 兼容客户端；
- `adapters/`：把 QQ/NoneBot 事件转换成内部格式；
- `integrations/`：负责启动真实 QQ 接入。

第三方能力采用延迟导入。例如没有安装 `faster-whisper` 时，语音识别会返回不可用，其他功能仍然可以运行。

这样做主要是为了：

1. 核心测试可以完全离线运行；
2. 更换消息协议时只需要修改适配层；
3. 只安装 Python 也能运行本地演示。

## 踩过的坑

详细记录在 [docs/踩坑记录.md](docs/踩坑记录.md)，这里只列几个实际遇到的问题。

| 现象 | 原因 |
|---|---|
| 工具始终不调用 | 请求模型时漏传了 `tools` 参数 |
| 语音发送失败 | 文本中混有英文，服务端缺少 `nltk` 资源并返回 HTTP 400 |
| 长音频转换崩溃 `0xC0000005` | Windows 页面文件被其他程序占满 |
| 高频指标下降 6% | 直接比较 44.1kHz 源文件和 40kHz 成品，结果没有意义 |
| 搜索《群青》得到《群青信号》 | 只做标题包含判断，没有区分完整歌名 |
| 30 秒请求生成 195 秒音频 | 中间文件重名，UVR5 跳过了重新处理 |
| “零依赖”测试在干净环境失败 | `test_ffmpeg` 顶层导入了 `numpy` |

这里有几个经验比较值得记下来：

“没有报错”不代表操作成功。Windows 的 `0xC0000005` 这类退出码，需要看进程级结果，不能只看 Python 异常。

各模块单独正确，也不代表整条链路接通。工具注册表、schema、执行器和回灌逻辑都没问题，但少传一个参数，工具就会完全失效。

本机环境过于完整也会掩盖问题。现在交付前会在干净目录重新运行一次 `--setup`。

## 维护脚本

```bash
python scripts/项目自检.py
```

会检查：

- 文档中的测试数量是否和实现一致；
- 是否混入不该提交的文件；
- `.env` 是否存在重复键、空值或 BOM；
- 是否有静默吞异常的代码；
- 是否残留个人信息；
- QQ、NapCat、NoneBot 链路是否可用；
- 是否存在硬编码路径。

相关脚本：

| 脚本 | 用途 |
|---|---|
| `项目自检.py` | 运行全部检查 |
| `隐私扫描.py` | 检查配置值是否残留在代码或文档中 |
| `env审计.py` | 检查 `.env` |
| `代码巡检.py` | 检查异常处理和过长函数 |
| `硬编码路径检查.py` | 检查机器相关路径 |
| `链路体检.py` | 检查 QQ、NapCat、NoneBot 链路 |
| `安装钩子.py` | 安装 pre-commit |
| `重置配置.py` | 备份并重建 `.env` |
| `推送GitHub.py` | 推送前检查代理和配置 |

密钥保护分三层：

- `.gitignore` 排除配置文件；
- 隐私扫描检查残留；
- pre-commit 阻止包含密钥的提交。

密钥一旦进过 Git 历史，之后只删除文件是不够的。

## 常见问题

### 在 cmd 中使用 `rm .env`

`rm` 是 Linux 和 macOS 的命令。Windows 下使用：

```bash
python scripts\重置配置.py
```

### 修改配置后没有生效

先检查是否有重复键：

```bash
python scripts\env审计.py .env
```

后出现的配置可能会覆盖前面的同名配置。

### `--serve` 提示缺少 NoneBot2

只有接入真实 QQ 时才需要它。运行 `--setup` 会自动安装。

如果只是本地测试，可以直接运行：

```bash
python -m aliceqq --repl
```

### 机器人不回复

先运行：

```bash
python scripts\链路体检.py
```

它可以帮助区分：

- 反向 WebSocket 没有连接；
- QQ 登录已经被踢下线；
- NoneBot 插件没有正常加载。

## 目前还没有解决的问题

1. QQ 适配层还没有在真实设备上充分验证。媒体发送和群管理动作需要真实 QQ 环境，目前没有完整的自动化测试。
2. 暂时没有完整的多轮工具编排。工具结果回灌后最多再请求一次模型，还不是完整的 ReAct 循环。
3. 反广告目前使用规则匹配，不是模型判断。优点是可解释，缺点是容易漏掉变形广告。
4. 唱歌链路只有意图识别和缓存测试，完整流程还没有自动化覆盖。
5. 当前是单进程部署，数据使用 SQLite。多实例部署需要改用共享存储。

## 素材和版权

仓库只包含代码。

角色图片、语音模型和生成的音频不放在仓库中，需要自行准备。当前主要用于本机测试。

如果用于公开服务或商业用途，请自行确认角色素材、声音、图片和音频的授权情况。具体说明见 [LICENSE](LICENSE) 末尾。

## 许可证

MIT。

许可证只覆盖仓库中的代码，不自动覆盖第三方素材、模型、声音或生成内容。
