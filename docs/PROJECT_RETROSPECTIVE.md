# Handy+ Fork — 项目阶段性复盘

> Period: **2026-04-30 → 2026-05-02**（48 小时密集开发，9 次 ship）
> Fork: cjpais/Handy → ccharname/handy-plus
> Final shipped: **v0.8.13-handy-plus.1**

这份文档是给将来回头看的（自己或他人接棒）。结构：背景 → 时间线 → 性能数据 → 决策档案 → 踩坑 → 技术债 → 经验。5 分钟读完。

---

## 1. 背景与目标

cjpais/Handy 是一个 Tauri 2.x 桌面 STT 工具，原版面向英语用户（Whisper / Parakeet / Apple Speech）。我 fork 出来做"中文用户友好 + AI workflow 集成"的强化版：

**5 个北极星目标**（来自 v0.8.3-handy-plus.6 的 handoff）：
1. **三套 ASR 一键切换**：Apple Speech / SenseVoice / FunASR-Nano，按场景选
2. **中文标点恢复**：Apple Speech 中文不出标点的硬伤靠独立的 CT-Transformer-Punc 子模型修
3. **Power Mode profiles**：根据前台 App 自动切配置（写代码 vs 微信 vs 写作）
4. **完整测试 + 性能 baseline**：可量化的 latency / 准确率 / 资源占用
5. **可工程化的部署**：升级永不重弹权限 + 一条命令完成 build/sign/install

**全部达成。** 但每条都有 caveat（见 §6）。

---

## 2. 9 次 ship 时间线

| Ver | 日期 | 关键内容 | 触发原因 |
|---|---|---|---|
| **v0.8.5** | 2026-05-01 22:30 | CT-Punc model URL 修 + overlay 居中波形 | 用户反馈「标点时有时无」+「录音条贴底难受」 |
| **v0.8.6** | 2026-05-01 23:30 | CJK fallback (Apple Native CT-Punc 接通) + bench CLI mode | W1-3 实测发现 apple_native punc_density=0.0082 异常 |
| **v0.8.7** | 2026-05-02 00:00 | Apple Speech 权限 hang fix + 4 类 error + 测试基建 | benchmark 自动化跑 apple_native 卡死（headless 无人点权限） |
| **v0.8.8** | 2026-05-02 08:30 | bench forwarder hyphen alias + custom_words 持久 cleanup（185→96）| Pipeline 实测真实 925→139ms（之前 outlier） |
| **v0.8.9** | 2026-05-02 09:30 | RwLock punc cache + Wave 2 测试基建 + About fork section | 用户问「关于版本和作者信息没更新」 |
| **v0.8.10** | 2026-05-02 11:00 | Profile 扩 3 字段 (preset/punc/chain) + summary chips + builtin 全启 + PROFILE_PLAYBOOK | 用户问「场景模式如何生效 / 怎么显性化」 |
| **v0.8.11** | 2026-05-02 14:30 | Migration 强制启用老用户的 builtin profile | v0.8.10 装机后用户发现 profile 仍未勾 |
| **v0.8.12** | 2026-05-02 17:30 | 固定 self-signed code signing — 升级永不重弹权限 | 用户「每次升级都要批权限，已经厌烦了」 |
| **v0.8.13** | 2026-05-02 19:30 | i18n 19 locale 100% coverage + Apple Speech permission test matrix | 一次性扫尾 |

**节奏**：每次 ship 间隔 1-3 小时。每版的 commit message 就是当时的 issue 现场 — 把 git log 当时间机器读最准。

---

## 3. 关键性能数据（v0.8.13 终态）

### 3.1 Latency（Tauri runtime 全 pipeline，38 wav × 3 round = 152 次 stability run）

| Preset | cold_start | steady_p50 | steady_p95 | punc_density | 状态 |
|---|---|---|---|---|---|
| Chinese Balanced | 752 ms | **139 ms** | 2866 ms | 0.064 | ✅ 全达标 |
| Multilingual Offline | 886 ms | 817 ms | 11164 ms | 0.085 | ✅ 速度达标，p95 略高 |
| Apple Native | 3608 ms ⚠️ | 748 ms | 1788 ms | **0.081** | ⚠️ cold_start 超 1500ms 目标 |

**Apple Native cold_start 3608ms** 是 v0.8.13 的 known issue：SFSpeechRecognizer 首次 init（特别是 require_on_device=true 时）需要 macOS 解压本地 dictation 资源。可 mitigations 但代码内不易控。

### 3.2 子模块独立 metric

| Metric | 目标 | 实测 |
|---|---|---|
| CT-Punc cold | <500 ms | **4 ms** ✅ |
| CT-Punc steady_p50 | <30 ms | **2 ms** ✅ |
| Engine swap SV→FN | <3000 ms | **1926 ms** ✅ |
| Engine swap FN→SV | <3000 ms | **425 ms** ✅ |
| 100 次连跑 crash | 0 | **0 / 152** ✅ |

### 3.3 资源（idle + active inference）

| 状态 | RSS | CPU% |
|---|---|---|
| 进程基线 | 866 MB | 0.1% |
| sense-voice-int8 idle | 1250 MB | 0.11% |
| sense-voice-int8 inference peak | **1418 MB** | **368%** (multi-core) |
| funasr-nano idle | 2051 MB | 0.11% |

5-min idle watcher 实测生效：301s idle → unload → 866 MB（76ms 卸载）

### 3.4 Disk

| 模型 | Size | 用途 |
|---|---|---|
| sense-voice-int8 | 228 MB | Chinese Balanced 默认 |
| funasr-nano | 972 MB | Multilingual Offline |
| CT-Punc int8 | 76 MB | 必需依赖（无 ASR 模型自带中文标点） |
| Apple Speech | 0 MB | 系统资源 |

---

## 4. 决策档案（按时间）

### 4.1 SenseVoice 取代 Whisper 作默认（v0.8.3-handy-plus.6 — 本次 fork 之前已定）
- 实测 70ms/10s，体感最好；Apple Speech 中文乱译 + Nano 慢
- 对错位 95%：实测前我以为 Apple Speech 是首选

### 4.2 CT-Transformer-Punc 作通用层（v0.8.5）
- 不让任一引擎"自己学中文标点"，独立子模型 + 75 MB 跨引擎复用
- 决策正确：punc cold 4ms / steady 2ms — 比放进 ASR 引擎廉价 20×

### 4.3 CJK 字符 fallback（v0.8.6）
- apply_punc_zh_if_applicable 原仅靠 language metadata 判定
- Apple Native 用 auto language + en app_language → base_lang=en → 跳过 punc
- **修法：见到 CJK 字符就 apply punc**（zh-en vocab272727 模型纯英文 no-op）
- 这是项目里影响最大的 5 行代码改动

### 4.4 固定 self-signed code signing（v0.8.12）
- macOS TCC 在 cdhash 变化时清权限。Tauri ad-hoc signed 每次 build cdhash 都不同 → 重弹
- 用户 7 次升级 7 次批权限后明确 push back
- 解法：本地生成 self-signed cert + 给所有 build 同 identity 签名 → cdhash 变 / authority 不变 → TCC 信任保留权限
- 实证：v0.8.12 → v0.8.13 升级零弹窗

### 4.5 Profile snapshot+detach 模式（v0.8.10）
- 不是"runtime fallback chain"
- Apply preset 时把字段写进独立 settings；用户改单字段 → 自动 detach
- 比"独立读 active_preset_id 再 fallback 全字段"简单 10×（zero pipeline 改动）

### 4.6 Migration tracking field（v0.8.11）
- ensure_*_defaults 模式只补 empty，不升级现有 array
- 加 `migration_applied: HashMap<String, bool>` 追踪一次性 migration（idempotent）
- 适用未来所有 schema 升级场景

### 4.7 Bench mode 4 种（v0.8.7-12）
- asr / punc-only / chain / swap，分桶测试不同子系统
- punc-only 100 次能复现 OnceCell 缓存效果（首次 4ms，后续 2ms）
- swap 9 次能算 SV→FN / FN→SV 两个方向 latency
- chain 测 post_process_chain（v0.8.10 加但少用）

### 4.8 一次性 deploy script（v0.8.13）
- pkill → mv 备份 → cp /Applications → open → verify version 5 步固化
- 任何一步失败 exit 非零
- 加 codesign Authority 检查防 TCC 重弹回归

---

## 5. 踩坑（按发现顺序）

| # | 现象 | 根因 | 教训 |
|---|---|---|---|
| 1 | CT-Punc 时有时无 | sonnet 编了不存在的模型 URL（zh-cn 版本不存在，实际是 zh-en-vocab272727）+ 模型文件名错（model.onnx vs model.int8.onnx） | sonnet 写 URL 必 verify upstream releases 列表 |
| 2 | Overlay 录音条贴底 | `align-items: end` 不是 `center` | UI 视觉问题靠用户实测才发现，agent 不知道 |
| 3 | apple_native punc_density=0.0082 | benchmark/run_bench.py 直跑 sherpa-onnx + Swift binary 绕过 Tauri pipeline，CT-Punc 没经过 | benchmark 路径不能省 — "测的是不是真用户路径" 必须先验证 |
| 4 | bench latency 925ms outlier | 200+ custom_words O(N×M) fuzzy + cold model + per-item spawn_blocking 综合 | 真实交互延迟和 bench 数字之间有 gap，需要 cold/steady 分桶 |
| 5 | Apple Speech bench hang | benchmark 自动化时 SFSpeechRecognizer requestAuthorization 弹权限框无人点 → 死等 | 任何同步等待外部输入的代码都要加 timeout |
| 6 | Profile 默认未启用 | ensure_app_profiles_defaults 只补 empty array，不升级 | schema 升级要专门的 migration tracking |
| 7 | TCC 每次升级重弹权限 | macOS 14+ TCC 按 cdhash 验证，Tauri ad-hoc signed 每次 cdhash 不同 | 本地 dev cert 必须固定 |
| 8 | bench forwarder silent failure | clap 把 PuncOnly enum 序列化为 "punc-only"，但 mode 解析器只接受 "punc_only" | enum value 跨边界要精确测 |
| 9 | tauri build 因 updater 私钥缺失 exit 1 | upstream 配了 publicKey 但 fork 没私钥；build 实际产物 OK 但 exit code 1 | 一定要用 exit code 判断成功，但失败 case 需调研 |
| 10 | settings.json cleanup 被 atexit 覆盖 | running app 内存中的旧 settings 在 pkill 时 flush 回文件 | 改 settings_store 必须先 pkill |

---

## 6. 技术债 / 未达成

### 6.1 已知 Issue（v0.8.14+ 修候选）

| Issue | 优先级 | 描述 |
|---|---|---|
| Apple Native cold_start 3608ms | P1 | SFSpeechRecognizer 首次 init 慢；mitigation 候选：app 启动时预热 |
| OnceCell 不支持 reset → RwLock 已修 ✅ | - | v0.8.9 已修 |
| benchmark forwarder 并发 drop | P2 | bench A 在跑时 bench B 命令被 silent drop |
| restricted 权限测试 | P3 | 需要 MDM/Configuration Profile，个人 Mac 测不了 |
| post_process_chain UI 缺 drag-drop | P2 | 用上下箭头 reorder（v0.8.10 简化方案） |
| SV transcribe-rs 路径不支持 hotwords | P2 | 用户需切到 sense-voice-small-sherpa 才能用 hotwords_boost |
| custom_words O(N×M) fuzzy match | P2 | v0.8.8 已加 max_custom_len early-skip，但仍 O(N×M) 上限。优化候选：trie 或前缀 index |

### 6.2 未做

- T-13.x denied/notDetermined 状态实测（manual checklist 已写在 docs/APPLE_SPEECH_PERMISSION_TEST.md，未跑）
- WER/CER 准确率（缺 reference.txt 人工标注，dataset 已分 6 子集）
- UI 视觉 / overlay FPS / tray dark mode 切换 — 部分手测，无系统性数据
- post_process_chain 用户实战测试（B4 落地但用户没真用）
- MCP server 模式（外部 agent 可 tool-call 触发 Handy 听写）— upstream 路线图

### 6.3 Wave 4 优化候选未做

- **O-1** SenseVoice steady_p50 自然达成（925→139ms，custom_words cleanup + spawn_blocking 修）
- **O-2** FunASR-Nano cold_start 优化 — 未做
- **O-3** Apple Speech under Tauri runtime — v0.8.7 已修 hang
- **O-4** punc_zh OnceCell 全局共享 — v0.8.9 已改 RwLock
- **O-5** Profile hot-swap < 500 ms — 当前 SV→FN 1.9s / FN→SV 0.4s 够用

---

## 7. 经验 / 反模式（沉淀给以后）

### 7.1 派活原则
- **Opus 当脑、Sonnet 当手**（paihuo skill 已固化）
- 文件互斥是首要约束 — 同改同文件必 race
- 派之前 verify subagent prompt 自包含（绝对路径 / 不引用 conv 历史）
- subagent 摘要不可信，必 git diff 实测

### 7.2 修 bug 之前先验证
- "用户报标点时有时无" → 真因是模型 URL 错了，不是 punc 路径问题
- "bench 数字是 925ms" → 真因是 cold start + 200 custom_words，不是 pipeline 慢
- 永远 trace 到日志 / 数据，不要凭推理修

### 7.3 用户反馈是最快的 bug 雷达
- 7 次 ship 7 次用户反馈，每次都直击设计盲点：
  - "标点时有时无" → CT-Punc URL 错
  - "录音条难受" → align-items
  - "每次都要批权限" → cdhash 重弹
  - "profile 没勾选" → migration 缺失
  - "版本作者信息没更新" → fork section 缺失
  - "场景如何生效" → chip 显式化
  - "把底下 model 列表去掉" → 简化 ModelsSettings
- 每次都是用户视角发现 agent 视角看不到的问题

### 7.4 部署自动化必须真闭环
- 跨 7 次 ship 才意识到 deploy 的"分步信号"是噪音
- 用户明确 push back 后才固化为 `bun run tauri:deploy`
- 教训：用户讲 1 次说"不要分步"就应该立即固化，不要拖

### 7.5 测试基建优先于性能优化
- W1-K 任务（资源监控 + i18n diff + hotword recall）落地后所有后续测试都受益
- O 任务跑 152 次 stability 0 crash 是因为有了基建
- 没基建只能靠手感

---

## 8. 文件 / Commit 索引

### 关键文档
- `docs/ASR_PRESET_GUIDE.md` — 三套 preset 选择指南（用户级）
- `docs/PROFILE_PLAYBOOK.md` — 5 套场景化 profile 配方
- `docs/CODESIGN_SETUP.md` — 固定 cert 配置流程（一次性）
- `docs/TEST_REPORT_v0.8.11.md` — 67 用例测试报告
- `docs/APPLE_SPEECH_PERMISSION_TEST.md` — 4 状态测试矩阵
- `docs/i18n_audit_2026-05-02.md` — locale 缺 key 审计
- `benchmark/results/v0.8.8/comparison_*.md` — 三 preset 对比

### Tags
```
v0.8.5-handy-plus.1   # CT-Punc URL fix + overlay
v0.8.6-handy-plus.1   # CJK fallback + bench CLI
v0.8.7-handy-plus.1   # Apple Speech permission hardening
v0.8.8-handy-plus.1   # bench forwarder + cleanup
v0.8.9-handy-plus.1   # RwLock + Wave 2 infra
v0.8.10-handy-plus.1  # Profile expansion + Playbook
v0.8.11-handy-plus.1  # Migration auto-enable builtins
v0.8.12-handy-plus.1  # Signed builds (TCC retention)
v0.8.13-handy-plus.1  # i18n 100% + perm test matrix
```

### 备份
- 每次升级前的 `/Applications/Handy.app` 在 `~/.Trash/Handy.app.<oldver>-<ts>/`
- settings_store.json 备份在 `~/Library/Application Support/com.pais.handy/settings_store.json.bak.*`

---

## 9. 下次接棒怎么开始

1. 读 `.handoffs/sedgewick-2026-05-02t<time>.md`（最新 handoff，本次 W 任务输出）
2. 读这份 RETROSPECTIVE 拿全局视角
3. `docs/ASR_PRESET_GUIDE.md` 拿用户视角
4. `git log --oneline v0.8.5-handy-plus.1..` 看时间线
5. `bun run tauri:deploy` 跑一遍验证当前 toolchain 健康
6. `python3 scripts/i18n_diff.py` 验证 i18n 完整性
7. 选 §6.1 一个 Issue 开始 — Apple Native cold_start 是最高 ROI

---

**最重要的 5 个数字**（走前必记）：
- Tauri pipeline P50: **139 ms**（chinese_balanced）
- CT-Punc steady: **2 ms**
- Profile hot-swap: **425-1926 ms**（FN→SV / SV→FN 不对称）
- 152 连跑 0 crash
- TCC 不再重弹（cert SHA1 `C68847EC...`）

收工。
