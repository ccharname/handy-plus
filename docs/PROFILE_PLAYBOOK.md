# Handy+ Profile Playbook

> 5 套场景化 Power Mode profile 最佳实践，基于 v0.8.9 实测数据。

Power Mode 让 Handy+ 根据当前前台 App 自动切换 ASR 配置。这份文档给你**直接可抄的配置方案** + 使用决策树。

---

## 快速决策树

```
你最想为这个场景优化什么？
├─ 速度（命令式短句，停顿即想立即出文字）
│   └─ Apple Native + Direct paste + auto_submit
├─ 标点 / 长句质量（写作、笔记、Slack 长消息）
│   └─ Chinese Balanced + 后处理链
├─ 多语种混读（技术对话、跨国协作）
│   └─ Multilingual Offline + 翻译链
├─ 特定术语高频（产品名 / 同事缩写 / 学术词汇）
│   └─ 任一 preset + custom_words_extra 局部加权
└─ 完全离线（保密 / 网络受限）
    └─ Multilingual Offline 或 Chinese Balanced（都本地）
```

---

## 5 套场景配方

### 1. Code / Terminal（写代码、Shell 命令）

| 字段 | 推荐值 | 理由 |
|---|---|---|
| ASR Preset override | **Apple Native** | 命令式短句首字延迟最低（748 ms p50） |
| Language override | `auto` | Apple Speech 用系统 locale，跟随你写注释的语言 |
| punc_zh_enabled | **Off**（M-3 落地后）| 代码不需要中文标点，避免 `git commit -m "feat：xxx"` 这种乱入 |
| paste_method | **Direct** | 直接打字到 cursor，最自然 |
| append_trailing_space | **Off** | 代码后不应带空格 |
| auto_submit | **Off** | 命令需要审完再回车 |
| post_process_chain | None / 空 | 保留 raw text，不要 LLM 改 |
| custom_words_extra | `Anthropic, Sonnet, codex, tmux, sherpa-onnx` 等技术术语 | 减少术语识别错误 |

**Matchers**: `com.microsoft.VSCode`, `com.todesktop.230313mzl4w4u92` (Cursor), `com.googlecode.iterm2`, `com.apple.Terminal`, `com.googlecode.iterm2`, process_name `Code` / `Cursor` / `Ghostty`

**Hot-swap engine**: 启用（`profile_hot_swap_engine = true`），让其它 profile 用 SenseVoice 时这里能自动切到 Apple Native。延迟代价 1-3 s 但写代码不连续录音感知不到。

---

### 2. Chat / IM（微信、Slack、iMessage、Discord）

| 字段 | 推荐值 | 理由 |
|---|---|---|
| ASR Preset override | **Chinese Balanced**（中文为主） / **Multilingual Offline**（混语）| 速度 + 标点都过得去 |
| Language override | `zh-Hans` 或 `auto` | 看你聊天对象 |
| punc_zh_enabled | **On** | 聊天必带句逗 |
| paste_method | **Clipboard (Cmd+V)** | 微信桌面版 Direct 方法对长文本不稳 |
| append_trailing_space | **On** | "吃饭了吗 " — 后接表情 / 接问询不卡 |
| auto_submit | **On** | 说完即发，不用手按 Enter |
| post_process_chain | `[润色 prompt]` | 聊天口语化转书面化（可选） |
| custom_words_extra | 同事 / 朋友昵称 / 业务高频词 | 名字识别准 |

**Matchers**: `com.tencent.xinWeChat` (微信), `com.apple.MobileSMS` (iMessage), `com.tinyspeck.slackmacgap` (Slack), `com.hnc.Discord`, `com.tencent.qq`

---

### 3. Writing / Notes（Obsidian、写作、长文档）

| 字段 | 推荐值 | 理由 |
|---|---|---|
| ASR Preset override | **Multilingual Offline** | FunASR-Nano 标点密度 0.085 最高（写作要标点完整）|
| Language override | `auto` | 写作可能突然引用英文术语 |
| punc_zh_enabled | **On** | 强制中文标点完整 |
| paste_method | **Direct** | Obsidian / Bear / Notion 都对 Direct 友好 |
| append_trailing_space | **On** | 段落间清晰 |
| auto_submit | **Off** | 写作不要自动回车 |
| post_process_chain | `[润色] → [格式化]` | 长文先润色再加 markdown 结构 |
| custom_words_extra | 项目专有名词 / 术语词典 | 学术 / 行业用词准确 |

**Matchers**: `md.obsidian`, `net.shinyfrog.bear`, `com.notion.notion`, `com.literatureandlatte.scrivener3`, `com.toketaware.osx-typora`

**注意**：长录音 (>30 s) 走 Multilingual Offline 的 cold_start 可达 8 s，**首次按键预热可放任 idle 时**让 idle watcher 自然 unload + reload；连续录音第 2 段起 steady ~800 ms。

---

### 4. Translation（跨语种翻译工作台）

| 字段 | 推荐值 | 理由 |
|---|---|---|
| ASR Preset override | **Multilingual Offline** | LLM-decoder 多语种最强 |
| Language override | `auto` | 自动检测源语言 |
| punc_zh_enabled | **On** | 中→外/外→中都可能含中文 |
| paste_method | **Clipboard** | 翻译工具一般有 input 框，Direct 不一定生效 |
| append_trailing_space | **Off** | |
| auto_submit | **Off** | |
| post_process_chain | `[翻译到 X 语言]` | 一段录音 → 自动翻译输出 |
| custom_words_extra | 双语对照术语 | 跨语翻译准 |

**Matchers**: `com.linear`, `com.deepl.app`, `com.kagi.kagimacOS`, browser tabs（process_name `Chrome`/`Safari`/`Arc` + window_title 含 "translate" / "DeepL"）

---

### 5. Code Review / PR 评审

| 字段 | 推荐值 | 理由 |
|---|---|---|
| ASR Preset override | **Apple Native** | 短评论快速出 |
| Language override | `auto` | 中英都可能 |
| punc_zh_enabled | **On** | 评论要标点 |
| paste_method | **Direct** | GitHub / GitLab Web 输入框 |
| append_trailing_space | **Off** | |
| auto_submit | **Off** | review 评论不要自动 submit |
| post_process_chain | None | 保留原意 |
| custom_words_extra | reviewer 名字 / 同事 ID / 工程术语（refactor / hotpath / regression）| 名字识别 |

**Matchers**: process_name `Chrome`/`Safari`/`Arc` + window_title_substring `pull/` / `review` / `merge_requests`

---

## Profile 设计原则

### 何时该 override 字段
- **覆盖**：场景明显不同于全局设置（如代码场景关中文标点，聊天场景开 auto_submit）
- **不覆盖**（保持 inherit）：场景跟全局一样，留 None 让全局升级时自动跟随

### 启用 hot_swap_engine 的判定
- ✅ 启用：你确实在不同场景需要不同 ASR 模型（Apple Speech 命令场景 + SenseVoice 中文写作场景）
- ❌ 不启用：你只想 override language / paste / chain，不想换引擎（避免 1-3 s 切换延迟）

### Custom_words_extra vs Global custom_words
- **全局 custom_words**：跨场景的高频术语（自己常用的产品名、同事昵称、技术词汇）
- **Profile 的 custom_words_extra**：场景独有术语，会跟全局 merge dedup（不替换）

### 不要做的事
- ❌ Profile A 配置和 Profile B 完全相同 — 直接合并 matchers
- ❌ Override 一堆字段后又跟全局相同 — 留 inherit 节省维护
- ❌ 在 builtin_default 上覆盖大量字段 — builtin_default 是兜底，应该留空（不会命中）

---

## 验证 Profile 是否生效

### 看 Tray 状态（推荐）
v0.8.7+ 录音时 tray icon 切换 idle/recording/transcribing。Power Mode 命中时未来会显示 profile 名（M-2 落地后 summary chip 会出现在 ProfilesPage 卡片上）。

### 看 log
```bash
tail -f ~/Library/Logs/com.pais.handy/Handy.log | grep "Power Mode resolved"
```
每次录音完会输出：
```
Power Mode resolved: profile=Some("Code") lang=auto paste=Some(Direct) model=apple-speech
```

### Pipeline timing log（v0.8.8+）
```bash
RUST_LOG=debug open /Applications/Handy.app
# 录音后看
grep "Pipeline timing" ~/Library/Logs/com.pais.handy/Handy.log
```
能看到 engine / custom_words / filter / punc 各段 ms 占比。

---

## 实测延迟参考（v0.8.8 Tauri pipeline）

| Preset | cold_start | steady_p50 | steady_p95 | punc_density |
|---|---|---|---|---|
| Chinese Balanced (SenseVoice-int8) | 752 ms | **139 ms** | 2866 ms | 0.064 |
| Multilingual Offline (FunASR-Nano) | 886 ms | **817 ms** | 11164 ms | 0.085 |
| Apple Native (Apple Speech) | 3608 ms ⚠️ | **748 ms** | 1788 ms | 0.081 |

数据出处：`benchmark/results/v0.8.8/comparison_2026-05-02T*.md`

---

## 已知不适用

- **Linux**：apple_native preset 不可用，profile 配 Apple Native 不生效
- **网络受限**：Apple Speech `require_on_device=true` 失败时仍尝试网络版（需关 `apple_speech_require_on_device`）
- **录音 < 1 s**：Apple Speech 偶尔 partial 不 fire，profile 切到 Chinese Balanced 更稳
- **录音 > 60 s**：FunASR-Nano P95 11 s 延迟显著，长会议录音建议用 history retranscribe 异步重跑

---

## 想加新场景？

1. ProfilesPage 点 "+ Add"
2. matchers 填 bundle_id（macOS 用 `mdls -name kMDItemCFBundleIdentifier /Applications/<App>.app` 查）
3. override 区填想覆盖的字段（其余留 inherit）
4. 启用 toggle
5. 启动该 App，录音验证 — log 看 `Power Mode resolved: profile=Some("<name>")`

如果 profile 一直不命中，最可能原因：
- bundle_id 拼错（多 ".app" 后缀 / 大小写）
- 该 App 没真正前台（被其他窗口压着）
- Profile 顺序：把更具体的 matcher 排在更通用的前面（first-match-enabled）

---

## 相关文档

- `docs/ASR_PRESET_GUIDE.md` — 三套 ASR Preset 选择指南
- `docs/TEST_REPORT_v0.8.8.md` — 性能 baseline 测试报告
- `src-tauri/src/profile_resolver.rs` — Profile 解析逻辑代码
- `src-tauri/src/settings.rs` — `AppProfile` struct + `default_app_profiles()`
