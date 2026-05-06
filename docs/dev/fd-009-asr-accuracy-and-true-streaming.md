---
fd: FD-009
title: ASR accuracy uplift + true streaming architecture
status: draft
created: 2026-05-06
deadline: review-by 2026-05-07 07:00
project: v2t (Handy fork)
authors: claude (research), zheng (decision)
supersedes: FD-006 (v2t-true-streaming chunked, abandoned 2026-05-06)
note: renamed from FD-007 → FD-009 (Lucky global FD-007/008 already taken)
---

# FD-009 — ASR 准确率提升 + 真流式架构

## 问题陈述

经 FD-006 chunked streaming 7 个 follow-up + 最终 D 路径回退 batch-only 后，v2t 当前状态：
1. **stable but slow** — 750-1500ms first-word latency（mlx batch inference + 模型整体出文字）
2. **准确率瓶颈** — 中文 + 英文术语 + 专名混杂场景错字明显，custom_words 50+ entry 效果有限
3. **触发可靠性** — Right Cmd push-to-talk 仍依赖 macOS CGEventTap，间歇性丢 keyup 导致"卡录音"

下一版必须同时回答：**(a) 怎么提升准确率？(b) 怎么在不重蹈 FD-006 复杂度坑的前提下做真流式？(c) 怎么从根因上消除 PTT keyup 丢失？**

## 决策原则

- **业界已有的不重造**：腾讯/讯飞/阿里 ASR 协议、Hammerspoon CGEventTap re-enable 模板、superwhisper NSTextInputClient marked-text 注入 — 都是验证多年的方案
- **真流式必须有真模型架构支撑** — Paraformer-zh-streaming 的 SCAMA chunk-attention 是中文唯一可商用 on-device 方案；Qwen3-ASR token-streaming 不算
- **小步可逆** — 每个 M 独立可上线，失败可单独回滚，不再做 FD-006 那种 6-milestone 串行强耦合

---

## Milestones（按 ROI 优先级排序）

每个 M 标注：工程量 / 期望收益 / 验证标准 / 落地引用。

### M0 — Baseline audit（取证）

**做什么**：跑一组固定 audio 样本（10 段，覆盖纯中文 / 中英混 / 含专名 / 短指令 / 长篇），记录当前 batch 模式的：(a) p50/p99 first-word latency，(b) 错字率（人工标注或与 GPT-4o transcribe 对比），(c) 各场景 RTF。固化到 `docs/dev/fd-007/baseline.md`。

**工程量**：S（半天）。**收益**：之后所有 M 才有可比基线。
**验证**：baseline.md 落盘，含 10 段样本 metrics。

---

### M1 — 修 handy_keys CGEventTap 两个 bug（短跑，根因）

**做什么**：fork `cjpais/handy-keys`，参照 [Issue #840](https://github.com/cjpais/Handy/issues/840) + Hammerspoon [libeventtap.m](https://github.com/Hammerspoon/hammerspoon/blob/master/extensions/eventtap/libeventtap.m)：

1. **Bug A 修复**：callback 内每帧从 `CGEventFlags` reconcile `current_modifiers`，不再依赖 toggle bits 跨事件累积（v0.1.4 行为）。
2. **Bug B 修复**：callback 内显式处理 `kCGEventTapDisabledByTimeout` / `kCGEventTapDisabledByUserInput` → 立即 `CGEventTapEnable(tap, true)`。

切到自家 fork。

**工程量**：S（半天）。**收益**：彻底消除 30-49 秒 keyup 卡死、sentinel false-negative。
**验证**：连续 push-to-talk 30 次，0 次 release event 延迟 > 200ms。

引用：[Hammerspoon libeventtap.m](https://github.com/Hammerspoon/hammerspoon/blob/master/extensions/eventtap/libeventtap.m)，[handy Issue #840](https://github.com/cjpais/Handy/issues/840)。

---

### M2 — Hybrid tap/hold 触发模式（短跑，UX 双保险）

**做什么**：在 transcription_coordinator 加一个 200ms timer：

- press → start timer + 开始 recording
- 200ms 内 release → 进入 toggle 模式（继续录到下次 press / silence-watchdog）
- 200ms 后仍 hold → PTT 模式（release 即停）

业界 default（VoiceInk 1.72 / Superwhisper / Aqua / TypeWhisper）。即使 keyup 偶尔丢，timer 已 toggle 状态对，**双保险**。

**工程量**：S-M（1-2 天）。**收益**：用户可以 tap 短录指令也可 hold 长录长段，且容错 keyup 丢失。
**验证**：tap < 200ms 触发 toggle，hold ≥ 200ms 触发 PTT，release after toggle 不影响录音；模拟 release event 丢失，toggle 状态仍能用第二次 tap 停。

引用：[Superwhisper docs](https://superwhisper.com/docs/get-started/settings-shortcuts)，[VoiceInk 1.72 release](https://github.com/Beingpax/VoiceInk/releases)。

---

### M3 — Qwen3-ASR `context=` system-prompt biasing（最高 ROI）

**做什么**：放弃 sherpa-onnx hotwords（≤16/60 entry silent-fail bug 已知，[issue #2307](https://github.com/k2-fsa/sherpa-onnx/issues/2307)），改走 `mlx-qwen3-asr` 的 `context=` 字段：

```
context = "User often dictates these terms: Claude, Tauri, Rust, MLX, OKR, "
        + "宝迈健身, 智服平台, ... (50+ custom_words rendered as comma-separated)"
```

Qwen3-ASR 训练时就喂过 context-biasing data（[Qwen3-ASR Tech Report](https://arxiv.org/html/2601.21337v1)），不是软提示。capacity 实测 ~1500 token 安全（LOGIC paper 数据 460+ entity 开始衰减）。

**工程量**：S（1-2 天）。**收益**：30-60% relative entity-WER 改善（论文数据），结构性绕开 sherpa cap。
**验证**：用 M0 的 10 段样本，专名识别准确率 +30pp 以上才算 pass。

引用：[mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)，[Qwen3-ASR Tech Report arxiv 2601.21337](https://arxiv.org/html/2601.21337v1)。

---

### M4 — Rolling history context priming

**做什么**：v2t 已存最近 5 条 history。把它们 join 到 M3 的 context 字符串里（最近 60s 内的，TTL，避免跨主题污染）：

```
context = M3_hotwords + "\n\nRecent transcripts:\n"
        + last_3_finals_within_60s.join("\n")
```

加一个 "session reset" hotkey（Esc 或 cmd+shift+r）让用户主动清 context 缓存。

**工程量**：M（1-2 天）。**收益**：10-25% relative WER on names across multi-utterance sessions（关键于 zheng 的专名痛点）。
**验证**：录 5 段连续相关内容，第 2-5 段专名识别准确率比独立段平均提升 ≥10%。

引用：[Wispr Flow Llama fine-tune approach](https://www.baseten.co/resources/customers/wispr-flow/)。

---

### M5 — Silero VAD v4 → v5 升级（drop-in）

**做什么**：替换 `src-tauri/resources/models/silero_vad_v4.onnx` → `silero_vad_v5.onnx`，调整 frame size 256/512 的 API 参数（v4 是 480/1536）。

**工程量**：S（半天）。**收益**：3× JIT 速度，10-20pp TPR 改善（噪声环境）。
**注意**：v5 start-of-speech latency 比 v4 高 ~320ms，对一字一顿短指令场景需 A/B test。如果 zheng 实测变差就回退。
**验证**：录"一字一顿"短指令，VAD 不切到中间字。

引用：[Silero VAD v5 release](https://github.com/snakers4/silero-vad/discussions/471)。

---

### M6 — Qwen3-ASR 0.6B-8bit → 1.7B-4bit（模型升级）

**做什么**：切 default model id `qwen3-asr-06b-8bit` → `qwen3-asr-17b-4bit`。M-series 上仍 < 0.5s/10s 推理。LibriSpeech 0.6B WER 2.29% → 1.7B 1.99%（13% rel reduction），中文 + 噪声 + 口音场景预期更大。

**工程量**：S（半天，主要是 mlx 模型量化 + 验证）。**收益**：10-25% rel WER on noisy / 专名 / code-switching。
**注意**：M2 Air 上 1.7B-4bit 推理速度需实测，可能需保留 0.6B-8bit 作为低端机 fallback。
**验证**：M0 的 10 段样本 first-word latency 不超 1.3× baseline，错字率降 10% 以上。

引用：[mlx-qwen3-asr README](https://github.com/moona3k/mlx-qwen3-asr/)。

---

### M7 — NSTextInputClient marked-text 注入 PoC（流式视觉效果不靠 backspace）

**做什么**：用 macOS Accessibility API 抓 frontmost app 的 focused element，如果它实现了 `NSTextInputClient` 协议（绝大多数 Cocoa/Electron/Tauri 都实现），调 `setMarkedText:selectedRange:replacementRange:` 推 partial（屏幕上以 underline 显示），`insertText:` commit final。

替代当前的 enigo CGEvent + clipboard paste 链路，**视觉错字回退由 macOS IME compose region 原生处理，不再 backspace+retype**。

**工程量**：M（3-5 天，主要是 objc-rs bridge + AX 元素抓取 + Tauri 集成）。**收益**：(a) 错字回退视觉完美，(b) 不破坏 undo stack，(c) Electron / 部分 Java app / 终端 raw-mode 不再丢字。
**风险**：AX 权限弹窗、个别 app 不实现 NSTextInputClient（fallback 到现有 enigo 路径）。
**验证**：在 Notes / Cursor / Slack / Safari / Terminal 各做一次，partial underline + final commit 视觉无缝；第一个不实现 NSTextInputClient 的目标 app 自动降级到 enigo。

引用：[manaflow-ai/cmux PR #1410](https://github.com/manaflow-ai/cmux/pull/1410)，[10xChengTu/input0](https://github.com/10xChengTu/input0)。

**注意**：**不是注册 IMK**！（IMK App Store 禁、用户须手动切换到该 IME 才生效，所有第三方 voice-input app 都不走 IMK）。仅做"NSTextInputClient 客户端使用方"。

---

### M8 — Paraformer-zh-streaming on sherpa-onnx PoC（真流式 backbone）

**做什么**：在 sherpa-onnx Rust crate v1.13+ 上跑 [funasr/paraformer-zh-streaming](https://huggingface.co/funasr/paraformer-zh-streaming)，cache-based 增量推理，chunk_size=[0,8,4] = 480ms first-partial。建独立 LoadedEngine variant，不替换 Qwen3-ASR-MLX，作为新的 preset 选项。

写 PoC 评估：(a) 中文准确率 vs Qwen3-ASR-MLX baseline，(b) first-partial latency 是否真的 480ms，(c) 错字回退由 chunk-attention 内置 lookahead 处理后的 effective WER。

**工程量**：M（1-2 周，sherpa-onnx Rust binding + cache 状态机 + IPC schema）。**收益**：480ms first-partial 真流式，模型架构内置 endpoint detection，不需要 v2t 自己做 chunking。
**风险**：sherpa-onnx hotwords 已知坑（v2t 已踩过），但 hotwords 已经在 M3 走 Qwen3-ASR context 路径解决，Paraformer 这条只走基础识别。
**验证**：first-partial < 600ms，500 字段中文 WER 不输 Qwen3-ASR-1.7B + M3 biasing。

引用：[funasr/paraformer-zh-streaming](https://huggingface.co/funasr/paraformer-zh-streaming)，[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)，[SCAMA paper arxiv 2006.01712](https://arxiv.org/abs/2006.01712)。

---

### M9 — 决策与默认 backend 选型

**做什么**：基于 M8 PoC 数据 + M3-M7 已上线效果，决定 v2t 默认 ASR backend：

- **保留 Qwen3-ASR-MLX-1.7B + context biasing + post-process** — batch 路径但准确率最高
- **或切 Paraformer-zh-streaming** — 真流式 480ms 但中文识别质量待验证
- **或双 backend 共存** — 用户可在 settings 选 "Streaming" / "Accurate"

写 `docs/dev/fd-007/decision.md` 记录数据 + 选型理由。

**工程量**：S（1 天，主要是 settings UI + preset 机制扩展）。
**验证**：默认 backend 切换流畅，用户能在两 preset 间切换不丢配置。

---

## 不做什么（明确排除项）

1. **不做 IMK 注册** — sandbox 地狱、App Store 禁、用户教育成本高、所有第三方 voice-input app 都没走、UX 反而变差（用户必须先切换到 v2t IME）
2. **不做 chunked streaming 重启动** — FD-006 的 1.5s 切片伪流式已永久放弃，M8 走的是真流式 backbone（Paraformer 的 chunk-attention 是模型内部机制，不是我们外挂切片）
3. **不做 toggle-only 模式** — M2 Hybrid 已自动覆盖 toggle 用法
4. **不做 IMK + AX 双路混合** — M7 仅做"NSTextInputClient 客户端"，不做 IMK 注册
5. **不引入 LOGIC logit-space biasing** — 仅在超过 500 hotwords 才有 ROI，v2t 当前 50 entry，不重造
6. **不做 Wispr 式 LLM fine-tune** — 用户量小，per-user fine-tune 工程基础设施投入不划算
7. **不引入 Parakeet/Canary** — 无中文支持，明确 disqualified

---

## 实施分组（建议时间线）

**Sprint A（本周内，0.5-1 周）— 短跑，收稳定基线**：
- M0 baseline audit（必须先做）
- M1 handy_keys 修两 bug
- M2 Hybrid 触发
- M3 Qwen3-ASR context biasing
- M5 Silero VAD v5

**Sprint B（下周，1-2 周）— 中跑，准确率深耕**：
- M4 rolling history context priming
- M6 Qwen3-ASR 1.7B 模型升级
- M7 NSTextInputClient 注入 PoC

**Sprint C（之后，1-2 周）— 真流式探索**：
- M8 Paraformer-zh-streaming sherpa PoC
- M9 决策

每个 M 独立 commit、独立 deploy、独立可回滚。M3 + M5 + M1 即使其他 M 不做也是显著改善。

---

## Must-Haves（成功判据）

FD-007 全部完成后，v2t 应满足：

1. **first-word latency p50 ≤ 500ms**（M8 真流式达成）或 batch 路径 p50 ≤ 1000ms（M6 升级后）
2. **专名识别准确率提升 ≥ 30pp**（M3 + M4 协同）
3. **0 次 push-to-talk keyup 卡死**（M1 + M2 双保险）
4. **错字回退视觉无缝**（M7 NSTextInputClient marked-text 在主流 app 工作）
5. **可逆**：任一 M 失败可独立回滚不影响其他

---

## 非可行项 / 已排除路径备忘

调研中考察过、明确排除：

- 苹果 SpeechAnalyzer macOS 26+：[gotcha 已记](../../../.claude/projects/-Users-zhengma-Developer-handy/memory/gotcha_apple_speechanalyzer_macos26.md) — installedLocales 不可信、preRunRecognition fatalError 不能 catch、Tauri 不抓 Swift print
- Whisper Large V3 Turbo：HF 模型卡明确说"Chinese is outlier with poor performance"
- Voxtral Mini 4B Realtime：Mistral 2025-Q1 发布、claims streaming、但**无中文 + zh-en code-switching 公开 benchmark**，不投基于盲推
- Phi-4-multimodal：无中文 benchmark、非 MLX-native、不投
- WhisperKit ANE 路径：中文质量已被 Argmax 自家 benchmark 标为"regression vs Qwen3-ASR"

---

## 参考资料汇总

ASR 真流式：
- [Paraformer-zh-streaming HuggingFace](https://huggingface.co/funasr/paraformer-zh-streaming)
- [sherpa-onnx Rust crate](https://docs.rs/sherpa-onnx/latest/sherpa_onnx/)
- [SCAMA 论文 arxiv 2006.01712](https://arxiv.org/abs/2006.01712)
- [WhisperFlow 实时流式 arxiv 2412.11272](https://arxiv.org/pdf/2412.11272)

准确率 / context biasing：
- [Qwen3-ASR Tech Report arxiv 2601.21337](https://arxiv.org/html/2601.21337v1)
- [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)
- [LOGIC arxiv 2601.15397](https://arxiv.org/html/2601.15397v1)
- [Whisper prompting guide](https://cookbook.openai.com/examples/whisper_prompting_guide)

CGEventTap / IME：
- [Handy Issue #840](https://github.com/cjpais/Handy/issues/840)（commit 624579e regression 定位）
- [Hammerspoon libeventtap.m](https://github.com/Hammerspoon/hammerspoon/blob/master/extensions/eventtap/libeventtap.m)
- [cmux PR #1410 NSTextInputClient marked-text](https://github.com/manaflow-ai/cmux/pull/1410)
- [10xChengTu/input0](https://github.com/10xChengTu/input0)
- [Apple DictationIM(8)](https://keith.github.io/xcode-man-pages/DictationIM.8.html)

中国厂商协议：
- [腾讯云 ASR WebSocket](https://cloud.tencent.com/document/product/1093/48982)
- [讯飞 IAT API（wpgs append/replace）](https://www.xfyun.cn/doc/asr/voicedictation/API.html)
- [阿里 Paraformer 流式](https://help.aliyun.com/zh/model-studio/websocket-for-paraformer-real-time-service)

VAD：
- [Silero v5 release](https://github.com/snakers4/silero-vad/discussions/471)
- [Picovoice VAD benchmark 2026](https://picovoice.ai/blog/best-voice-activity-detection-vad/)

---

🐳 zheng 看到此 FD 后请就 Sprint 顺序拍板，或对任一 M 提出删除/合并/拆分意见。M0/M1/M3/M5 是建议无脑先做的"低工程量高 ROI"四件套。
