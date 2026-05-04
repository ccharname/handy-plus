# M4 perf baseline

## Status: pending zheng to run M0+M1+M2+M3+M4 build for 1 day

重新跑：

```bash
./scripts/handy-logs.sh breakdown --since 7d > docs/dev/m4-baseline.md
```

再回填本文档。

## 架构性能优化（已在 M4 commit 落地，不依赖运行时数据）

- **VAD session reuse** — Qwen3 chunking 和 FunASR-Nano pre-filter 路径统一使用 `inference_vad: Arc<Mutex<Option<SileroVad>>>` 缓存，避免每次 transcription 重建 onnxruntime session（冷启动 5–30 ms）。`acquire_inference_vad()` 每次调用重置 LSTM state（h_tensor/c_tensor）确保独立分类。
- **Resampler reuse** — `FrameResampler` 已经随 AudioRecorder worker thread 生命周期存活（每次录音不重建），确认为 cache_hit=true 路径。
- **Lazy model load** — 启动时 engine=None；仅在 `initiate_model_load()` 触发时（第一次 start_recording）按需加载对应 preset 模型。startup → tray 路径不触发任何模型加载。
- **mmap audit** — whisper.cpp 通过 `WhisperContextParameters` 默认启用 OS mmap（内核 page cache 映射）；sherpa-onnx (SenseVoice/Qwen3) 的 onnxruntime session 通过文件路径创建，ort 默认使用 OS mmap 加载 .onnx 文件。两条路径均已 mmap，无需修改。
- **clippy sweep** — 37 warnings → 0 warnings（全部修复或加 #[allow] 注释）。
- **Memory cap** — transcription 历史在 SQLite DB 存储，`cleanup_by_count` + `cleanup_by_time` 已有双重清理机制，default limit=5；in-process buffer 按录音会话生命周期管理，每次 Start 命令清空。
- **t0_hotkey dispatch** — handler.rs 路径无 sync I/O：settings 读取 Mutex<HashMap>（< 1 µs），TranscriptionCoordinator.send_input 走 channel，ACTION_MAP 是 static。已确认 non-blocking。
- **Metal/Vulkan build flags** — 见 M4 commit message。

## SLA hard gates（待数据后 verify）

见 `./scripts/handy-logs.sh assert ...` 命令组（FD-001 M4 verify 节）。
