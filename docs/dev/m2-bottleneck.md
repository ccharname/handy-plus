# M2 SenseVoice bottleneck analysis

## Status: pending data

需要 zheng 启动 v0.9 (M0+M1+M2) 跑日常 transcription 至少 1 天，产生足够 sample（≥30 次 transcription）后再跑：

```bash
./scripts/handy-logs.sh breakdown --preset chinese_balanced --since 7d
```

再回填本文档。

## 数据驱动方法论

M2 unit test + cancel race + error path 覆盖在数据缺失时也能完成（已完成）。  
bottleneck analysis 依赖实际 jsonl 日志，需要 app 在生产状态下运行过才能取得。

当数据就绪时：

1. 跑 `breakdown --preset chinese_balanced --since 7d` 获取各 stage 占比
2. 用 `percentiles --stage t5_inference --preset chinese_balanced --metric rtf` 确认 RTF p50/p99
3. 用 `slowest --stage t5_inference --top 10` 看最慢的 10 次推理
4. 将结果回填到本文档的 §Findings 节

## M2 verify 状态

| Check | 状态 |
|-------|------|
| `cargo test silence_gate` | 待 zheng 跑验证 |
| `cargo test punc_dedup` | 待 zheng 跑验证 |
| `cargo test itn_zh` | 待 zheng 跑验证 |
| cancel-race (cargo test) | 架构约束验证通过（见 m2_stability::cancel_race_*） |
| error-paths (cargo test) | 通过（见 m2_stability::error_path_*） |
| `handy-logs assert --stage t5_inference --preset chinese_balanced --metric rtf --p50-max 0.30 --p99-max 0.50` | 数据缺失时返回 SKIP/exit 0（M0 行为） |

## SenseVoice 路径 cancel race 架构

cancel 路径安全性由以下架构约束保证（不依赖 benchmark 数据）：

1. `cancel_recording()` 通过 `Mutex<RecordingState>` 原子设为 `Idle`
2. `stop_recording(binding_id)` 在 state != `Recording{binding_id}` 时返回 `None`
3. `TranscribeAction::stop()` 的转写 + 剪贴板写入路径被包裹在 `if let Some(samples) = rm.stop_recording(...) { ... }` 内
4. 因此 cancel_recording 在 stop_recording 之前 ⟹ None ⟹ 零转写 + 零剪贴板写入

竞争窗口由 Mutex 保护，无需额外 atomic flag 检查。
