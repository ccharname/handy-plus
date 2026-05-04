# M4 perf after

## Status: pending M4 build 跑一天后回填

重新跑：

```bash
./scripts/handy-logs.sh breakdown --since 1h > docs/dev/m4-after.md
./scripts/handy-logs.sh compare --baseline docs/dev/m4-baseline.md --current docs/dev/m4-after.md
```

再回填本文档，diff 对比 baseline 和 after 各 stage 占比变化。
