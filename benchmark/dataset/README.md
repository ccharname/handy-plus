# Benchmark Dataset

Place your WAV files here before running the benchmark.

## WAV file requirements

- Format: 16-bit PCM, 16 kHz, mono (recommended)
- If a file uses a different sample rate, the transcription pipeline will still work but results may differ slightly from live recording
- Naming: any filename ending in `.wav` (case-insensitive)
- Recommended: 20-100 files, each 3-30 seconds long

## Optional: reference.txt

Create a `reference.txt` alongside your WAVs for WER evaluation:

```
今天天气真不错。
这是第二条测试语句。
...
```

Rules:
- One reference per line, in the same order as the WAV files sorted alphabetically by filename
- Each line corresponds to the WAV file at the same position in the sorted list
- Leave a line empty if you have no reference for that file

## Sourcing test audio

A quick way to collect 50 representative samples on macOS:

```bash
ls "$HOME/Library/Application Support/com.pais.handy/recordings/" | grep '\.wav$' | sort | head -50 | while read f; do
    cp "$HOME/Library/Application Support/com.pais.handy/recordings/$f" .
done
```

Alternatively, record short sentences yourself or use a public dataset such as
AISHELL-1 (Chinese) or LibriSpeech (English).

## Supported preset IDs

| preset_id              | Engine              | Language  |
| ---------------------- | ------------------- | --------- |
| `chinese_balanced`     | SenseVoice int8     | zh-Hans   |
| `multilingual_offline` | FunASR-Nano         | auto      |
| `apple_native`         | Apple Speech        | auto      |
