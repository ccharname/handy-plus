# ASR Benchmark Corpus

Chinese speech benchmark corpus for evaluating FunASR-Nano, SenseVoice, and other ASR engines via CER (Character Error Rate).

## Quick stats

| Metric | Value |
|--------|-------|
| Total utterances | 36 |
| Total duration | ~3.8 min |
| Format | 16kHz mono PCM_16 WAV |
| Languages | Mandarin (+ 4 dialect samples) |

## Sources

### 1. sherpa-onnx FunASR-Nano test_wavs (16 files, prefix `sn_`)

- **License**: Apache 2.0
- **Origin**: [FunAudioLLM/FunAudioLLM.github.io](https://github.com/FunAudioLLM/FunAudioLLM.github.io/tree/master/funasr/static/audios), mirrored at [csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30](https://huggingface.co/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30)
- **Local path**: `~/Library/Application Support/com.pais.handy/models/sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/`
- **Content**: Mandarin conversations, lyrics, RAG knowledge-domain sentences, and dialect samples (Hunan, Minnan, Shanghai, Cantonese)
- **Redistribution**: Permitted under Apache 2.0

### 2. AISHELL-1 speaker S0002 (20 files, prefix `as1_`)

- **License**: CC BY-SA 4.0 — attribution required, share-alike
- **Origin**: [AISHELL/AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) (openslr.org/33)
- **Transcript source**: `data_aishell/transcript/aishell_transcript_v0.8.txt`
- **Content**: News/finance domain, studio-quality read speech, single female speaker (S0002), utterances 2–15 seconds
- **Redistribution**: Permitted under CC BY-SA 4.0 with attribution

## Tag schema

| Tag | Meaning | Count |
|-----|---------|-------|
| `pure_zh` | Standard Mandarin | 33 |
| `tech_terms` | Finance/science/policy terminology | 19 |
| `numbers_dates` | Contains numerals, dates, or counts | 7 |
| `long` | ≥10 seconds | 3 |
| `dialect` | Non-Mandarin variety (Cantonese / Shanghainese / Minnan / Hunan) | 4 |

Note: `zh_en_mix` is not represented because both source datasets are Mandarin-only. To add code-switching samples, see the ASCEND dataset (CAiRE/ASCEND on HuggingFace, CC BY 4.0).

## Provenance

Full per-file provenance (source dataset, original utterance ID) is in `_provenance.jsonl`. This file is not included in the public corpus — regenerate by running `_download.py`.

## Re-generating the corpus

Prerequisites:
```bash
pip install soundfile huggingface_hub
brew install ffmpeg   # for ffprobe
```

Then run:
```bash
python3 tests/fixtures/asr-corpus/_download.py
```

The script:
1. Copies the 16 Chinese files from the local sherpa-onnx model directory (requires the FunASR-Nano model to be downloaded first via the app).
2. Downloads speaker S0002 from AISHELL-1 on HuggingFace (~35 MB), extracts 20 utterances, and converts to 16kHz mono PCM_16.

## License compliance

- **Apache 2.0 files** (`sn_*`): Can be freely committed, redistributed, and used commercially.
- **CC BY-SA 4.0 files** (`as1_*`): Can be redistributed with attribution. Derivative datasets must also be CC BY-SA 4.0. Academic/benchmark use is fully permitted.
- **No sensitive content**: All audio is from public speech corpora with explicit dataset licenses. No personally identifying information beyond what is present in the original datasets.

## Content note

Audio contains no offensive content. The AISHELL-1 samples are read-speech news text. The sherpa-onnx samples include conversational speech, song lyrics, and academic knowledge sentences.
