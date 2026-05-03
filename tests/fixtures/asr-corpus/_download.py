#!/usr/bin/env python3
"""
Chinese ASR Benchmark Corpus Builder — tests/fixtures/asr-corpus/

Sources:
  1. sherpa-onnx FunASR-Nano test_wavs (Apache 2.0)
     https://huggingface.co/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30
     — already on disk after the model is downloaded.  16kHz mono, 16-bit PCM.

  2. AISHELL-1 speaker S0002 (CC BY-SA 4.0, openslr.org/33)
     https://huggingface.co/datasets/AISHELL/AISHELL-1
     — ~35 MB single-speaker tarball, 16kHz mono, 16-bit PCM.
     Transcripts from aishell_transcript_v0.8.txt (character-spaced → joined).

Usage:
  python3 tests/fixtures/asr-corpus/_download.py

Requirements:
  pip install soundfile huggingface_hub
  ffprobe (homebrew: brew install ffmpeg)

Output: tests/fixtures/asr-corpus/*.wav  +  manifest.jsonl  +  _provenance.jsonl
"""

import json
import os
import re
import subprocess
import tempfile
import tarfile
import soundfile as sf
import numpy as np

CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))

SHERPA_WAVS_DIR = os.path.expanduser(
    "~/Library/Application Support/com.pais.handy/"
    "models/sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs"
)

# ── Sherpa-onnx: curated Chinese entries with ground-truth refs ────────────────
SHERPA_ENTRIES = [
    ("dia_hunan.wav",      "但总来讲孙膑对兵法的理解运用比庞涓略胜一筹。",      ["pure_zh","dialect"]),
    ("dia_minnan.wav",     "嗯，下摆若有机会吧，因为即久吼开了吼卷啊遮厉害，会倒贴钱啊。", ["dialect"]),
    ("dia_sh.wav",         "人跟狗，包括人跟动物接触长了，全有感情。葛末随了阿拉社会个富裕。", ["dialect"]),
    ("dia_yue.wav",        "啲身体好劲啊，跟住咧佢哋有一个人咧就突然可能就有高原反应啦，突然间就啊窒息咗，即系晕晕咗。", ["dialect"]),
    ("far_2.wav",          "然后被冠以了渣男线的称号，好了，不管这个，那么前方即将到达沈杜公路站，左边是8号线。", ["pure_zh","numbers_dates"]),
    ("far_4.wav",          "唯一的遗憾就是他那个八宝鸭还有烤鸭都没吃上，估计得提前预定吧，只能怪我自己没有做好功课。", ["pure_zh"]),
    ("far_5.wav",          "别紧张，我只是在这边逛街，然后看到你们在这边拍照，想跟你交个朋友认识一下。", ["pure_zh"]),
    ("lyrics.wav",         "我看到我的身后盯着我的人群，喜欢或恨不一样的神情，我知道这可能就是所谓的成名，我知道必须往前一步也不能停。", ["pure_zh","long"]),
    ("lyrics_2.wav",       "明明那么远，为何却感觉离他那么近？闭上眼，你甚至能背出他所有押韵。虽然不听说唱了，但你已学会自信。我代表所有中文说唱歌手向你致敬。如今面对困难的你，早已不再抱怨。", ["pure_zh","long"]),
    ("lyrics_3.wav",       "你听啊秋末的落叶，你听它叹息着离别，只剩我独自领略海与山风和月，你听啊。", ["pure_zh","long"]),
    ("rag_biochemistry.wav","利用三磷酸腺苷的水解所产生的能量来驱动其他化学反应。",  ["pure_zh","tech_terms"]),
    ("rag_chemistry.wav",  "比如说酯在当时被认为是一种含氧酸盐。",              ["pure_zh","tech_terms"]),
    ("rag_history.wav",    "由罗马皇帝钦点的犹地亚王大希律王统治期间。",          ["pure_zh"]),
    ("rag_math.wav",       "对微分形式的积分是微分几何中的基本概念。",            ["pure_zh","tech_terms"]),
    ("rag_medical.wav",    "肾脏中肾小球囊上的细胞膜孔隙很小。",               ["pure_zh","tech_terms"]),
    ("rag_physics.wav",    "根据碰撞理论月面样本缺少挥发性物质。",              ["pure_zh","tech_terms"]),
]

AISHELL_DATASET = "AISHELL/AISHELL-1"
AISHELL_SPEAKER = "S0002"
AISHELL_TARGET  = 20   # utterances to include


def ensure_16k_mono(src_path, dst_path):
    data, sr = sf.read(src_path, dtype="int16", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype("int16")
    sf.write(dst_path, data, 16000, subtype="PCM_16")


def wav_duration(path):
    r = subprocess.run(
        ["ffprobe","-v","quiet","-show_streams","-select_streams","a", path],
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        if line.startswith("duration="):
            try:
                return float(line.split("=")[1])
            except ValueError:
                pass
    return 0.0


def assign_tags_aishell(text, duration):
    tags = ["pure_zh"]
    if re.search(r'[一二三四五六七八九十百千万亿零\d]{2,}|[年月日号点分秒]', text):
        tags.append("numbers_dates")
    if re.search(r'政策|金融|经济|货币|贷款|公积金|三磷酸腺苷|微分|肾小球|碰撞|限购|银行|住房', text):
        tags.append("tech_terms")
    if duration >= 10.0:
        tags.append("long")
    return list(dict.fromkeys(tags))


def cleanup_placeholders():
    for fname in ["01_pure_zh.wav","02_zh_en_mix.wav","03_numbers_dates.wav",
                  "04_long_sentence.wav","05_tech_terms.wav"]:
        p = os.path.join(CORPUS_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"  removed placeholder: {fname}")


def copy_sherpa_wavs():
    entries = []
    for fname, ref, tags in SHERPA_ENTRIES:
        src = os.path.join(SHERPA_WAVS_DIR, fname)
        if not os.path.exists(src):
            print(f"  SKIP (not found — model not downloaded?): {fname}")
            continue
        dst_name = f"sn_{fname}"
        dst = os.path.join(CORPUS_DIR, dst_name)
        ensure_16k_mono(src, dst)
        entries.append({
            "wav": dst_name, "ref": ref, "tags": tags,
            "_source": "sherpa-onnx-funasr-nano/test_wavs (FunAudioLLM, Apache-2.0)",
            "_origin_id": fname,
        })
        print(f"  {dst_name}")
    return entries


def download_aishell_speaker(speaker=AISHELL_SPEAKER, n=AISHELL_TARGET):
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download transcript
        tx_path = hf_hub_download(
            AISHELL_DATASET,
            "data_aishell/transcript/aishell_transcript_v0.8.txt",
            repo_type="dataset",
            local_dir=tmpdir,
        )
        transcripts = {}
        with open(tx_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    transcripts[parts[0]] = "".join(parts[1:])

        # Download speaker tarball
        tar_path = hf_hub_download(
            AISHELL_DATASET,
            f"data_aishell/wav/{speaker}.tar.gz",
            repo_type="dataset",
            local_dir=tmpdir,
        )
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmpdir)

        wav_dir = os.path.join(tmpdir, "train", speaker)
        entries = []
        for wav_fname in sorted(os.listdir(wav_dir)):
            if not wav_fname.endswith(".wav"):
                continue
            utt_id = wav_fname.replace(".wav", "")
            if utt_id not in transcripts:
                continue
            ref = transcripts[utt_id]
            src = os.path.join(wav_dir, wav_fname)
            dur = wav_duration(src)
            if dur < 2.0 or dur > 15.0:
                continue
            dst_name = f"as1_{wav_fname}"
            dst = os.path.join(CORPUS_DIR, dst_name)
            ensure_16k_mono(src, dst)
            tags = assign_tags_aishell(ref, dur)
            entries.append({
                "wav": dst_name, "ref": ref, "tags": tags,
                "_source": f"{AISHELL_DATASET} speaker {speaker} (CC BY-SA 4.0, openslr.org/33)",
                "_origin_id": utt_id,
            })
            print(f"  {dst_name}  {dur:.1f}s  {ref[:40]}")
            if len(entries) >= n:
                break

    return entries


def write_manifest(entries):
    manifest_path = os.path.join(CORPUS_DIR, "manifest.jsonl")
    provenance_path = os.path.join(CORPUS_DIR, "_provenance.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf, \
         open(provenance_path, "w", encoding="utf-8") as pf:
        for e in entries:
            public = {"wav": e["wav"], "ref": e["ref"], "tags": e["tags"]}
            mf.write(json.dumps(public, ensure_ascii=False) + "\n")
            pf.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(entries)} entries → manifest.jsonl + _provenance.jsonl")


if __name__ == "__main__":
    print("=== ASR Corpus Builder ===\n")

    print("[0] Removing placeholder wavs …")
    cleanup_placeholders()

    print("\n[1] Copying sherpa-onnx test_wavs …")
    entries = copy_sherpa_wavs()
    print(f"  → {len(entries)} entries")

    print(f"\n[2] Downloading AISHELL-1 speaker {AISHELL_SPEAKER} from HuggingFace …")
    entries += download_aishell_speaker()
    print(f"  → {len(entries)} total entries")

    if len(entries) < 30:
        print(f"\nWARNING: only {len(entries)} entries — target is ≥30.")

    write_manifest(entries)
    print(f"\n=== Done: {len(entries)} wav files ===")
